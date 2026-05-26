"""
第 8 节 GRPO 实验入口脚手架。

这个脚本负责“单次 GRPO run”的工程外壳：

1. 读取 MATH-like train / validation 数据；
2. 加载 Qwen2.5-Math-1.5B policy 与 tokenizer；
3. 根据第 8 节实验问题选择 prompt、reward_fn、GRPOConfig；
4. 调用 cs336_alignment.grpo.grpo_train_loop；
5. 把 metrics、rollout examples、run_config、final checkpoint 写入磁盘。

当前脚本默认采用“较快正式短跑”的 off-policy GRPO 配置：

- `n_grpo_steps=50`
- `learning_rate=2e-5`
- `rollout_batch_size=256`
- `group_size=8`
- `epochs_per_rollout_batch=2`
- `loss_type=grpo_clip`
- `eval_every=10`
- `eval_limit=512`

正式服务器推荐使用 `--rollout-backend vllm --eval-backend vllm`。为兼容
新版 vLLM，这里默认采用 checkpoint-sync：每次 rollout/eval 前把当前
policy 保存到 `/data` 下的临时目录，vLLM 从该目录加载，生成完成后立即删除
临时 checkpoint，避免 30G 系统盘爆掉。

推荐 smoke test:

    PYTHONPATH=. python scripts/train_grpo.py \
      --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
      --train-path /data/math/train.jsonl \
      --val-path /data/math/val.jsonl \
      --output-dir /data/outputs/grpo/smoke \
      --experiment smoke \
      --n-grpo-steps 2 \
      --rollout-batch-size 8 \
      --group-size 2 \
      --train-batch-size 8 \
      --gradient-accumulation-steps 4 \
      --eval-every 1 \
      --eval-limit 16 \
      --sampling-max-tokens 256 \
      --eval-max-new-tokens 256
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import GRPOConfig, grpo_train_loop

logger = logging.getLogger(__name__)

QUESTION_KEYS = ("problem", "question", "prompt", "instruction", "input")
ANSWER_KEYS = ("expected_answer", "answer", "ground_truth", "target", "final_answer")

ExperimentName = Literal[
    "smoke",
    "grpo_learning_rate",
    "grpo_baselines",
    "grpo_length_normalization",
    "grpo_group_standard_deviation",
    "grpo_off_policy",
    "grpo_off_policy_sweep",
    "grpo_off_policy_clip_ablation",
    "grpo_prompt_ablation",
]


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON array 或 JSONL；/data/math 下的文件经常是 JSON array。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按候选字段取第一个非空字段。"""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def normalize_math_records(
    records: list[dict[str, Any]],
    *,
    prompt_template: str,
) -> list[dict[str, str]]:
    """
    将数据统一为 problem/prompt/answer。

    grpo_train_loop 内部也会做一次 normalize；这里提前 normalize 的价值是：
    - 让 run_config / 输出样例更可读；
    - 及早过滤缺字段样本，避免长跑中途才报错。
    """
    normalized: list[dict[str, str]] = []
    skipped = 0
    for record in records:
        problem = first_present(record, QUESTION_KEYS)
        answer = first_present(record, ANSWER_KEYS)
        if problem is None or answer is None:
            skipped += 1
            continue
        normalized.append(
            {
                "problem": str(problem),
                "prompt": prompt_template.format(question=str(problem)),
                "answer": str(answer),
            }
        )
    if skipped:
        logger.warning("Skipped %d examples without problem/answer.", skipped)
    return normalized


def init_tokenizer(model_path: str):
    """加载 tokenizer，并确保 pad token 可用。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def init_policy(model_path: str, device: str, gradient_checkpointing: bool):
    """加载 policy model。"""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy.to(device)
    policy.train()
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    return policy


def configure_data_disk_env(args) -> None:
    """
    把缓存和临时目录放到 /data，保护 30G 系统盘。

    用户也可以在 shell 中提前设置这些环境变量；这里仅在未设置时填默认值。
    """
    cache_dir = Path(args.cache_dir)
    tmp_dir = Path(args.tmp_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))


def init_vllm(model_path: str, device: str, seed: int, gpu_memory_utilization: float):
    """初始化 vLLM；兼容新旧 vLLM 的 device / worker API 差异。"""
    from vllm import LLM

    try:
        from vllm.model_executor import set_random_seed as vllm_set_random_seed
    except ImportError:
        try:
            from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed
        except ImportError:
            vllm_set_random_seed = None
    if vllm_set_random_seed is not None:
        vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    try:
        import vllm.worker.worker  # noqa: F401

        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None,
        )
    except ImportError:
        profiling_patch = nullcontext()

    kwargs = {
        "model": model_path,
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    device_index = device.split(":")[-1] if device.startswith("cuda:") else None

    with world_size_patch, profiling_patch:
        try:
            return LLM(**kwargs, device=device)
        except TypeError as error:
            if "device" not in str(error):
                raise
            old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if device_index is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            try:
                return LLM(**kwargs)
            finally:
                if old_visible_devices is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices


def shutdown_vllm(llm) -> None:
    """尽力释放 vLLM engine 和 CUDA cache。"""
    if llm is None:
        return
    candidates = [
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core", None),
    ]
    for target in candidates:
        if target is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(target, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as error:  # noqa: BLE001
                    logger.debug("Ignoring vLLM cleanup error from %s: %s", method_name, error)
                break
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_policy_for_vllm(policy, tokenizer, tmp_root: Path, tag: str) -> Path:
    """
    保存当前 policy 为临时 HF checkpoint，供 vLLM 加载。

    调用方负责在 vLLM 关闭后删除该目录。
    """
    model_dir = tmp_root / tag
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def make_sampling_params(*, n: int, temperature: float, top_p: float, min_tokens: int, max_tokens: int):
    """
    构造 vLLM SamplingParams。

    为了保留 `</answer>` 供 strict reward parser 使用，这里不把 stop 设为
    `</answer>`；生成后由 reward/parser 自行处理。若后续要严格“第二个
    </answer> 停止”，可以在这里按服务器 vLLM 版本加入 stop/include_stop_str。
    """
    from vllm import SamplingParams

    kwargs = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("min_tokens", None)
        return SamplingParams(**kwargs)


def resolve_prompt_and_reward(args) -> tuple[str, Any, str]:
    """
    根据实验问题选择 prompt 与 reward function。

    第 8 节 prompt ablation 要求：
    - R1-Zero：r1_zero.prompt + r1_zero_reward_fn
    - question-only：question_only.prompt + question_only_reward_fn
    """
    if args.experiment == "grpo_prompt_ablation" and args.prompt_variant == "question_only":
        prompt_path = args.question_only_prompt_path
        return Path(prompt_path).read_text(encoding="utf-8"), question_only_reward_fn, prompt_path

    prompt_path = args.prompt_template_path
    return Path(prompt_path).read_text(encoding="utf-8"), r1_zero_reward_fn, prompt_path


def build_config(args) -> GRPOConfig:
    """把命令行参数收束为 GRPOConfig。"""
    if args.experiment == "grpo_off_policy_clip_ablation" and args.loss_type == "grpo_no_clip":
        raise NotImplementedError(
            "grpo_off_policy_clip_ablation requires adding a new loss_type "
            "'grpo_no_clip' in cs336_alignment.grpo before running this setting."
        )

    return GRPOConfig(
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_memory_utilization=args.gpu_memory_utilization,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        max_grad_norm=args.max_grad_norm,
    )


def validate_experiment_args(args) -> None:
    """
    第 8 节各问题的关键约束检查。

    这里只检查“必须成形”的设置；真正 sweep 的取值范围由外层脚本或手动命令控制。
    """
    if args.experiment == "grpo_baselines" and args.loss_type not in {
        "no_baseline",
        "reinforce_with_baseline",
    }:
        raise ValueError("grpo_baselines should compare no_baseline and reinforce_with_baseline.")

    if args.experiment in {
        "grpo_off_policy",
        "grpo_off_policy_sweep",
        "grpo_off_policy_clip_ablation",
    }:
        if args.loss_type != "grpo_clip":
            raise ValueError("Off-policy GRPO experiments should use loss_type=grpo_clip.")
        if args.epochs_per_rollout_batch <= 1 and args.train_batch_size == args.rollout_batch_size:
            logger.warning(
                "This looks on-policy. For off-policy sweeps, increase "
                "epochs_per_rollout_batch or reduce train_batch_size."
            )

    if args.experiment == "grpo_prompt_ablation" and args.prompt_variant not in {
        "r1_zero",
        "question_only",
    }:
        raise ValueError("prompt_variant must be r1_zero or question_only.")

    if args.length_normalization != "masked_mean":
        raise NotImplementedError(
            "length_normalization=masked_normalize is required by the assignment ablation, "
            "but grpo_microbatch_train_step currently implements masked_mean only. "
            "Add a normalize_constant path before running this experiment."
        )

    if not str(args.output_dir).startswith("/data/"):
        logger.warning(
            "output_dir is not under /data. The server system disk is only 30G; "
            "prefer /data/outputs/... for formal runs."
        )
    if not str(args.tmp_model_dir).startswith("/data/"):
        logger.warning(
            "tmp_model_dir is not under /data. vLLM checkpoint-sync writes a "
            "temporary full model per rollout/eval; use /data/tmp/... to avoid "
            "filling the system disk."
        )


def build_rollout_and_eval_fns(args):
    """
    构造 rollout/eval 后端。

    Transformers 后端返回 None，让 grpo_train_loop 使用内部默认实现。
    vLLM 后端采用 checkpoint-sync：保存当前 policy -> vLLM 加载 -> 生成
    -> shutdown -> 删除临时 checkpoint。这样慢一些，但和之前 SFT 服务器
    经验一致，最稳，也不会持续占用系统盘。
    """
    if args.rollout_backend == "transformers" and args.eval_backend == "transformers":
        return None, None

    tmp_root = Path(args.tmp_model_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    def rollout_fn(*, policy, tokenizer, prompts, group_size, config, step):
        if args.rollout_backend != "vllm":
            raise ValueError(f"Unsupported rollout backend: {args.rollout_backend}")

        model_dir = save_policy_for_vllm(
            policy=policy,
            tokenizer=tokenizer,
            tmp_root=tmp_root,
            tag=f"rollout_step_{step:06d}",
        )
        llm = None
        try:
            llm = init_vllm(
                model_path=str(model_dir),
                device=args.rollout_device,
                seed=args.seed + int(step),
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            sampling_params = make_sampling_params(
                n=group_size,
                temperature=config.sampling_temperature,
                top_p=args.top_p,
                min_tokens=config.sampling_min_tokens,
                max_tokens=config.sampling_max_tokens,
            )
            outputs = llm.generate(prompts, sampling_params)
            responses: list[str] = []
            for output in outputs:
                responses.extend(candidate.text.strip() for candidate in output.outputs)
            return responses
        finally:
            shutdown_vllm(llm)
            shutil.rmtree(model_dir, ignore_errors=True)

    def eval_fn(*, policy, tokenizer, reward_fn, validation_examples, config, step):
        if args.eval_backend != "vllm":
            raise ValueError(f"Unsupported eval backend: {args.eval_backend}")

        eval_records = validation_examples[: args.eval_limit] if args.eval_limit > 0 else validation_examples
        if not eval_records:
            return {}

        model_dir = save_policy_for_vllm(
            policy=policy,
            tokenizer=tokenizer,
            tmp_root=tmp_root,
            tag=f"eval_step_{step:06d}",
        )
        llm = None
        try:
            llm = init_vllm(
                model_path=str(model_dir),
                device=args.rollout_device,
                seed=args.seed + 10_000 + int(step),
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            sampling_params = make_sampling_params(
                n=1,
                temperature=0.0,
                top_p=1.0,
                min_tokens=0,
                max_tokens=args.eval_max_new_tokens,
            )
            prompts = [record["prompt"] for record in eval_records]
            outputs = llm.generate(prompts, sampling_params)
            responses = [output.outputs[0].text.strip() for output in outputs]

            rewards = []
            format_rewards = []
            answer_rewards = []
            for record, response in zip(eval_records, responses):
                metrics = reward_fn(response, record["answer"])
                rewards.append(float(metrics["reward"]))
                format_rewards.append(float(metrics.get("format_reward", 0.0)))
                answer_rewards.append(float(metrics.get("answer_reward", 0.0)))

            denom = max(len(eval_records), 1)
            return {
                "eval/num_examples": float(len(eval_records)),
                "eval/reward": sum(rewards) / denom,
                "eval/format_reward": sum(format_rewards) / denom,
                "eval/answer_reward": sum(answer_rewards) / denom,
            }
        finally:
            shutdown_vllm(llm)
            shutil.rmtree(model_dir, ignore_errors=True)

    return rollout_fn if args.rollout_backend == "vllm" else None, eval_fn if args.eval_backend == "vllm" else None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_outputs(args, result: dict[str, Any], policy, tokenizer) -> None:
    """保存本次 run 的指标、rollouts、summary 和最终 checkpoint。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "metrics.jsonl", result["metrics"])
    write_jsonl(output_dir / "rollout_examples.jsonl", result["rollout_examples"])

    summary = {
        "num_metrics": len(result["metrics"]),
        "num_rollout_examples": len(result["rollout_examples"]),
        "optimizer_steps": result["optimizer_steps"],
        "derived": result["derived"],
        "final_metrics": result["metrics"][-1] if result["metrics"] else {},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.save_final_checkpoint:
        checkpoint_dir = output_dir / "checkpoints" / "final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)


def run_final_full_eval(args, policy, tokenizer, reward_fn, val_records: list[dict[str, str]]) -> None:
    """
    训练结束后跑一次完整 validation。

    为节省系统盘，只写 summary，不保存 5000 条逐样本 responses。
    """
    if not args.final_full_eval:
        return

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "final_full_eval.summary.json"

    if args.eval_backend == "vllm":
        tmp_root = Path(args.tmp_model_dir)
        model_dir = save_policy_for_vllm(
            policy=policy,
            tokenizer=tokenizer,
            tmp_root=tmp_root,
            tag="final_full_eval",
        )
        llm = None
        try:
            llm = init_vllm(
                model_path=str(model_dir),
                device=args.rollout_device,
                seed=args.seed + 999_999,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            sampling_params = make_sampling_params(
                n=1,
                temperature=0.0,
                top_p=1.0,
                min_tokens=0,
                max_tokens=args.eval_max_new_tokens,
            )
            prompts = [record["prompt"] for record in val_records]
            outputs = llm.generate(prompts, sampling_params)
            responses = [output.outputs[0].text.strip() for output in outputs]
        finally:
            shutdown_vllm(llm)
            shutil.rmtree(model_dir, ignore_errors=True)
    else:
        # Transformers full eval 只作为调试 fallback；正式服务器建议用 vLLM。
        responses = []
        was_training = policy.training
        policy.eval()
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            for record in val_records:
                inputs = tokenizer([record["prompt"]], return_tensors="pt", padding=True).to(args.train_device)
                output_ids = policy.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.eval_max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_len = inputs["input_ids"].shape[1]
                responses.append(tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip())
        finally:
            tokenizer.padding_side = old_padding_side
            if was_training:
                policy.train()

    rewards = []
    format_rewards = []
    answer_rewards = []
    for record, response in zip(val_records, responses):
        metrics = reward_fn(response, record["answer"])
        rewards.append(float(metrics["reward"]))
        format_rewards.append(float(metrics.get("format_reward", 0.0)))
        answer_rewards.append(float(metrics.get("answer_reward", 0.0)))

    denom = max(len(val_records), 1)
    summary = {
        "num_examples": len(val_records),
        "eval/reward": sum(rewards) / denom,
        "eval/format_reward": sum(format_rewards) / denom,
        "eval/answer_reward": sum(answer_rewards) / denom,
        "eval_max_new_tokens": args.eval_max_new_tokens,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="grpo")
    parser.add_argument(
        "--experiment",
        choices=[
            "smoke",
            "grpo_learning_rate",
            "grpo_baselines",
            "grpo_length_normalization",
            "grpo_group_standard_deviation",
            "grpo_off_policy",
            "grpo_off_policy_sweep",
            "grpo_off_policy_clip_ablation",
            "grpo_prompt_ablation",
        ],
        default="grpo_off_policy",
    )

    parser.add_argument("--prompt-template-path", default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--question-only-prompt-path", default="cs336_alignment/prompts/question_only.prompt")
    parser.add_argument("--prompt-variant", choices=("r1_zero", "question_only"), default="r1_zero")

    parser.add_argument("--n-grpo-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--loss-type",
        choices=("no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"),
        default="grpo_clip",
    )
    parser.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--length-normalization",
        choices=("masked_mean", "masked_normalize"),
        default="masked_mean",
    )

    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--rollout-device", default="cuda:1")
    parser.add_argument("--rollout-backend", choices=("transformers", "vllm"), default="vllm")
    parser.add_argument("--eval-backend", choices=("transformers", "vllm"), default="vllm")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)

    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--final-full-eval", action="store_true")
    parser.add_argument("--num-logged-rollouts", type=int, default=16)
    parser.add_argument("--cache-dir", default="/data/cache")
    parser.add_argument("--tmp-dir", default="/data/tmp")
    parser.add_argument("--tmp-model-dir", default="/data/tmp/grpo_vllm_policy")

    args = parser.parse_args()
    validate_experiment_args(args)
    return args


def main() -> None:
    args = parse_args()
    configure_data_disk_env(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))

    prompt_template, reward_fn, prompt_path = resolve_prompt_and_reward(args)
    config = build_config(args)

    run_config = vars(args) | {
        "resolved_prompt_path": prompt_path,
        "reward_fn": reward_fn.__name__,
        "grpo_config": config.__dict__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_records = normalize_math_records(
        read_records(args.train_path),
        prompt_template=prompt_template,
    )
    val_records = normalize_math_records(
        read_records(args.val_path),
        prompt_template=prompt_template,
    )
    logger.info("Loaded %d train records and %d validation records.", len(train_records), len(val_records))

    tokenizer = init_tokenizer(args.model_path)
    policy = init_policy(
        model_path=args.model_path,
        device=args.train_device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_fn, eval_fn = build_rollout_and_eval_fns(args)

    metrics_path = output_dir / "metrics.jsonl"

    def log_fn(metrics: dict[str, float]) -> None:
        logger.info("%s", json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    result = grpo_train_loop(
        policy=policy,
        reward_fn=reward_fn,
        train_examples=train_records,
        validation_examples=val_records,
        config=config,
        tokenizer=tokenizer,
        optimizer=optimizer,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
        log_fn=log_fn,
        seed=args.seed,
        device=args.train_device,
        prompt_template=prompt_template,
        eval_every=args.eval_every,
        eval_limit=args.eval_limit,
        eval_max_new_tokens=args.eval_max_new_tokens,
        max_seq_len=args.max_seq_len,
        num_logged_rollouts=args.num_logged_rollouts,
        cliprange=args.cliprange,
        top_p=args.top_p,
    )

    run_final_full_eval(args, policy, tokenizer, reward_fn, val_records)
    save_outputs(args, result, policy, tokenizer)
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
