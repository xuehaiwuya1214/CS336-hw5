"""
4.3 SFT 训练脚本。

这个脚本只负责“单次训练实验”。不同数据规模和 filtered/full 对比实验
由 scripts/run_sft_experiments.sh 调用本脚本多次完成，这样每个实验都是
独立进程，服务器显存释放更干净，也更容易断点重跑。

最小 smoke test:
    python scripts/train_sft.py \
        --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B \
        --train-path /data/math/sft_gpt-oss-120b.jsonl \
        --val-path /data/math/val.json \
        --output-dir outputs/sft/smoke \
        --num-train-examples 16 \
        --max-steps 5 \
        --eval-every 5 \
        --eval-limit 16 \
        --eval-backend transformers

服务器推荐：训练模型放 cuda:0，vLLM 验证放 cuda:1。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import wandb
except ImportError:  # wandb 是建议项，不应阻塞本地 smoke test。
    wandb = None

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step

logger = logging.getLogger(__name__)

QUESTION_KEYS = ("problem", "question", "prompt", "instruction", "input")
ANSWER_KEYS = ("expected_answer", "answer", "ground_truth", "target", "final_answer")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """
    读取 JSON array 或 JSONL。

    你的 /data/math 文件虽然命名为 .jsonl，但内容可能是以 [ 开头的 JSON
    array。这里自动识别两种格式，避免服务器上因为扩展名踩坑。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return data

    records = []
    for line in text.splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按候选字段顺序取第一个非空字段。"""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def load_prompt_template(path: str | Path) -> str:
    """读取 r1_zero prompt 模板。"""
    return Path(path).read_text(encoding="utf-8")


def format_math_prompt(problem: str, prompt_template: str) -> str:
    """把数学题填入 r1_zero prompt。"""
    return prompt_template.format(question=problem)


def normalize_sft_records(
    records: list[dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, str]]:
    """
    将不同来源的 SFT 数据统一成 {"prompt": str, "response": str}。

    官方 SFT 数据通常已经有 prompt/response。你的替代数据有
    problem/reasoning_trace，因此这里会自动把 problem 套入 r1_zero prompt。
    """
    normalized = []
    for record in records:
        prompt = record.get("prompt")
        if prompt is None:
            problem = first_present(record, QUESTION_KEYS)
            if problem is None:
                raise KeyError(f"Cannot find problem/prompt field in keys: {sorted(record)}")
            prompt = format_math_prompt(str(problem), prompt_template)

        response = record.get("response")
        if response is None:
            response = record.get("reasoning_trace")
        if response is None:
            raise KeyError(f"Cannot find response/reasoning_trace field in keys: {sorted(record)}")

        normalized.append({"prompt": str(prompt), "response": str(response)})
    return normalized


def normalize_val_records(
    records: list[dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, str]]:
    """将 validation 数据统一成 prompt/problem/answer 三个字段。"""
    normalized = []
    for record in records:
        problem = first_present(record, QUESTION_KEYS)
        answer = first_present(record, ANSWER_KEYS)
        if problem is None or answer is None:
            raise KeyError(f"Validation record missing problem or answer: {sorted(record)}")
        normalized.append(
            {
                "problem": str(problem),
                "prompt": format_math_prompt(str(problem), prompt_template),
                "answer": str(answer),
            }
        )
    return normalized


def select_train_records(
    records: list[dict[str, str]],
    num_train_examples: int,
    seed: int,
) -> list[dict[str, str]]:
    """选择训练样例；-1 或 0 表示使用全量。"""
    if num_train_examples is None or num_train_examples <= 0:
        return records
    if num_train_examples > len(records):
        raise ValueError(
            f"Requested {num_train_examples} examples, but dataset only has {len(records)}."
        )
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = sorted(indices[:num_train_examples])
    return [records[i] for i in selected]


class SFTDataset(Dataset):
    """简单的 prompt/response SFT Dataset。"""

    def __init__(self, records: list[dict[str, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.records[idx]


def build_sft_batch(
    records: list[dict[str, str]],
    tokenizer,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """
    将一个 batch 的 prompt/response 转成 input_ids/labels/response_mask。

    这里和 4.2 的 tokenize_prompt_and_output 逻辑一致，但增加了 max_seq_len
    截断，防止服务器训练时遇到超长样例导致 OOM。
    """
    prompt_token_ids = [
        tokenizer.encode(record["prompt"], add_special_tokens=False) for record in records
    ]
    response_token_ids = [
        tokenizer.encode(record["response"], add_special_tokens=False) for record in records
    ]

    full_rows = []
    full_masks = []
    for prompt_ids, response_ids in zip(prompt_token_ids, response_token_ids):
        full_ids = (prompt_ids + response_ids)[:max_seq_len]
        response_mask = [False] * len(full_ids)
        response_start = min(len(prompt_ids), len(full_ids))
        for idx in range(response_start, len(full_ids)):
            response_mask[idx] = True
        if len(full_ids) < 2 or not any(response_mask[1:]):
            # 极端长 prompt 会挤掉 response。保留一个无损样例不如直接报错，
            # 因为没有 response token 的样例对 SFT loss 没贡献。
            raise ValueError(
                "A sample has no trainable response tokens after truncation. "
                "Increase --max-seq-len or filter long prompts."
            )
        full_rows.append(full_ids)
        full_masks.append(response_mask)

    pad_id = tokenizer.pad_token_id
    max_len = max(len(row) for row in full_rows)
    padded_rows = []
    padded_masks = []
    for row, mask in zip(full_rows, full_masks):
        pad_len = max_len - len(row)
        padded_rows.append(row + [pad_id] * pad_len)
        padded_masks.append(mask + [False] * pad_len)

    input_ids = torch.tensor([row[:-1] for row in padded_rows], dtype=torch.long)
    labels = torch.tensor([row[1:] for row in padded_rows], dtype=torch.long)
    response_mask = torch.tensor([mask[1:] for mask in padded_masks], dtype=torch.bool)
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def make_collate_fn(tokenizer, max_seq_len: int):
    """构造 DataLoader collate_fn。"""

    def collate(records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        return build_sft_batch(records, tokenizer=tokenizer, max_seq_len=max_seq_len)

    return collate


def init_policy(model_path: str, train_device: str, gradient_checkpointing: bool):
    """加载 policy model。"""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(train_device)
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def init_tokenizer(model_path: str):
    """加载 tokenizer，并确保存在 pad token。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def init_vllm(model_path: str, device: str, seed: int, gpu_memory_utilization: float):
    """
    初始化 vLLM，用于 validation generation。

    这段逻辑来自作业文档。vLLM 放在独立 GPU 时，device 通常传 cuda:1。
    """
    from vllm import LLM

    # vLLM 不同版本中 set_random_seed 的位置变过。它只影响可复现性，
    # 不应该因为导入路径变化阻塞训练。
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

    # 旧版 vLLM 需要 patch 这个 profiling 检查；新版 vLLM 已经没有
    # vllm.worker.worker 这个路径。create=True 不能创建不存在的父模块，
    # 所以这里先探测模块，失败就使用 no-op context。
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
    # vLLM 新版本不再接受 device=；旧版本接受。若不支持 device，
    # 通过临时 CUDA_VISIBLE_DEVICES 让 vLLM 只看到目标 GPU。
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


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm) -> bool:
    """
    尝试把当前 policy 权重热加载到 vLLM instance 中。

    vLLM 内部对象路径在不同版本差异很大。旧版可以直接访问
    llm_engine.model_executor.driver_worker...；新版可能没有该属性。
    成功返回 True，失败返回 False，由调用方决定是否 fallback。
    """
    try:
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(policy.state_dict().items())
        return True
    except AttributeError as error:
        logger.warning("Could not hot-load policy weights into vLLM: %s", error)
        return False


def reinit_vllm_from_policy_checkpoint(
    *,
    policy,
    tokenizer,
    output_dir: Path,
    step: int,
    eval_device: str,
    seed: int,
    gpu_memory_utilization: float,
):
    """
    vLLM 热加载失败时的兼容 fallback：保存当前 policy 到临时目录，
    再从该目录重新初始化 vLLM。速度慢一些，但跨 vLLM 版本更稳。
    """
    eval_model_dir = output_dir / "tmp_vllm_policy" / f"step_{step:06d}"
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(eval_model_dir)
    tokenizer.save_pretrained(eval_model_dir)
    llm = init_vllm(
        model_path=str(eval_model_dir),
        device=eval_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return llm, eval_model_dir


def shutdown_vllm(llm) -> None:
    """
    尽量释放 vLLM 相关资源。

    vLLM 的内部 API 在不同版本里变化很快，所以这里只做 best-effort：
    如果当前版本暴露了 shutdown/close 就调用；没有的话依赖对象析构和
    CUDA cache 清理。这样不会因为清理接口不存在而影响主流程。
    """
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
                    logger.debug("Ignoring vLLM %s cleanup error: %s", method_name, error)
                break

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def generate_with_policy(
    policy,
    tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """不用 vLLM 时，直接用当前 policy 做慢速 validation generation。"""
    policy.eval()
    responses = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Eval generate"):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            output_ids = policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            for sequence in output_ids:
                responses.append(
                    tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
                )
    finally:
        tokenizer.padding_side = old_padding_side
    policy.train()
    return responses


def generate_with_vllm(llm, prompts: list[str], max_new_tokens: int) -> list[str]:
    """用 vLLM 做 validation generation。"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def evaluate_policy(
    *,
    step: int,
    policy,
    tokenizer,
    val_records: list[dict[str, str]],
    output_dir: Path,
    train_device: str,
    eval_backend: str,
    llm,
    eval_limit: int,
    eval_batch_size: int,
    eval_max_new_tokens: int,
    eval_device: str,
    seed: int,
    vllm_gpu_memory_utilization: float,
    vllm_sync_mode: str,
) -> dict[str, float]:
    """在 validation set 上评估当前 policy，并把明细和 summary 写到磁盘。"""
    if eval_limit and eval_limit > 0:
        eval_records = val_records[:eval_limit]
    else:
        eval_records = val_records
    prompts = [record["prompt"] for record in eval_records]

    if eval_backend == "none":
        return {}
    if eval_backend == "vllm":
        if vllm_sync_mode == "checkpoint" or llm is None:
            # 新版 vLLM 不一定支持从训练进程直接热加载权重。checkpoint 模式
            # 每次评估把当前 policy 存成临时 HF checkpoint，再让 vLLM 从该
            # checkpoint 启动；慢一点，但最稳。
            eval_llm, eval_model_dir = reinit_vllm_from_policy_checkpoint(
                policy=policy,
                tokenizer=tokenizer,
                output_dir=output_dir,
                step=step,
                eval_device=eval_device,
                seed=seed,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
            try:
                responses = generate_with_vllm(
                    eval_llm,
                    prompts,
                    max_new_tokens=eval_max_new_tokens,
                )
            finally:
                shutdown_vllm(eval_llm)
                del eval_llm
                shutil.rmtree(eval_model_dir, ignore_errors=True)
        else:
            if not load_policy_into_vllm_instance(policy, llm):
                raise RuntimeError(
                    "当前 vLLM 版本不支持脚本里的热加载路径。请使用 "
                    "--vllm-sync-mode checkpoint 重新运行。"
                )
            responses = generate_with_vllm(llm, prompts, max_new_tokens=eval_max_new_tokens)
    elif eval_backend == "transformers":
        responses = generate_with_policy(
            policy=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            device=train_device,
            max_new_tokens=eval_max_new_tokens,
            batch_size=eval_batch_size,
        )
    else:
        raise ValueError(f"Unknown eval backend: {eval_backend}")

    records = []
    num_format = 0
    num_answer = 0
    num_reward = 0
    for val_record, response in zip(eval_records, responses):
        metrics = r1_zero_reward_fn(response=response, ground_truth=val_record["answer"])
        num_format += int(metrics["format_reward"])
        num_answer += int(metrics["answer_reward"])
        num_reward += int(metrics["reward"])
        records.append(
            {
                "problem": val_record["problem"],
                "prompt": val_record["prompt"],
                "ground_truth": val_record["answer"],
                "model_response": response,
                "metrics": metrics,
            }
        )

    denom = max(len(eval_records), 1)
    summary = {
        "eval_step": step,
        "num_examples": len(eval_records),
        "eval/format_accuracy": num_format / denom,
        "eval/answer_accuracy": num_answer / denom,
        "eval/reward": num_reward / denom,
    }

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"step_{step:06d}.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    (eval_dir / f"step_{step:06d}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def save_checkpoint(policy, tokenizer, output_dir: Path, step: int) -> None:
    """保存 HuggingFace checkpoint。"""
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saved checkpoint to %s", checkpoint_dir)


def setup_wandb(args) -> None:
    """按作业建议初始化 wandb；未启用时无动作。"""
    if not args.use_wandb:
        return
    if wandb is None:
        raise ImportError("wandb is not installed, but --use-wandb was set.")
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def log_metrics(args, metrics: dict[str, Any]) -> None:
    """记录到日志和 wandb。"""
    logger.info("%s", json.dumps(metrics, ensure_ascii=False))
    if args.use_wandb and wandb is not None:
        wandb.log(metrics)


def train_one_run(args) -> None:
    """执行一次完整 SFT 实验。"""
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_wandb(args)

    prompt_template = load_prompt_template(args.prompt_template_path)
    raw_train_records = read_records(args.train_path)
    raw_val_records = read_records(args.val_path)
    train_records = normalize_sft_records(raw_train_records, prompt_template)
    train_records = select_train_records(
        train_records,
        num_train_examples=args.num_train_examples,
        seed=args.seed,
    )
    val_records = normalize_val_records(raw_val_records, prompt_template)

    logger.info("Loaded %d train records from %s", len(train_records), args.train_path)
    logger.info("Loaded %d validation records from %s", len(val_records), args.val_path)

    tokenizer = init_tokenizer(args.model_path)
    policy = init_policy(
        args.model_path,
        train_device=args.train_device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    train_dataset = SFTDataset(train_records)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, max_seq_len=args.max_seq_len),
        drop_last=False,
    )

    if args.max_steps > 0:
        total_optimizer_steps = args.max_steps
    else:
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_optimizer_steps = steps_per_epoch * args.num_epochs

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(int(total_optimizer_steps * args.warmup_ratio), args.warmup_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    llm = None
    if args.eval_backend == "vllm" and args.vllm_sync_mode != "checkpoint":
        llm = init_vllm(
            model_path=args.model_path,
            device=args.eval_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    args_path = output_dir / "run_config.json"
    args_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")

    optimizer.zero_grad(set_to_none=True)
    train_step = 0
    micro_step = 0
    stop_training = False

    if args.eval_at_start and args.eval_backend != "none":
        eval_metrics = evaluate_policy(
            step=0,
            policy=policy,
            tokenizer=tokenizer,
            val_records=val_records,
            output_dir=output_dir,
            train_device=args.train_device,
            eval_backend=args.eval_backend,
            llm=llm,
            eval_limit=args.eval_limit,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_device=args.eval_device,
            seed=args.seed,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_sync_mode=args.vllm_sync_mode,
        )
        log_metrics(args, {"eval_step": 0, **eval_metrics})

    for epoch in range(args.num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in progress:
            batch = {key: value.to(args.train_device) for key, value in batch.items()}
            output = get_response_log_probs(
                model=policy,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=False,
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=output["log_probs"],
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=args.normalize_constant,
            )
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                grad_norm = clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1

                metrics = {
                    "train_step": train_step,
                    "train/loss": float(metadata["unscaled_loss"].item()),
                    "train/scaled_loss": float(loss.detach().item()),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "train/num_response_tokens": int(metadata["num_response_tokens"].item()),
                }
                progress.set_postfix(loss=metrics["train/loss"], lr=metrics["train/lr"])
                if train_step % args.log_every == 0:
                    log_metrics(args, metrics)

                if (
                    args.eval_backend != "none"
                    and args.eval_every > 0
                    and train_step % args.eval_every == 0
                ):
                    eval_metrics = evaluate_policy(
                        step=train_step,
                        policy=policy,
                        tokenizer=tokenizer,
                        val_records=val_records,
                        output_dir=output_dir,
                        train_device=args.train_device,
                        eval_backend=args.eval_backend,
                        llm=llm,
                        eval_limit=args.eval_limit,
                        eval_batch_size=args.eval_batch_size,
                        eval_max_new_tokens=args.eval_max_new_tokens,
                        eval_device=args.eval_device,
                        seed=args.seed,
                        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        vllm_sync_mode=args.vllm_sync_mode,
                    )
                    log_metrics(args, {"eval_step": train_step, **eval_metrics})

                if args.save_every > 0 and train_step % args.save_every == 0:
                    save_checkpoint(policy, tokenizer, output_dir, train_step)

                if args.max_steps > 0 and train_step >= args.max_steps:
                    stop_training = True
                    break

        # 处理 epoch 末尾不足一个 gradient_accumulation_steps 的剩余梯度。
        if not stop_training and micro_step % args.gradient_accumulation_steps != 0:
            grad_norm = clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_step += 1
            log_metrics(
                args,
                {
                    "train_step": train_step,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                },
            )
            if args.max_steps > 0 and train_step >= args.max_steps:
                stop_training = True

        if stop_training:
            break

    save_checkpoint(policy, tokenizer, output_dir, train_step)
    if args.eval_backend != "none":
        eval_metrics = evaluate_policy(
            step=train_step,
            policy=policy,
            tokenizer=tokenizer,
            val_records=val_records,
            output_dir=output_dir,
            train_device=args.train_device,
            eval_backend=args.eval_backend,
            llm=llm,
            eval_limit=args.eval_limit,
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_device=args.eval_device,
            seed=args.seed,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_sync_mode=args.vllm_sync_mode,
        )
        log_metrics(args, {"eval_step": train_step, **eval_metrics})

    shutdown_vllm(llm)
    if args.use_wandb and wandb is not None:
        wandb.finish()


def set_seed(seed: int) -> None:
    """设置随机种子，保证数据抽样可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="sft")
    parser.add_argument(
        "--prompt-template-path",
        default="cs336_alignment/prompts/r1_zero.prompt",
    )

    parser.add_argument("--num-train-examples", type=int, default=-1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--normalize-constant", type=float, default=2048.0)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    parser.add_argument(
        "--eval-backend",
        choices=("none", "transformers", "vllm"),
        default="vllm",
    )
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=-1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--vllm-sync-mode",
        choices=("checkpoint", "hot_load"),
        default="checkpoint",
        help=(
            "vLLM 评估时如何同步当前训练权重。checkpoint 最兼容；"
            "hot_load 只适合旧版 vLLM 内部 API。"
        ),
    )

    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-assignment5-sft")

    args = parser.parse_args()
    if args.eval_backend == "vllm" and args.train_device == args.eval_device:
        logger.warning(
            "train_device and eval_device are both %s. "
            "For assignment-style runs, use separate GPUs if possible.",
            args.train_device,
        )
    return args


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    train_one_run(parse_args())
