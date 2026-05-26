"""
第 5 节 Expert Iteration 训练脚本。

Expert Iteration 的一次 step 做三件事：
1. 从训练题库采样一批问题，并为每题采样 G 个 responses。
2. 用 r1_zero_reward_fn 过滤出正确 responses，构造临时 SFT 数据 Dsft。
3. 用 Dsft 对当前 policy 做若干 epoch 的 SFT，然后在 validation set 上评估。

本脚本是“单个 EI 配置”的入口。不同 rollout counts / SFT epochs /
EI batch size 的实验由 scripts/run_expert_iteration_experiments.sh 分别启动。

服务器推荐：
    - policy 训练放 cuda:0
    - vLLM rollout/eval 放 cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean
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
except ImportError:
    wandb = None

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import get_response_log_probs, masked_normalize, sft_microbatch_train_step

logger = logging.getLogger(__name__)

QUESTION_KEYS = ("problem", "question", "prompt", "instruction", "input")
ANSWER_KEYS = ("expected_answer", "answer", "ground_truth", "target", "final_answer")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON array 或 JSONL 文件。/data/math 下的文件通常是 JSON array。"""
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
    """按候选字段名取第一个非空字段。"""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def load_prompt_template(path: str | Path) -> str:
    """读取 r1_zero prompt 模板。"""
    return Path(path).read_text(encoding="utf-8")


def format_prompt(problem: str, prompt_template: str) -> str:
    """把数学题填入 r1_zero prompt。"""
    return prompt_template.format(question=problem)


def normalize_math_records(
    records: list[dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, str]]:
    """
    将 train/val 数据统一为 problem/prompt/answer 三个字段。

    Expert Iteration 需要标准答案来给 rollout 打 reward。若数据里有空答案
    记录，直接跳过，比训练中途报错更适合服务器长跑。
    """
    normalized = []
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
                "prompt": format_prompt(str(problem), prompt_template),
                "answer": str(answer),
            }
        )
    if skipped:
        logger.warning("Skipped %d records without problem/answer.", skipped)
    return normalized


def select_problem_batch(
    records: list[dict[str, str]],
    batch_size: int,
    step: int,
    seed: int,
) -> list[dict[str, str]]:
    """
    为当前 EI step 采样一批问题。

    这里用 seed + step 保证每个 EI step 可复现，同时不同 step 采到的问题不同。
    """
    if batch_size > len(records):
        raise ValueError(f"EI batch size {batch_size} > dataset size {len(records)}.")
    rng = random.Random(seed + step)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[i] for i in indices[:batch_size]]


class SFTDataset(Dataset):
    """临时 Dsft 数据集。"""

    def __init__(self, records: list[dict[str, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.records[idx]


def build_sft_batch(records: list[dict[str, str]], tokenizer, max_seq_len: int) -> dict[str, torch.Tensor]:
    """将 Dsft 的 prompt/response batch 转为 SFT 训练张量。"""
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
            raise ValueError(
                "A sample has no response token after truncation. "
                "Increase --max-seq-len or lower rollout max tokens."
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

    return {
        "input_ids": torch.tensor([row[:-1] for row in padded_rows], dtype=torch.long),
        "labels": torch.tensor([row[1:] for row in padded_rows], dtype=torch.long),
        "response_mask": torch.tensor([mask[1:] for mask in padded_masks], dtype=torch.bool),
    }


def make_collate_fn(tokenizer, max_seq_len: int):
    """构造 DataLoader collate_fn。"""

    def collate(records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        return build_sft_batch(records, tokenizer=tokenizer, max_seq_len=max_seq_len)

    return collate


def init_tokenizer(model_path: str):
    """加载 tokenizer，并设置 pad token。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def init_policy(model_path: str, train_device: str, gradient_checkpointing: bool):
    """加载当前 policy。"""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy.to(train_device)
    policy.train()
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    return policy


def init_vllm(model_path: str, device: str, seed: int, gpu_memory_utilization: float):
    """初始化 vLLM；通常放在 cuda:1。"""
    from vllm import LLM

    # vLLM 不同版本中 set_random_seed 的位置不同；导入失败时跳过即可。
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


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm) -> bool:
    """尝试把 policy 权重同步到 vLLM；失败时返回 False。"""
    try:
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(policy.state_dict().items())
        return True
    except AttributeError as error:
        logger.warning("Could not hot-load policy weights into vLLM: %s", error)
        return False


def reinit_vllm_from_policy_checkpoint(policy, tokenizer, output_dir: Path, step: int, args):
    """热加载失败时，保存 policy checkpoint 并重新初始化 vLLM。"""
    eval_model_dir = output_dir / "tmp_vllm_policy" / f"ei_step_{step:02d}"
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(eval_model_dir)
    tokenizer.save_pretrained(eval_model_dir)
    return init_vllm(
        str(eval_model_dir),
        device=args.eval_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )


def _make_sampling_params(args, n: int, seed: int):
    """
    构造 vLLM SamplingParams。

    include_stop_str_in_output 让输出保留 </answer>，否则 reward 函数会因为缺少
    closing tag 产生大量 false negative。旧版 vLLM 若不支持该参数，则回退。
    """
    from vllm import SamplingParams

    kwargs = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "max_tokens": args.rollout_max_tokens,
        "min_tokens": args.rollout_min_tokens,
        "n": n,
        "seed": seed,
        "stop": ["</answer>"],
    }
    try:
        return SamplingParams(**kwargs, include_stop_str_in_output=True)
    except TypeError:
        return SamplingParams(**kwargs)


def _ensure_answer_close(response: str) -> str:
    """兼容不保留 stop string 的 vLLM 版本。"""
    if "<answer>" in response and "</answer>" not in response:
        return response + "</answer>"
    return response


def generate_rollouts_vllm(
    *,
    llm,
    policy,
    prompts: list[str],
    args,
    seed: int,
    output_dir: Path,
    step: int,
    tokenizer,
) -> list[list[str]]:
    """用 vLLM 为每个 prompt 采样 G 个 responses。"""
    if not load_policy_into_vllm_instance(policy, llm):
        llm = reinit_vllm_from_policy_checkpoint(policy, tokenizer, output_dir, step, args)
    sampling_params = _make_sampling_params(args, n=args.rollouts_per_problem, seed=seed)
    outputs = llm.generate(prompts, sampling_params)
    all_responses = []
    for output in outputs:
        responses = [_ensure_answer_close(item.text.strip()) for item in output.outputs]
        all_responses.append(responses)
    return all_responses


@torch.no_grad()
def generate_rollouts_transformers(
    *,
    policy,
    tokenizer,
    prompts: list[str],
    train_device: str,
    args,
) -> list[list[str]]:
    """
    用 Transformers 采样 rollouts。这个模式主要用于单卡 smoke test，
    完整 EI 实验建议用 vLLM。
    """
    policy.eval()
    tokenizer.padding_side = "left"
    all_responses: list[list[str]] = []
    for prompt in tqdm(prompts, desc="Rollout generate"):
        inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(train_device)
        output_ids = policy.generate(
            **inputs,
            max_new_tokens=args.rollout_max_tokens,
            min_new_tokens=args.rollout_min_tokens,
            do_sample=True,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            num_return_sequences=args.rollouts_per_problem,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        responses = [
            _ensure_answer_close(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip())
            for seq in output_ids
        ]
        all_responses.append(responses)
    tokenizer.padding_side = "right"
    policy.train()
    return all_responses


def score_and_filter_rollouts(
    problem_batch: list[dict[str, str]],
    all_responses: list[list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, float]]:
    """
    奖励函数打分，并保留 reward=1 的 question-response pair 作为 Dsft。
    """
    rollout_records = []
    correct_sft_records = []
    rewards = []
    format_rewards = []
    answer_rewards = []

    for problem_record, responses in zip(problem_batch, all_responses):
        for response_idx, response in enumerate(responses):
            metrics = r1_zero_reward_fn(
                response=response,
                ground_truth=problem_record["answer"],
            )
            rewards.append(metrics["reward"])
            format_rewards.append(metrics["format_reward"])
            answer_rewards.append(metrics["answer_reward"])
            rollout_record = {
                "problem": problem_record["problem"],
                "prompt": problem_record["prompt"],
                "ground_truth": problem_record["answer"],
                "response_index": response_idx,
                "model_response": response,
                "metrics": metrics,
            }
            rollout_records.append(rollout_record)
            if metrics["reward"] == 1.0:
                correct_sft_records.append(
                    {
                        "prompt": problem_record["prompt"],
                        "response": response,
                    }
                )

    num_rollouts = max(len(rollout_records), 1)
    rollout_metrics = {
        "rollout/num_rollouts": len(rollout_records),
        "rollout/num_correct": len(correct_sft_records),
        "rollout/reward": sum(rewards) / num_rollouts,
        "rollout/format_reward": sum(format_rewards) / num_rollouts,
        "rollout/answer_reward": sum(answer_rewards) / num_rollouts,
    }
    return rollout_records, correct_sft_records, rollout_metrics


def estimate_response_entropy(
    policy,
    tokenizer,
    records: list[dict[str, str]],
    args,
) -> float:
    """
    估计 response token entropy。

    为避免完整计算过慢，只取 Dsft 中前 --entropy-sample-size 条。
    """
    if not records:
        return 0.0
    sample = records[: args.entropy_sample_size]
    loader = DataLoader(
        SFTDataset(sample),
        batch_size=args.sft_batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_seq_len),
    )
    values = []
    policy.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(args.train_device) for key, value in batch.items()}
            output = get_response_log_probs(
                model=policy,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=True,
            )
            masked_entropy = masked_normalize(
                output["token_entropy"],
                batch["response_mask"],
                dim=None,
                normalize_constant=max(int(batch["response_mask"].sum().item()), 1),
            )
            values.append(float(masked_entropy.item()))
    policy.train()
    return mean(values) if values else 0.0


def sft_update_on_records(
    *,
    policy,
    tokenizer,
    records: list[dict[str, str]],
    optimizer,
    scheduler,
    args,
) -> dict[str, float]:
    """用本轮过滤得到的 Dsft 对 policy 做 SFT 更新。"""
    if not records:
        return {
            "sft/num_examples": 0,
            "sft/mean_loss": 0.0,
            "sft/optimizer_steps": 0,
        }

    loader = DataLoader(
        SFTDataset(records),
        batch_size=args.sft_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_seq_len),
        drop_last=False,
    )
    losses = []
    optimizer_steps = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(args.sft_epochs_per_ei_step):
        for batch in tqdm(loader, desc="EI SFT update"):
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
            losses.append(float(metadata["unscaled_loss"].item()))
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    if micro_step % args.gradient_accumulation_steps != 0:
        clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    return {
        "sft/num_examples": len(records),
        "sft/mean_loss": mean(losses) if losses else 0.0,
        "sft/optimizer_steps": optimizer_steps,
    }


@torch.no_grad()
def evaluate_transformers(policy, tokenizer, val_records: list[dict[str, str]], args) -> dict[str, float]:
    """单卡/调试用 validation evaluation。"""
    policy.eval()
    tokenizer.padding_side = "left"
    eval_records = val_records[: args.eval_limit] if args.eval_limit > 0 else val_records
    responses = []
    for start in tqdm(range(0, len(eval_records), args.eval_batch_size), desc="Eval"):
        batch = eval_records[start : start + args.eval_batch_size]
        prompts = [record["prompt"] for record in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.train_device)
        output_ids = policy.generate(
            **inputs,
            max_new_tokens=args.eval_max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        responses.extend(
            [
                tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in output_ids
            ]
        )
    tokenizer.padding_side = "right"
    policy.train()
    return summarize_eval(eval_records, responses)


def evaluate_vllm(policy, tokenizer, llm, val_records: list[dict[str, str]], args, output_dir: Path, step: int) -> dict[str, float]:
    """vLLM validation evaluation。"""
    from vllm import SamplingParams

    eval_records = val_records[: args.eval_limit] if args.eval_limit > 0 else val_records
    if not load_policy_into_vllm_instance(policy, llm):
        llm = reinit_vllm_from_policy_checkpoint(policy, tokenizer, output_dir, step, args)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.eval_max_tokens,
    )
    outputs = llm.generate([record["prompt"] for record in eval_records], sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    return summarize_eval(eval_records, responses)


def summarize_eval(eval_records: list[dict[str, str]], responses: list[str]) -> dict[str, float]:
    """汇总 validation accuracy。"""
    rewards = []
    format_rewards = []
    answer_rewards = []
    for record, response in zip(eval_records, responses):
        metrics = r1_zero_reward_fn(response=response, ground_truth=record["answer"])
        rewards.append(metrics["reward"])
        format_rewards.append(metrics["format_reward"])
        answer_rewards.append(metrics["answer_reward"])
    denom = max(len(eval_records), 1)
    return {
        "eval/num_examples": len(eval_records),
        "eval/reward": sum(rewards) / denom,
        "eval/answer_accuracy": sum(answer_rewards) / denom,
        "eval/format_accuracy": sum(format_rewards) / denom,
    }


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """保存 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(policy, tokenizer, output_dir: Path, ei_step: int) -> None:
    """保存当前 EI step 后的 checkpoint。"""
    checkpoint_dir = output_dir / "checkpoints" / f"ei_step_{ei_step:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saved checkpoint to %s", checkpoint_dir)


def setup_wandb(args) -> None:
    """初始化 wandb。"""
    if not args.use_wandb:
        return
    if wandb is None:
        raise ImportError("wandb is not installed, but --use-wandb was set.")
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.define_metric("ei_step")
    wandb.define_metric("rollout/*", step_metric="ei_step")
    wandb.define_metric("sft/*", step_metric="ei_step")
    wandb.define_metric("eval/*", step_metric="ei_step")
    wandb.define_metric("entropy/*", step_metric="ei_step")


def log_metrics(args, metrics: dict[str, Any]) -> None:
    """输出日志，并可选写 wandb。"""
    logger.info("%s", json.dumps(metrics, ensure_ascii=False))
    if args.use_wandb and wandb is not None:
        wandb.log(metrics)


def run_expert_iteration(args) -> None:
    """执行完整 Expert Iteration 实验。"""
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    setup_wandb(args)

    prompt_template = load_prompt_template(args.prompt_template_path)
    train_records = normalize_math_records(read_records(args.train_path), prompt_template)
    val_records = normalize_math_records(read_records(args.val_path), prompt_template)
    logger.info("Loaded %d train problems from %s", len(train_records), args.train_path)
    logger.info("Loaded %d val problems from %s", len(val_records), args.val_path)

    tokenizer = init_tokenizer(args.model_path)
    policy = init_policy(
        args.model_path,
        train_device=args.train_device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    total_sft_steps = estimate_total_sft_steps(args)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(args.warmup_steps, int(total_sft_steps * args.warmup_ratio)),
        num_training_steps=max(total_sft_steps, 1),
    )

    llm = None
    if args.rollout_backend == "vllm" or args.eval_backend == "vllm":
        llm = init_vllm(
            args.model_path,
            device=args.eval_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    if args.eval_at_start:
        eval_metrics = evaluate_current_policy(
            policy,
            tokenizer,
            llm,
            val_records,
            args,
            output_dir,
            0,
        )
        log_metrics(args, {"ei_step": 0, **eval_metrics})

    for ei_step in range(1, args.num_ei_steps + 1):
        logger.info("Starting EI step %d/%d", ei_step, args.num_ei_steps)
        step_dir = output_dir / f"ei_step_{ei_step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        problem_batch = select_problem_batch(
            train_records,
            batch_size=args.ei_batch_size,
            step=ei_step,
            seed=args.seed,
        )
        prompts = [record["prompt"] for record in problem_batch]
        if args.rollout_backend == "vllm":
            all_responses = generate_rollouts_vllm(
                llm=llm,
                policy=policy,
                prompts=prompts,
                args=args,
                seed=args.seed + ei_step,
                output_dir=output_dir,
                step=ei_step,
                tokenizer=tokenizer,
            )
        elif args.rollout_backend == "transformers":
            all_responses = generate_rollouts_transformers(
                policy=policy,
                tokenizer=tokenizer,
                prompts=prompts,
                train_device=args.train_device,
                args=args,
            )
        else:
            raise ValueError(f"Unknown rollout backend: {args.rollout_backend}")

        rollout_records, correct_sft_records, rollout_metrics = score_and_filter_rollouts(
            problem_batch,
            all_responses,
        )
        save_jsonl(step_dir / "rollouts.jsonl", rollout_records)
        save_jsonl(step_dir / "correct_sft.jsonl", correct_sft_records)

        entropy_value = estimate_response_entropy(policy, tokenizer, correct_sft_records, args)
        sft_metrics = sft_update_on_records(
            policy=policy,
            tokenizer=tokenizer,
            records=correct_sft_records,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        eval_metrics = evaluate_current_policy(policy, tokenizer, llm, val_records, args, output_dir, ei_step)
        metrics = {
            "ei_step": ei_step,
            **rollout_metrics,
            **sft_metrics,
            "entropy/response_token_entropy": entropy_value,
            **eval_metrics,
        }
        (step_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log_metrics(args, metrics)
        save_checkpoint(policy, tokenizer, output_dir, ei_step)

    if args.use_wandb and wandb is not None:
        wandb.finish()


def evaluate_current_policy(policy, tokenizer, llm, val_records: list[dict[str, str]], args, output_dir: Path, step: int) -> dict[str, float]:
    """按配置选择 validation backend。"""
    if args.eval_backend == "none":
        return {}
    if args.eval_backend == "vllm":
        if llm is None:
            raise ValueError("eval_backend=vllm requires vLLM.")
        return evaluate_vllm(policy, tokenizer, llm, val_records, args, output_dir, step)
    if args.eval_backend == "transformers":
        return evaluate_transformers(policy, tokenizer, val_records, args)
    raise ValueError(f"Unknown eval backend: {args.eval_backend}")


def estimate_total_sft_steps(args) -> int:
    """粗略估计 scheduler 总步数。"""
    correct_upper_bound = max(args.ei_batch_size * args.rollouts_per_problem, 1)
    micro_batches = math.ceil(correct_upper_bound / args.sft_batch_size)
    optim_steps_per_epoch = math.ceil(micro_batches / args.gradient_accumulation_steps)
    return max(optim_steps_per_epoch * args.sft_epochs_per_ei_step * args.num_ei_steps, 1)


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", default="/data/math/train.jsonl")
    parser.add_argument("--val-path", default="/data/math/val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="expert_iteration")
    parser.add_argument("--prompt-template-path", default="cs336_alignment/prompts/r1_zero.prompt")

    parser.add_argument("--num-ei-steps", type=int, default=5)
    parser.add_argument("--ei-batch-size", type=int, default=512)
    parser.add_argument("--rollouts-per-problem", type=int, default=4)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-max-tokens", type=int, default=1024)
    parser.add_argument("--rollout-min-tokens", type=int, default=4)

    parser.add_argument("--sft-epochs-per-ei-step", type=int, default=1)
    parser.add_argument("--sft-batch-size", type=int, default=1)
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
    parser.add_argument("--rollout-backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--eval-backend", choices=("none", "vllm", "transformers"), default="vllm")
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=-1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--entropy-sample-size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-assignment5-ei")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    run_expert_iteration(parse_args())
