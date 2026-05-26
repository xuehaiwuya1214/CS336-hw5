"""
GRPO 组件脚手架。

本文件对应 task-explaination/cs336_assignment5_7_2_GRPO_implementation.md。
这里先把 7.2 要求的函数接口、类型标注、返回约定和实现步骤搭好；核心数学
逻辑保留为 TODO，便于你后续逐个函数实现并运行 tests/test_grpo.py 验证。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output


PolicyGradientLossType = Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
]

RewardFn = Callable[[str, str], dict[str, float]]

logger = logging.getLogger(__name__)

QUESTION_KEYS = ("problem", "question", "prompt", "instruction", "input")
ANSWER_KEYS = ("expected_answer", "answer", "ground_truth", "target", "final_answer")


@dataclass(frozen=True)
class GRPOConfig:
    """
    GRPO train loop 的配置占位。

    这里放的是作业文档建议的关键超参数。完整 train loop 还会需要模型路径、
    数据路径、输出路径、评估间隔、vLLM 配置等工程参数；这些可以在真正实现
    `grpo_train_loop` 时继续扩展。
    """

    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.85
    loss_type: PolicyGradientLossType = "reinforce_with_baseline"
    use_std_normalization: bool = True
    max_grad_norm: float = 1.0


def _not_implemented(name: str) -> None:
    """统一抛出 TODO，避免空函数静默返回 None。"""
    raise NotImplementedError(f"{name} is a scaffold. Fill in the TODO implementation.")


def _require_same_shape(left: Tensor, right: Tensor, *, left_name: str, right_name: str) -> None:
    """轻量 shape 检查 helper；实现核心逻辑时可按需使用。"""
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have the same shape, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )


def _require_batch_column(tensor: Tensor, *, name: str) -> None:
    """检查 reward/advantage 张量是否为 `(batch_size, 1)`。"""
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape (batch_size, 1), got {tuple(tensor.shape)}.")


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按候选字段顺序取第一个非空字段。"""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _load_prompt_template(kwargs: dict[str, Any]) -> str | None:
    """读取 prompt 模板；调用方也可以直接传入 prompt_template。"""
    if "prompt_template" in kwargs and kwargs["prompt_template"] is not None:
        return str(kwargs["prompt_template"])

    prompt_template_path = kwargs.get("prompt_template_path")
    if prompt_template_path is None:
        default_path = Path("cs336_alignment/prompts/r1_zero.prompt")
        if default_path.exists():
            prompt_template_path = default_path

    if prompt_template_path is None:
        return None
    return Path(prompt_template_path).read_text(encoding="utf-8")


def _normalize_math_examples(
    examples: list[dict[str, Any]],
    prompt_template: str | None,
    *,
    require_answer: bool,
) -> list[dict[str, str]]:
    """
    将不同来源的数据统一为 problem/prompt/answer。

    GRPO rollout 和 evaluation 都需要 prompt；reward 计算还需要 answer。
    若样本没有 prompt，则用 problem/question 套入 r1_zero 模板。
    """
    normalized: list[dict[str, str]] = []
    skipped = 0
    for record in examples:
        prompt = record.get("prompt")
        problem = _first_present(record, QUESTION_KEYS)
        if prompt is None:
            if problem is None or prompt_template is None:
                skipped += 1
                continue
            prompt = prompt_template.format(question=str(problem))
        if problem is None:
            problem = prompt

        answer = _first_present(record, ANSWER_KEYS)
        if require_answer and answer is None:
            skipped += 1
            continue

        normalized.append(
            {
                "problem": str(problem),
                "prompt": str(prompt),
                "answer": "" if answer is None else str(answer),
            }
        )

    if skipped:
        logger.warning("Skipped %d examples without required fields.", skipped)
    return normalized


def _truncate_after_second_answer_tag(response: str) -> str:
    """
    按作业建议，在第二个 </answer> 之后截断。

    vLLM 可以在 stop strings 上停止；Transformers 简化实现中先生成再后处理。
    """
    close_tag = "</answer>"
    first = response.find(close_tag)
    if first == -1:
        return response.strip()
    second = response.find(close_tag, first + len(close_tag))
    if second == -1:
        return response[: first + len(close_tag)].strip()
    return response[: second + len(close_tag)].strip()


@torch.no_grad()
def _generate_rollouts_transformers(
    *,
    policy: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    group_size: int,
    config: GRPOConfig,
    device: torch.device,
    top_p: float,
) -> list[str]:
    """默认 rollout 生成器：用当前 policy 为每个 prompt 采样 group_size 个回答。"""
    if tokenizer is None:
        raise ValueError("tokenizer is required when rollout_fn is not provided.")

    was_training = policy.training
    policy.eval()
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    rollout_responses: list[str] = []
    try:
        for prompt in prompts:
            inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
            output_ids = policy.generate(
                **inputs,
                do_sample=True,
                temperature=config.sampling_temperature,
                top_p=top_p,
                num_return_sequences=group_size,
                min_new_tokens=config.sampling_min_tokens,
                max_new_tokens=config.sampling_max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            for sequence in output_ids:
                response = tokenizer.decode(
                    sequence[prompt_len:],
                    skip_special_tokens=True,
                )
                rollout_responses.append(_truncate_after_second_answer_tag(response))
    finally:
        tokenizer.padding_side = old_padding_side
        if was_training:
            policy.train()

    return rollout_responses


def _tokenize_grpo_batch(
    *,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    max_seq_len: int | None,
) -> dict[str, Tensor]:
    """
    将 rollout prompt/response 转成 input_ids/labels/response_mask。

    默认复用 4.2 的 tokenize_prompt_and_output；若传入 max_seq_len，则先做
    简单截断，避免长 rollout 把显存打穿。
    """
    batch = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    if max_seq_len is None or batch["input_ids"].shape[1] <= max_seq_len:
        return batch

    return {
        "input_ids": batch["input_ids"][:, :max_seq_len],
        "labels": batch["labels"][:, :max_seq_len],
        "response_mask": batch["response_mask"][:, :max_seq_len],
    }


@torch.no_grad()
def _compute_old_log_probs(
    *,
    policy: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    microbatch_size: int,
    device: torch.device,
) -> Tensor:
    """在 policy 更新前缓存 old log-probs，供 off-policy GRPO-Clip 复用。"""
    was_training = policy.training
    policy.eval()
    chunks: list[Tensor] = []
    try:
        for start in range(0, input_ids.shape[0], microbatch_size):
            end = start + microbatch_size
            output = get_response_log_probs(
                model=policy,
                input_ids=input_ids[start:end].to(device),
                labels=labels[start:end].to(device),
                return_token_entropy=False,
            )
            chunks.append(output["log_probs"].detach().cpu())
    finally:
        if was_training:
            policy.train()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _evaluate_grpo_policy(
    *,
    policy: torch.nn.Module,
    tokenizer: Any,
    reward_fn: RewardFn,
    validation_records: list[dict[str, str]],
    limit: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    """默认 validation：每题贪心生成一个回答，并汇总 reward。"""
    if tokenizer is None:
        return {}

    eval_records = validation_records[:limit] if limit > 0 else validation_records
    if not eval_records:
        return {}

    was_training = policy.training
    policy.eval()
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    try:
        for record in eval_records:
            inputs = tokenizer([record["prompt"]], return_tensors="pt", padding=True).to(device)
            output_ids = policy.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(
                output_ids[0, prompt_len:],
                skip_special_tokens=True,
            ).strip()
            metrics = reward_fn(response, record["answer"])
            rewards.append(float(metrics["reward"]))
            format_rewards.append(float(metrics.get("format_reward", 0.0)))
            answer_rewards.append(float(metrics.get("answer_reward", 0.0)))
    finally:
        tokenizer.padding_side = old_padding_side
        if was_training:
            policy.train()

    denom = max(len(eval_records), 1)
    return {
        "eval/num_examples": float(len(eval_records)),
        "eval/reward": sum(rewards) / denom,
        "eval/format_reward": sum(format_rewards) / denom,
        "eval/answer_reward": sum(answer_rewards) / denom,
    }


def compute_group_normalized_rewards(
    reward_fn: RewardFn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    计算 rollout rewards，并按 question group 得到 advantages。

    Args:
        reward_fn: 根据 `(response, ground_truth)` 返回 reward 字典的函数。
            预期至少包含 `"reward"`，通常也包含 `"format_reward"` 和
            `"answer_reward"`。
        rollout_responses: policy 生成的 responses。长度应为
            `n_prompts_per_rollout_batch * group_size`。
        repeated_ground_truths: 与 responses 一一对应的标准答案；同一个问题的
            ground truth 会重复 `group_size` 次。
        group_size: 每个 prompt 采样的 response 数量。
        advantage_eps: 标准差归一化时的数值稳定项。
        normalize_by_std: 为 True 时使用 `(r - mean) / (std + eps)`；为 False
            时只使用 `r - mean`。

    Returns:
        advantages: `(rollout_batch_size,)`，每个 response 的 group-normalized reward。
        raw_rewards: `(rollout_batch_size,)`，reward_fn 返回的原始 reward。
        metadata: 可记录 rollout batch 的 reward mean/std、format mean、answer mean 等。

    TODO:
        1. 校验 responses / ground_truths 长度一致，且能被 group_size 整除。
        2. 调用 reward_fn 得到 raw reward / format reward / answer reward。
        3. reshape 为 `(n_groups, group_size)`。
        4. 对每组减均值；按 normalize_by_std 决定是否除以组内 std。
        5. flatten 回 `(rollout_batch_size,)`，并返回 metadata。
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length, "
            f"got {len(rollout_responses)} and {len(repeated_ground_truths)}."
        )

    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size == 0:
        raise ValueError("rollout_responses must not be empty.")
    if rollout_batch_size % group_size != 0:
        raise ValueError(
            "rollout batch size must be divisible by group_size, "
            f"got rollout_batch_size={rollout_batch_size}, group_size={group_size}."
        )

    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]

    raw_rewards = torch.tensor(
        [float(score["reward"]) for score in reward_dicts],
        dtype=torch.float32,
    )

    # 每一行对应同一个 prompt 的 group_size 个 rollout。
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)

    advantages = grouped_rewards - group_means
    if normalize_by_std:
        advantages = advantages / (group_stds + advantage_eps)
    advantages = advantages.reshape(rollout_batch_size)

    format_rewards = torch.tensor(
        [float(score.get("format_reward", 0.0)) for score in reward_dicts],
        dtype=torch.float32,
    )
    answer_rewards = torch.tensor(
        [float(score.get("answer_reward", 0.0)) for score in reward_dicts],
        dtype=torch.float32,
    )

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std().item()),
        "reward_min": float(raw_rewards.min().item()),
        "reward_max": float(raw_rewards.max().item()),
        "format_reward_mean": float(format_rewards.mean().item()),
        "answer_reward_mean": float(answer_rewards.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
        "group_reward_mean_std": float(group_means.reshape(-1).std().item()),
        "group_reward_std_mean": float(group_stds.reshape(-1).mean().item()),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """
    计算 naive policy-gradient 的 per-token loss。

    Args:
        raw_rewards_or_advantages: `(batch_size, 1)`，可以是 raw rewards，也可以
            是已经 group-normalized 的 advantages。
        policy_log_probs: `(batch_size, sequence_length)`，当前 policy 在各 token
            上的 log-probability。

    Returns:
        `(batch_size, sequence_length)` 的 per-token loss。

    TODO:
        1. 校验 raw_rewards_or_advantages 是 `(batch_size, 1)`。
        2. 将它 broadcast 到 sequence 维度。
        3. 返回 `-raw_rewards_or_advantages * policy_log_probs`。
    """
    _require_batch_column(raw_rewards_or_advantages, name="raw_rewards_or_advantages")
    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape (batch_size, sequence_length), "
            f"got {tuple(policy_log_probs.shape)}."
        )
    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            "raw_rewards_or_advantages and policy_log_probs must have the same "
            f"batch size, got {raw_rewards_or_advantages.shape[0]} and "
            f"{policy_log_probs.shape[0]}."
        )

    # 同一个 response 的标量 reward/advantage 会广播到该 response 的所有 token。
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    计算 GRPO-Clip 的 per-token clipped policy-gradient loss。

    Args:
        advantages: `(batch_size, 1)`，每个 response 的 advantage。
        policy_log_probs: `(batch_size, sequence_length)`，当前 policy 的 token log-probs。
        old_log_probs: `(batch_size, sequence_length)`，旧 policy 的 token log-probs。
            注意：调用方应保证该张量不需要梯度。
        cliprange: PPO/GRPO clipping epsilon，例如 0.2。

    Returns:
        loss: `(batch_size, sequence_length)` 的 per-token clipped loss。
        metadata: 建议至少包含 `"clip_fraction"` 或 per-token clipped mask，
            供后续训练日志使用。

    TODO:
        1. 校验 policy_log_probs 和 old_log_probs 形状一致。
        2. 计算 ratio = exp(policy_log_probs - old_log_probs)。
        3. 计算 unclipped objective 和 clipped objective。
        4. 按作业公式取 `-min(unclipped, clipped)`。
        5. 返回 loss 和 clipping 相关 metadata。
    """
    _require_batch_column(advantages, name="advantages")
    _require_same_shape(
        policy_log_probs,
        old_log_probs,
        left_name="policy_log_probs",
        right_name="old_log_probs",
    )
    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape (batch_size, sequence_length), "
            f"got {tuple(policy_log_probs.shape)}."
        )
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            "advantages and policy_log_probs must have the same batch size, "
            f"got {advantages.shape[0]} and {policy_log_probs.shape[0]}."
        )

    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clip(ratio, min=1 - cliprange, max=1 + cliprange)

    # advantages 的形状是 (batch_size, 1)，会广播到每个 token。
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    loss = -torch.minimum(unclipped, clipped)
    metadata = {
        "clip_fraction": torch.mean((torch.abs(ratio - 1.0) > cliprange).float()).detach(),
        "clipped_mask": (torch.abs(ratio - 1.0) > cliprange).detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: PolicyGradientLossType,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    policy-gradient loss 的统一分发入口。

    Args:
        policy_log_probs: `(batch_size, sequence_length)`，当前 policy 的 token log-probs。
        loss_type: `"no_baseline"`、`"reinforce_with_baseline"` 或 `"grpo_clip"`。
        raw_rewards: `loss_type == "no_baseline"` 时使用。
        advantages: `loss_type in {"reinforce_with_baseline", "grpo_clip"}` 时使用。
        old_log_probs: `loss_type == "grpo_clip"` 时使用。
        cliprange: `loss_type == "grpo_clip"` 时使用。

    Returns:
        loss: `(batch_size, sequence_length)` 的 per-token loss。
        metadata: 底层 loss 函数返回的辅助统计。

    TODO:
        - no_baseline: 用 raw_rewards 调 compute_naive_policy_gradient_loss。
        - reinforce_with_baseline: 用 advantages 调 compute_naive_policy_gradient_loss。
        - grpo_clip: 调 compute_grpo_clip_loss。
        - 对缺失参数给出清晰错误信息。
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("当 loss_type 为 'no_baseline' 时，必须提供 raw_rewards。")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("当 loss_type 为 'reinforce_with_baseline' 时，必须提供 advantages。")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}

    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("当 loss_type 为 'grpo_clip' 时，必须提供 advantages, old_log_probs 和 cliprange。")
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """
    在 mask 为 True/1 的位置上计算均值。

    Args:
        tensor: 待聚合张量。
        mask: 与 tensor 同形状的布尔或 0/1 mask。
        dim: 若为 None，在所有 masked elements 上求均值；否则沿指定维度求均值。

    Returns:
        masked mean，shape 与 `tensor.mean(dim=dim)` 的语义一致。

    TODO:
        1. 校验 tensor 与 mask 形状一致。
        2. 将 mask 转成 tensor.dtype。
        3. 分子为 `(tensor * mask).sum(dim)`。
        4. 分母为 `mask.sum(dim)`。若某个位置没有任何有效元素，保持 PyTorch
           原生除以 0 行为，返回 NaN，和作业 snapshot 约定一致。
    """
    _require_same_shape(tensor, mask, left_name="tensor", right_name="mask")

    mask = mask.to(dtype=tensor.dtype)
    numerator = (tensor * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim)
    return numerator / denominator



def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: PolicyGradientLossType,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    在一个 GRPO microbatch 上计算 loss 并执行 backward。

    Args:
        policy_log_probs: `(batch_size, sequence_length)`，当前 policy log-probs。
        response_mask: `(batch_size, sequence_length)`，response token 为 True/1。
        gradient_accumulation_steps: loss backward 前需要除以的累积步数。
        loss_type: policy-gradient loss 类型。
        raw_rewards: no_baseline 所需的 raw rewards。
        advantages: reinforce_with_baseline / grpo_clip 所需的 advantages。
        old_log_probs: grpo_clip 所需的 old policy log-probs。
        cliprange: grpo_clip 所需的 clipping epsilon。

    Returns:
        loss: 标量 tensor，已经除以 gradient_accumulation_steps。
        metadata: 建议包含 unscaled_loss、num_response_tokens、clip_fraction 等。

    TODO:
        1. 调 compute_policy_gradient_loss 得到 per-token loss。
        2. 用 masked_mean 在 sequence 维度聚合到 per-example loss。
        3. 对 batch 取均值，并除以 gradient_accumulation_steps。
        4. 调用 loss.backward()。
        5. 返回 loss 与 detached metadata。
    """
    _require_same_shape(
        policy_log_probs,
        response_mask,
        left_name="policy_log_probs",
        right_name="response_mask",
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, "
            f"got {gradient_accumulation_steps}."
        )

    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    unscaled_loss = per_example_loss.mean()
    scaled_loss = unscaled_loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {
        "unscaled_loss": unscaled_loss.detach(),
        "scaled_loss": scaled_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    metadata.update(
        {
            key: value.detach() if isinstance(value, Tensor) else value
            for key, value in loss_metadata.items()
        }
    )
    return scaled_loss, metadata

def grpo_train_loop(
    *,
    policy: torch.nn.Module,
    reward_fn: RewardFn,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    config: GRPOConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    完整 GRPO train loop 的工程入口占位。

    Args:
        policy: 待训练的 policy model。
        reward_fn: rollout 评分函数。
        train_examples: 训练问题列表，后续实现中会格式化为 r1_zero prompts。
        validation_examples: validation 问题列表，用于周期性评估。
        config: GRPOConfig。
        **kwargs: 预留给 tokenizer、vLLM、optimizer、logger、output_dir 等工程对象。

    Returns:
        训练摘要字典，例如最终 validation rewards、输出路径、关键日志等。

    TODO:
        1. 根据 config 推导 micro_train_batch_size、n_prompts_per_rollout_batch。
        2. 每个 GRPO step：采样 prompts，调用 vLLM 生成 group rollouts。
        3. 调 compute_group_normalized_rewards 得到 raw rewards / advantages。
        4. tokenize prompt + response，计算 policy_log_probs。
        5. off-policy 设置下计算并缓存 old_log_probs。
        6. 分 microbatch 调 grpo_microbatch_train_step。
        7. optimizer.step、gradient clipping、日志、checkpoint、validation。
    """
    tokenizer = kwargs.get("tokenizer")
    optimizer = kwargs.get("optimizer")
    rollout_fn = kwargs.get("rollout_fn")
    eval_fn = kwargs.get("eval_fn")
    log_fn = kwargs.get("log_fn")
    seed = int(kwargs.get("seed", 0))
    top_p = float(kwargs.get("top_p", 1.0))
    eval_every = int(kwargs.get("eval_every", 10))
    eval_limit = int(kwargs.get("eval_limit", 1024))
    eval_max_new_tokens = int(kwargs.get("eval_max_new_tokens", config.sampling_max_tokens))
    max_seq_len = kwargs.get("max_seq_len")
    num_logged_rollouts = int(kwargs.get("num_logged_rollouts", 4))

    if tokenizer is None and rollout_fn is None:
        raise ValueError("tokenizer is required unless a custom rollout_fn is provided.")
    if tokenizer is None:
        raise ValueError("tokenizer is required for scoring rollout log-probabilities.")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.train_batch_size % config.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps  #TODO : 1

    if config.rollout_batch_size % config.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size

    if config.train_batch_size < config.group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size.")
    if config.train_batch_size > config.rollout_batch_size:
        raise ValueError("train_batch_size must be less than or equal to rollout_batch_size.")
    if config.rollout_batch_size % micro_train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by micro_train_batch_size.")

    n_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size  #TODO : 1

    device = kwargs.get("device")
    if device is None:
        device = next(policy.parameters()).device
    device = torch.device(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

    prompt_template = _load_prompt_template(kwargs)
    train_records = _normalize_math_examples(
        train_examples,
        prompt_template,
        require_answer=True,
    )
    validation_records = _normalize_math_examples(
        validation_examples,
        prompt_template,
        require_answer=True,
    )
    if not train_records:
        raise ValueError("No usable train examples after normalization.")

    rng = random.Random(seed)
    metrics_history: list[dict[str, float]] = []
    rollout_examples: list[dict[str, Any]] = []
    optimizer_step = 0

    derived = {
        "micro_train_batch_size": micro_train_batch_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "n_microbatches_per_rollout_batch": n_microbatches_per_rollout_batch,
    }

    def emit(metrics: dict[str, float]) -> None:
        """统一记录 metrics，方便外部接 wandb 或普通 logger。"""
        metrics_history.append(metrics)
        if log_fn is not None:
            log_fn(metrics)
        else:
            logger.info("%s", metrics)

    for grpo_step in range(1, config.n_grpo_steps + 1):    #TODO : 2
        # 每个 GRPO step 重新采样 prompt，并用当前 policy 生成 on-policy rollouts。
        if n_prompts_per_rollout_batch <= len(train_records):
            prompt_records = rng.sample(train_records, n_prompts_per_rollout_batch)
        else:
            prompt_records = rng.choices(train_records, k=n_prompts_per_rollout_batch)

        prompts = [record["prompt"] for record in prompt_records]
        ground_truths = [record["answer"] for record in prompt_records]

        if rollout_fn is None:
            rollout_responses = _generate_rollouts_transformers(
                policy=policy,
                tokenizer=tokenizer,
                prompts=prompts,
                group_size=config.group_size,
                config=config,
                device=device,
                top_p=top_p,
            )
        else:
            generated = rollout_fn(
                policy=policy,
                tokenizer=tokenizer,
                prompts=prompts,
                group_size=config.group_size,
                config=config,
                step=grpo_step,
            )
            if generated and isinstance(generated[0], list):
                rollout_responses = [response for group in generated for response in group]
            else:
                rollout_responses = list(generated)

        expected_rollouts = n_prompts_per_rollout_batch * config.group_size
        if len(rollout_responses) != expected_rollouts:
            raise ValueError(
                f"Expected {expected_rollouts} rollout responses, got {len(rollout_responses)}."
            )

        repeated_prompts = [
            prompt
            for prompt in prompts
            for _ in range(config.group_size)
        ]
        repeated_ground_truths = [
            answer
            for answer in ground_truths
            for _ in range(config.group_size)
        ]
        #TODO : 3   
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )
        #TODO : 4/5
        tokenized = _tokenize_grpo_batch(
            tokenizer=tokenizer,
            prompts=repeated_prompts,
            responses=rollout_responses,
            max_seq_len=max_seq_len,
        )
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        response_mask = tokenized["response_mask"]

        old_log_probs = None
        if config.loss_type == "grpo_clip":
            # old_log_probs 在 rollout batch 生成后、任何 policy update 前计算一次。
            # 多个 epochs_per_rollout_batch 时复用它，不对 old log-probs 求导。
            old_log_probs = _compute_old_log_probs(
                policy=policy,
                input_ids=input_ids,
                labels=labels,
                microbatch_size=micro_train_batch_size,
                device=device,
            )

        advantages = advantages.unsqueeze(1)
        raw_rewards = raw_rewards.unsqueeze(1)
        all_indices = list(range(config.rollout_batch_size))
        #TODO : 6 
        for rollout_epoch in range(config.epochs_per_rollout_batch):
            # 默认 on-policy 设置下 train_batch_size == rollout_batch_size；
            # 若用户做 ablation 使用较小 train_batch_size，则每个 epoch 随机取子集。
            epoch_indices = all_indices.copy()
            rng.shuffle(epoch_indices)
            train_indices = epoch_indices[: config.train_batch_size]

            optimizer.zero_grad(set_to_none=True)
            microbatch_metadata: dict[str, Tensor] = {}
            entropy_values: list[Tensor] = []

            for start in range(0, config.train_batch_size, micro_train_batch_size):
                batch_indices = train_indices[start : start + micro_train_batch_size]
                batch_input_ids = input_ids[batch_indices].to(device)
                batch_labels = labels[batch_indices].to(device)
                batch_response_mask = response_mask[batch_indices].to(device)

                output = get_response_log_probs(
                    model=policy,
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                    return_token_entropy=True,
                )

                old_log_probs_batch = None
                if old_log_probs is not None:
                    old_log_probs_batch = old_log_probs[batch_indices].to(device)

                loss, microbatch_metadata = grpo_microbatch_train_step(
                    policy_log_probs=output["log_probs"],
                    response_mask=batch_response_mask,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    loss_type=config.loss_type,
                    raw_rewards=raw_rewards[batch_indices].to(device),
                    advantages=advantages[batch_indices].to(device),
                    old_log_probs=old_log_probs_batch,
                    cliprange=kwargs.get("cliprange", 0.2),
                )

                entropy_values.append(
                    masked_mean(
                        output["token_entropy"].detach(),
                        batch_response_mask,
                        dim=None,
                    ).detach()
                )

            grad_norm = clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_step += 1
#TODO : 7
            metrics: dict[str, float] = {
                "grpo_step": float(grpo_step),
                "rollout_epoch": float(rollout_epoch),
                "optimizer_step": float(optimizer_step),
                "train/loss": float(microbatch_metadata["scaled_loss"].item()),
                "train/unscaled_loss": float(microbatch_metadata["unscaled_loss"].item()),
                "train/grad_norm": float(grad_norm),
                "train/token_entropy": float(torch.stack(entropy_values).mean().item()),
                "train/num_response_tokens": float(microbatch_metadata["num_response_tokens"].item()),
                "rollout/reward": float(reward_metadata["reward_mean"]),
                "rollout/reward_std": float(reward_metadata["reward_std"]),
                "rollout/format_reward": float(reward_metadata["format_reward_mean"]),
                "rollout/answer_reward": float(reward_metadata["answer_reward_mean"]),
            }
            if "clip_fraction" in microbatch_metadata:
                metrics["train/clip_fraction"] = float(microbatch_metadata["clip_fraction"].item())
            emit(metrics)

        if len(rollout_examples) < num_logged_rollouts:
            for prompt, response, answer in zip(
                repeated_prompts,
                rollout_responses,
                repeated_ground_truths,
            ):
                rollout_examples.append(
                    {
                        "grpo_step": grpo_step,
                        "prompt": prompt,
                        "response": response,
                        "ground_truth": answer,
                        "reward": reward_fn(response, answer),
                    }
                )
                if len(rollout_examples) >= num_logged_rollouts:
                    break

        if eval_every > 0 and grpo_step % eval_every == 0:
            if eval_fn is not None:
                eval_metrics = eval_fn(
                    policy=policy,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    validation_examples=validation_records,
                    config=config,
                    step=grpo_step,
                )
            else:
                eval_metrics = _evaluate_grpo_policy(
                    policy=policy,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    validation_records=validation_records,
                    limit=eval_limit,
                    max_new_tokens=eval_max_new_tokens,
                    device=device,
                )
            if eval_metrics:
                emit(
                    {
                        "grpo_step": float(grpo_step),
                        "optimizer_step": float(optimizer_step),
                        **{key: float(value) for key, value in eval_metrics.items()},
                    }
                )

    return {
        "metrics": metrics_history,
        "rollout_examples": rollout_examples,
        "config": config,
        "derived": derived,
        "optimizer_steps": optimizer_step,
    }


__all__ = [
    "GRPOConfig",
    "PolicyGradientLossType",
    "RewardFn",
    "compute_group_normalized_rewards",
    "compute_naive_policy_gradient_loss",
    "compute_grpo_clip_loss",
    "compute_policy_gradient_loss",
    "masked_mean",
    "grpo_microbatch_train_step",
    "grpo_train_loop",
]
