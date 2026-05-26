"""
GRPO 组件脚手架。

本文件对应 task-explaination/cs336_assignment5_7_2_GRPO_implementation.md。
这里先把 7.2 要求的函数接口、类型标注、返回约定和实现步骤搭好；核心数学
逻辑保留为 TODO，便于你后续逐个函数实现并运行 tests/test_grpo.py 验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor


PolicyGradientLossType = Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
]

RewardFn = Callable[[str, str], dict[str, float]]


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
    _not_implemented("compute_group_normalized_rewards")


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
    _not_implemented("compute_naive_policy_gradient_loss")


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
    _not_implemented("compute_grpo_clip_loss")


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
    _not_implemented("compute_policy_gradient_loss")


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
        4. 分母为 `mask.sum(dim)`，注意避免除 0。
    """
    _not_implemented("masked_mean")


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
    _not_implemented("grpo_microbatch_train_step")


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
    _not_implemented("grpo_train_loop")


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
