"""
SFT 组件脚手架。

本文件对应 task-explaination/4.2-4.3 sft-tasks.md 中 4.2 的函数组件。
它只保留接口、类型标注、返回约定和实现提示；核心逻辑保留为 TODO。

正式可运行实现见：

    cs336_alignment/sft.py

建议学习顺序：

1. 先读这个 scaffold，弄清楚每个函数输入输出是什么；
2. 再看 tests/test_sft.py，理解测试在检查什么；
3. 最后对照 cs336_alignment/sft.py 中的完整实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _not_implemented(name: str) -> None:
    """统一抛出 TODO，避免 scaffold 函数静默返回 None。"""
    raise NotImplementedError(f"{name} is a scaffold. Fill in the TODO implementation.")


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """
    返回可用于 padding 的 token id。

    TODO:
        1. 优先使用 tokenizer.pad_token_id。
        2. 如果 pad_token_id 不存在，可以退回 tokenizer.eos_token_id。
        3. 两者都不存在时抛出 ValueError。
    """
    _not_implemented("_pad_token_id")


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    将 prompt 与 response 拼接成 causal LM 训练样例。

    Args:
        prompt_strs: batch 内每条样例的 prompt 文本。
        output_strs: batch 内每条样例的 response / completion 文本。
        tokenizer: Hugging Face tokenizer。

    Returns:
        一个包含三个张量的 dict：
        - input_ids: `(batch_size, sequence_length)`，拼接序列去掉最后一个 token。
        - labels: `(batch_size, sequence_length)`，拼接序列去掉第一个 token。
        - response_mask: `(batch_size, sequence_length)`，只在 response token 对齐后为 True。

    注意：
        causal LM 训练时，模型看 `input_ids[:, i]`，预测 `labels[:, i]`。
        因此 response_mask 也必须和 labels 对齐，而不是和原始 full sequence 对齐。

    TODO:
        1. 检查 prompt_strs 与 output_strs 长度一致且非空。
        2. 分别 tokenize prompt 和 output，通常 add_special_tokens=False。
        3. 拼接 prompt_ids + output_ids。
        4. padding 到 batch 内最大长度。
        5. 构造 input_ids = padded[:-1]，labels = padded[1:]。
        6. 构造右移后的 response_mask，只让 response token 参与 loss。
    """
    _not_implemented("tokenize_prompt_and_output")


def compute_entropy(logits: Tensor) -> Tensor:
    """
    计算每个 token 位置上词表分布的熵。

    Args:
        logits: `(batch_size, sequence_length, vocab_size)`。

    Returns:
        `(batch_size, sequence_length)`，每个位置的 entropy。

    TODO:
        1. 用 logsumexp 得到 log normalizer。
        2. 用 softmax 得到概率。
        3. 使用 H(p)=logsumexp(logits)-sum_i p_i*logits_i。
    """
    _not_implemented("compute_entropy")


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """
    计算 labels 中每个 token 在 causal LM 下的条件 log-probability。

    Args:
        model: causal LM policy。
        input_ids: `(batch_size, sequence_length)`。
        labels: `(batch_size, sequence_length)`，每个位置要预测的 token id。
        return_token_entropy: 是否额外返回每个位置的 token entropy。

    Returns:
        - log_probs: `(batch_size, sequence_length)`，labels 对应 token 的 log-prob。
        - token_entropy: 可选，`(batch_size, sequence_length)`。

    TODO:
        1. 调用 model(input_ids)，取 logits。
        2. 对 vocab 维度做 log_softmax。
        3. 用 torch.gather 取出 labels 对应 token 的 log-prob。
        4. 如果需要 entropy，调用 compute_entropy。
    """
    _not_implemented("get_response_log_probs")


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """
    只累加 mask=True 的位置，并除以固定归一化常数。

    Args:
        tensor: 待聚合张量。
        mask: 与 tensor 同形状的 bool / 0-1 mask。
        dim: 沿哪个维度求和；None 表示所有维度。
        normalize_constant: 固定归一化常数。

    Returns:
        `(tensor * mask).sum(dim) / normalize_constant`。

    注意：
        这个函数不是普通 masked mean。它用于 Dr. GRPO / SFT 中的固定长度
        归一化，因此 denominator 是调用方传入的 normalize_constant。

    TODO:
        1. 检查 normalize_constant 非 0。
        2. 将 mask 转为 tensor.dtype。
        3. 对 masked tensor 求和并除以 normalize_constant。
    """
    _not_implemented("masked_normalize")


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    执行一个 SFT microbatch 的 loss 计算与 backward。

    Args:
        policy_log_probs: `(batch_size, sequence_length)`，每个 label token 的 log-prob。
        response_mask: `(batch_size, sequence_length)`，response token 为 True。
        gradient_accumulation_steps: 梯度累积步数；loss backward 前需要除以它。
        normalize_constant: 每条样例 response loss 的固定归一化常数。

    Returns:
        loss: 标量 tensor，已经除以 gradient_accumulation_steps，并已 backward。
        metadata: 建议包含 unscaled_loss、num_response_tokens 等调试信息。

    TODO:
        1. 检查 gradient_accumulation_steps > 0。
        2. SFT token loss 是 -policy_log_probs。
        3. 用 masked_normalize 在 response token 上聚合每条样例的 loss。
        4. 对 batch 取均值，再除以 gradient_accumulation_steps。
        5. 调用 loss.backward()。
        6. 返回 loss 和 detached metadata。
    """
    _not_implemented("sft_microbatch_train_step")


def log_generations(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    token_entropies: Tensor | None = None,
    response_lengths: list[int] | Tensor | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    汇总并可选保存 generation 记录。

    Args:
        prompts: 输入 prompts。
        responses: 模型生成 responses。
        ground_truths: 标准答案。
        reward_fn: 返回 format_reward / answer_reward / reward 的函数。
        token_entropies: 可选 token entropy，用于分析生成不确定性。
        response_lengths: 可选 response 长度；若不传，可用简单规则估计。
        output_path: 可选 JSON 输出路径。

    Returns:
        一个 summary dict，通常包含：
        - records: 每条样例的 prompt / response / reward。
        - mean_format_reward
        - mean_answer_reward
        - mean_reward
        - mean_token_entropy
        - mean_response_length
        - mean_correct_response_length
        - mean_incorrect_response_length

    TODO:
        1. 检查 prompts / responses / ground_truths 长度一致。
        2. 对每条 response 调 reward_fn。
        3. 统计平均 reward、长度、entropy。
        4. 若 output_path 不为 None，写出 JSON。
        5. 返回 summary。
    """
    _not_implemented("log_generations")
