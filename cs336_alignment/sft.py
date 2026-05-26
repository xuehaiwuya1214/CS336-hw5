from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """返回可用于 padding 的 token id；Qwen 系列通常用 eos 作为 pad。"""
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    将 prompt 与 response 拼接成 causal LM 训练样例。

    返回的三个张量都已经右移对齐：
    - input_ids: 拼接序列去掉最后一个 token，作为模型输入。
    - labels: 拼接序列去掉第一个 token，作为 next-token 目标。
    - response_mask: 与 labels 对齐，只在 response token 位置为 True。

    注意这里不添加 BOS/EOS 等 special tokens。作业测试和 SFT 数据都把
    prompt/response 当作普通文本片段处理，然后由 padding 对齐 batch。
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")
    if not prompt_strs:
        raise ValueError("At least one prompt/output pair is required.")

    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]
    full_token_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids)
    ]

    max_len = max(len(token_ids) for token_ids in full_token_ids)
    if max_len < 2:
        raise ValueError("Each prompt/output pair must contain at least two tokens.")

    pad_id = _pad_token_id(tokenizer)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []

    for prompt_ids, output_ids, token_ids in zip(
        prompt_token_ids, output_token_ids, full_token_ids
    ):
        padded = token_ids + [pad_id] * (max_len - len(token_ids))

        # full_response_mask 标记原始拼接序列中的 response token。
        # labels[i] 预测的是 padded[i + 1]，所以最后也要同步右移。
        full_response_mask = [False] * len(padded)
        response_start = len(prompt_ids)
        response_end = response_start + len(output_ids)
        for idx in range(response_start, response_end):
            full_response_mask[idx] = True

        input_rows.append(padded[:-1])
        label_rows.append(padded[1:])
        mask_rows.append(full_response_mask[1:])

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """
    计算每个 token 位置上的词表分布熵。

    对 logits 直接 softmax 再 log 容易数值不稳定；这里使用等价形式：
    H(p) = logsumexp(logits) - sum_i softmax(logits)_i * logits_i
    """
    log_normalizer = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    expected_logits = torch.sum(probs * logits, dim=-1)
    return log_normalizer - expected_logits


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """
    计算 labels 中每个 token 在 causal LM 下的条件 log-probability。

    本函数不应用 response_mask；prompt/padding 的过滤在 train step 中完成。
    """
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {"log_probs": selected_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """
    只累加 mask=True 的位置，并除以固定归一化常数。

    这个函数用于 Dr. GRPO / SFT 中的固定长度归一化，而不是按有效 token
    数量求平均。因此 denominator 是调用方传入的 normalize_constant。
    """
    if normalize_constant == 0:
        raise ValueError("normalize_constant must be non-zero.")
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    执行一个 SFT microbatch 的 loss 计算与反向传播。

    SFT loss 是 response token 上的 negative log-likelihood。这里先对每条
    样例的 response token loss 求和并除以 normalize_constant，再对 batch
    求平均，最后按 gradient_accumulation_steps 缩放并 backward。
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if normalize_constant is None:
        normalize_constant = 1.0

    per_example_loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "unscaled_loss": per_example_loss.mean().detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata


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

    该函数用于训练/评估时快速检查模型生成质量。它不会假设具体 reward_fn，
    只要求 reward_fn 返回 format_reward、answer_reward、reward 三个键。
    """
    if not (len(prompts) == len(responses) == len(ground_truths)):
        raise ValueError("prompts, responses, and ground_truths must have the same length.")

    rewards = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    if response_lengths is None:
        response_lengths_list = [len(response.split()) for response in responses]
    elif isinstance(response_lengths, Tensor):
        response_lengths_list = [int(length) for length in response_lengths.detach().cpu()]
    else:
        response_lengths_list = [int(length) for length in response_lengths]

    correct_lengths = [
        length
        for length, reward in zip(response_lengths_list, rewards)
        if reward.get("answer_reward", 0.0) == 1.0
    ]
    incorrect_lengths = [
        length
        for length, reward in zip(response_lengths_list, rewards)
        if reward.get("answer_reward", 0.0) != 1.0
    ]

    records = [
        {
            "prompt": prompt,
            "model_response": response,
            "ground_truth": ground_truth,
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
            "reward": reward["reward"],
            "response_length": length,
        }
        for prompt, response, ground_truth, reward, length in zip(
            prompts, responses, ground_truths, rewards, response_lengths_list
        )
    ]
    if token_entropies is not None:
        entropy_values = token_entropies.detach().float().cpu()
        mean_token_entropy = float(entropy_values.mean().item())
    else:
        mean_token_entropy = None

    summary = {
        "records": records,
        "mean_format_reward": mean(reward["format_reward"] for reward in rewards)
        if rewards
        else 0.0,
        "mean_answer_reward": mean(reward["answer_reward"] for reward in rewards)
        if rewards
        else 0.0,
        "mean_reward": mean(reward["reward"] for reward in rewards) if rewards else 0.0,
        "mean_token_entropy": mean_token_entropy,
        "mean_response_length": mean(response_lengths_list)
        if response_lengths_list
        else 0.0,
        "mean_correct_response_length": mean(correct_lengths)
        if correct_lengths
        else 0.0,
        "mean_incorrect_response_length": mean(incorrect_lengths)
        if incorrect_lengths
        else 0.0,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary
