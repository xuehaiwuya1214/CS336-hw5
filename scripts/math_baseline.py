"""
在 MATH-like JSONL 数据上评估 Qwen2.5-Math 风格模型的 zero-shot baseline。

这个脚本对应作业 math_baseline 小题里的流程：
1. 读取 validation JSONL 样例。
2. 使用 r1_zero.prompt 把每道题格式化成语言模型 prompt。
3. 用 Transformers 或 vLLM 生成模型回答。
4. 用 r1_zero_reward_fn 计算 format_reward / answer_reward / reward。
5. 把每条样例、prompt、模型输出、ground truth、reward 指标写入磁盘。
6. 汇总三类结果数量，并抽样保存每类结果，方便后续写分析。

正式测试数据来源：
- 作业原版要求的 MATH validation 路径是 /data/a5-alignment/MATH/validation.jsonl。
- 如果没有访问权限，可以使用开源替代数据：
  https://huggingface.co/datasets/garg-aayush/sft-cs336-assign5-datasets
- 该仓库中的 sft-reason/val.jsonl 可作为 validation 数据。
- 推荐下载到本仓库的 data/open_math/ 目录：
    huggingface-cli download garg-aayush/sft-cs336-assign5-datasets \
        sft-reason/val.jsonl sft-reason/baseline_results.jsonl \
        --repo-type dataset \
        --local-dir data/open_math
- 下载后正式输入路径就是：
    data/open_math/sft-reason/val.jsonl

Local GSM8K test with Hugging Face Transformers:
    python scripts/math_baseline.py \
        --input-path data/gsm8k/test.jsonl \
        --output-path outputs/gsm8k_qwen25_math_15b_10.jsonl \
        --generation-mode transformers \
        --model-name-or-path /home/xuehaiwuya/models/Qwen2.5-Math-1.5B \
        --limit 10 \
        --max-tokens 512 \
        --batch-size 1

Server run:
    python scripts/math_baseline.py \
        --input-path data/open_math/sft-reason/val.jsonl \
        --output-path outputs/math_baseline_qwen25_math_15b.jsonl \
        --generation-mode vllm \
        --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
        --num-gpus 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm
from xopen import xopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


# 不同开源数据集的字段名不完全一致。这里列出常见字段名，让脚本可以
# 同时兼容原始 MATH、Hugging Face 替代数据、以及 GSM8K。
QUESTION_KEYS = ("question", "problem", "prompt", "instruction", "input")
ANSWER_KEYS = (
    "expected_answer",
    "answer",
    "ground_truth",
    "target",
    "final_answer",
    "solution",
)


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """
    读取 JSONL 或 JSON array 文件。

    一些替代数据虽然扩展名叫 .jsonl，实际内容却是以 `[` 开头的 JSON list。
    这里和 train_sft.py 保持一致，自动兼容两种格式。
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] == "[":
        examples = json.loads(text)
        if not isinstance(examples, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return examples[:limit] if limit is not None else examples

    examples = []
    for line in text.splitlines():
        if line.strip():
            examples.append(json.loads(line))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def first_present(example: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按候选字段名顺序取第一个存在且非空的值。"""
    for key in keys:
        value = example.get(key)
        if value is not None and value != "":
            return value
    return None


def get_question(example: dict[str, Any]) -> str:
    """从一条样例中取出题目文本。"""
    question = first_present(example, QUESTION_KEYS)
    if question is None:
        raise KeyError(f"Could not find a question field. Available keys: {sorted(example)}")
    return str(question)


def get_ground_truth(example: dict[str, Any]) -> str | list[str]:
    """从一条样例中取出标准答案；grader 支持字符串答案，也支持多个可接受答案。"""
    answer = first_present(example, ANSWER_KEYS)
    if answer is None:
        raise KeyError(f"Could not find an answer field. Available keys: {sorted(example)}")
    return answer


def normalize_ground_truth_for_reward(answer: str | list[str]) -> str | list[str]:
    """
    把不同数据集的标准答案整理成 reward_fn 更容易比较的形式。

    GSM8K 的 answer 字段通常是完整解析过程，最后用 "#### 18" 标出最终答案。
    对 reward 来说，我们只需要最终答案，所以这里抽取 #### 后面的部分。
    MATH-like 数据如果本来就是最终答案，则保持不变。
    """
    if isinstance(answer, list):
        return [str(item).rsplit("####", maxsplit=1)[-1].strip().replace(",", "") for item in answer]
    if isinstance(answer, str) and "####" in answer:
        return answer.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
    return answer


def load_r1_zero_template(path: str | Path) -> str:
    """读取作业提供的 r1_zero prompt 模板。"""
    return Path(path).read_text()


def format_prompt(template: str, question: str) -> str:
    """把题目填入 prompt 模板中的 {question} 占位符。"""
    return template.format(question=question)


def vllm_generate(
    prompts: list[str],
    model_name_or_path: str,
    num_gpus: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[str]:
    """
    用 vLLM 真正调用模型生成回答。

    这个模式适合放到有 GPU 的服务器上跑完整 validation 集。
    """
    from vllm import LLM, SamplingParams

    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    raw_outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in raw_outputs]


def transformers_generate(
    prompts: list[str],
    model_name_or_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
) -> list[str]:
    """
    用 Hugging Face Transformers 在本地生成回答。

    这个模式比 vLLM 更适合先在 WSL + 单张消费级显卡上做 10 条左右的链路测试。
    4060 8G 建议 batch_size=1，max_tokens 先设 512，跑通后再逐步增加。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    device_map = "auto" if has_cuda else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if not has_cuda:
        model = model.to("cpu")
    model.eval()
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    do_sample = temperature > 0.0
    responses = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        for sequence in output_ids:
            response_ids = sequence[prompt_length:]
            responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())

    return responses


def make_bucket_key(metrics: dict[str, float]) -> str:
    """把 reward 指标映射到作业要求的三类 bucket 名称。"""
    format_reward = int(metrics["format_reward"])
    answer_reward = int(metrics["answer_reward"])
    return f"format_{format_reward}_answer_{answer_reward}"


def build_analysis_examples(
    records: list[dict[str, Any]],
    examples_per_bucket: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    为每类结果抽样保存若干条，供回答作业 (b) 使用。

    每条样例只保留分析所需字段，避免分析文件太大。
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {
        "format_1_answer_1": [],
        "format_1_answer_0": [],
        "format_0_answer_0": [],
    }
    for record in records:
        bucket_key = make_bucket_key(record["metrics"])
        if bucket_key in buckets:
            buckets[bucket_key].append(
                {
                    "question": get_question(record["example"]),
                    "model_response": record["model_response"],
                    "ground_truth": record["ground_truth"],
                    "metrics": record["metrics"],
                }
            )

    samples = {}
    for bucket_key, bucket_records in buckets.items():
        if len(bucket_records) <= examples_per_bucket:
            samples[bucket_key] = bucket_records
        else:
            samples[bucket_key] = rng.sample(bucket_records, examples_per_bucket)
    return samples


def summarize(metrics: list[dict[str, float]]) -> dict[str, Any]:
    """
    汇总作业要求分析的核心指标。

    format_1_answer_1：格式正确，答案也正确。
    format_1_answer_0：格式正确，但答案错误。
    format_0_answer_0：格式错误，因此答案也视为错误。
    """
    buckets = Counter(
        (
            int(metric["format_reward"]),
            int(metric["answer_reward"]),
        )
        for metric in metrics
    )
    return {
        "num_examples": len(metrics),
        "mean_reward": mean(metric["reward"] for metric in metrics) if metrics else 0.0,
        "mean_format_reward": mean(metric["format_reward"] for metric in metrics) if metrics else 0.0,
        "mean_answer_reward": mean(metric["answer_reward"] for metric in metrics) if metrics else 0.0,
        "format_1_answer_1": buckets[(1, 1)],
        "format_1_answer_0": buckets[(1, 0)],
        "format_0_answer_0": buckets[(0, 0)],
    }


def evaluate(
    input_path: str | Path,
    output_path: str | Path,
    prompt_template_path: str | Path,
    generation_mode: str,
    model_name_or_path: str,
    num_gpus: int,
    limit: int | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    analysis_examples_per_bucket: int,
    seed: int,
) -> dict[str, Any]:
    """执行完整评估流程：读取数据、构造 prompt、获得输出、打分、落盘、汇总。"""
    examples = read_jsonl(input_path, limit=limit)
    logger.info("Read %d examples from %s", len(examples), input_path)

    template = load_r1_zero_template(prompt_template_path)
    prompts = [format_prompt(template, get_question(example)) for example in examples]

    # 两种 generation_mode 的用途：
    # transformers：本地用 Hugging Face Transformers 调模型，适合 4060 小样本测试。
    # vllm：服务器上真实加载模型生成。
    if generation_mode == "vllm":
        responses = vllm_generate(
            prompts=prompts,
            model_name_or_path=model_name_or_path,
            num_gpus=num_gpus,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    elif generation_mode == "transformers":
        responses = transformers_generate(
            prompts=prompts,
            model_name_or_path=model_name_or_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")

    if len(responses) != len(examples):
        raise ValueError(f"Got {len(responses)} responses for {len(examples)} examples")

    # 确保输出目录存在。每行输出保留完整上下文，后续可以直接抽样写分析。
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_records = []
    with xopen(output_path, "w") as fout:
        for example, prompt, response in tqdm(
            zip(examples, prompts, responses),
            total=len(examples),
        ):
            raw_ground_truth = get_ground_truth(example)
            ground_truth = normalize_ground_truth_for_reward(raw_ground_truth)
            metrics = r1_zero_reward_fn(response=response, ground_truth=ground_truth)
            all_metrics.append(metrics)
            # 每条 JSONL 记录包含：
            # - 原始样例 example
            # - 送入模型的 prompt
            # - 模型回答 model_response
            # - 标准答案 ground_truth
            # - 奖励函数返回的 metrics
            record = {
                "example": example,
                "prompt": prompt,
                "model_response": response,
                "raw_ground_truth": raw_ground_truth,
                "ground_truth": ground_truth,
                "metrics": metrics,
            }
            all_records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(all_metrics)
    summary_path = output_path.with_suffix(".summary.json")
    examples_path = output_path.with_suffix(".examples.json")

    analysis_examples = build_analysis_examples(
        records=all_records,
        examples_per_bucket=analysis_examples_per_bucket,
        seed=seed,
    )
    summary_payload = {
        **summary,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name_or_path": model_name_or_path,
        "generation_mode": generation_mode,
        "limit": limit,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n"
    )
    examples_path.write_text(
        json.dumps(analysis_examples, ensure_ascii=False, indent=2) + "\n"
    )

    logger.info("Wrote per-example results to %s", output_path)
    logger.info("Wrote summary to %s", summary_path)
    logger.info("Wrote sampled analysis examples to %s", examples_path)
    for key, value in summary.items():
        logger.info("%s: %s", key, value)
    return summary


def main() -> None:
    """解析命令行参数，并启动评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--prompt-template-path",
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("transformers", "vllm"),
        default="transformers",
    )
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--analysis-examples-per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    evaluate(**vars(args))
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
