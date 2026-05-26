"""
本地 Qwen2.5-Math baseline 评测脚本。

这个脚本是 scripts/math_baseline.py 的轻量封装，用来一次性跑两组结果：

1. 前 1000 条样例：方便快速得到一个稳定一些的小规模 baseline；
2. 全量测试集/验证集：用于最终报告。

默认使用本地模型：

    /home/xuehaiwuya/models/Qwen2.5-Math-1.5B

默认数据集：

    data/math/val.jsonl

如果想在 GSM8K test 上跑，可以改 `--input-path data/gsm8k/test.jsonl`。

示例：

    PYTHONPATH=. python scripts/run_local_qwen_baseline.py

只跑前 1000 条：

    PYTHONPATH=. python scripts/run_local_qwen_baseline.py --run first1000

只跑全量：

    PYTHONPATH=. python scripts/run_local_qwen_baseline.py --run full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from math_baseline import evaluate

logger = logging.getLogger(__name__)

RunMode = Literal["first1000", "full", "both"]


def output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    """
    返回 math_baseline.py 会写出的三个文件路径。

    evaluate() 主输出为 `{prefix}.jsonl`，同时自动派生：
    - `{prefix}.summary.json`
    - `{prefix}.examples.json`
    """
    result_path = output_dir / f"{prefix}.jsonl"
    return (
        result_path,
        result_path.with_suffix(".summary.json"),
        result_path.with_suffix(".examples.json"),
    )


def maybe_run_eval(
    *,
    prefix: str,
    limit: int | None,
    args: argparse.Namespace,
) -> dict:
    """按给定 limit 跑一组 baseline；如果结果已存在且未 overwrite，则直接跳过。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path, summary_path, examples_path = output_paths(output_dir, prefix)

    if summary_path.exists() and result_path.exists() and not args.overwrite:
        logger.info("Skip %s because %s already exists. Use --overwrite to rerun.", prefix, summary_path)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    logger.info("Start %s baseline: limit=%s, output=%s", prefix, limit, result_path)
    summary = evaluate(
        input_path=args.input_path,
        output_path=result_path,
        prompt_template_path=args.prompt_template_path,
        generation_mode=args.generation_mode,
        model_name_or_path=args.model_name_or_path,
        num_gpus=args.num_gpus,
        limit=limit,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        analysis_examples_per_bucket=args.analysis_examples_per_bucket,
        seed=args.seed,
    )
    logger.info("Finished %s baseline.", prefix)
    logger.info("Result files: %s, %s, %s", result_path, summary_path, examples_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/math/val.jsonl")
    parser.add_argument("--output-dir", default="output/local_qwen_baseline")
    parser.add_argument(
        "--model-name-or-path",
        default="/home/xuehaiwuya/models/Qwen2.5-Math-1.5B",
    )
    parser.add_argument(
        "--prompt-template-path",
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument("--generation-mode", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--run", choices=("first1000", "full", "both"), default="both")
    parser.add_argument("--first-limit", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--analysis-examples-per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))
    logger.info("input_path=%s", args.input_path)
    logger.info("model_name_or_path=%s", args.model_name_or_path)
    logger.info("output_dir=%s", args.output_dir)

    summaries: dict[str, dict] = {}
    if args.run in {"first1000", "both"}:
        summaries["first1000"] = maybe_run_eval(
            prefix="first1000",
            limit=args.first_limit,
            args=args,
        )
    if args.run in {"full", "both"}:
        summaries["full"] = maybe_run_eval(
            prefix="full",
            limit=None,
            args=args,
        )

    summary_path = Path(args.output_dir) / "combined_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote combined summary to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
