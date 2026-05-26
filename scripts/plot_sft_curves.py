"""
汇总并绘制 4.3 SFT validation accuracy curves。

输入是一个或多个 SFT 实验输出目录，例如：

    /data/outputs/sft/filtered_full_fast_subset_eval

脚本会读取其中 eval/step_xxxxxx.summary.json，输出：

- sft_curves.csv：所有实验、所有 eval step 的指标表
- sft_report.md：便于写作业报告的 Markdown 摘要
- sft_validation_curves.png：validation accuracy 曲线图

如果服务器没有 matplotlib，脚本仍然会正常输出 CSV 和 Markdown，只跳过画图。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SUMMARY_RE = re.compile(r"step_(\d+)\.summary\.json$")


def read_eval_summaries(experiment_dir: Path) -> list[dict[str, object]]:
    """读取单个实验目录下的 eval summary，并按 step 排序。"""
    eval_dir = experiment_dir / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval dir: {eval_dir}")

    rows = []
    for path in eval_dir.glob("step_*.summary.json"):
        match = SUMMARY_RE.match(path.name)
        if match is None:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        step = int(match.group(1))
        rows.append(
            {
                "experiment": experiment_dir.name,
                "step": step,
                "num_examples": summary.get("num_examples", summary.get("eval/num_examples")),
                "answer_accuracy": summary.get("eval/answer_accuracy"),
                "format_accuracy": summary.get("eval/format_accuracy"),
                "reward": summary.get("eval/reward"),
                "summary_path": str(path),
            }
        )
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """把曲线数据写成 CSV，方便复制到报告或表格软件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "step",
        "num_examples",
        "answer_accuracy",
        "format_accuracy",
        "reward",
        "summary_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows_by_experiment: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    """生成一份简短 Markdown 摘要，作为作业报告草稿素材。"""
    lines = ["# SFT validation results", ""]
    for experiment, rows in rows_by_experiment.items():
        if not rows:
            continue
        final = rows[-1]
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append(f"- eval points: {len(rows)}")
        lines.append(f"- final step: {final['step']}")
        lines.append(f"- validation examples per eval: {final['num_examples']}")
        lines.append(f"- final answer accuracy: {final['answer_accuracy']}")
        lines.append(f"- final format accuracy: {final['format_accuracy']}")
        lines.append(f"- final reward: {final['reward']}")
        lines.append("")
        lines.append("| step | answer_accuracy | format_accuracy | reward | num_examples |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['step']} | {row['answer_accuracy']} | "
                f"{row['format_accuracy']} | {row['reward']} | {row['num_examples']} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(rows_by_experiment: dict[str, list[dict[str, object]]], output_path: Path) -> bool:
    """绘制 answer accuracy 曲线；没有 matplotlib 时返回 False。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for experiment, rows in rows_by_experiment.items():
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        accuracies = [float(row["answer_accuracy"]) for row in rows]
        plt.plot(steps, accuracies, marker="o", label=experiment)

    plt.xlabel("optimizer step")
    plt.ylabel("validation answer accuracy")
    plt.title("SFT validation accuracy curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        action="append",
        required=True,
        help="SFT experiment output dir. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows_by_experiment = {}
    all_rows = []

    for experiment_dir_str in args.experiment_dir:
        experiment_dir = Path(experiment_dir_str)
        rows = read_eval_summaries(experiment_dir)
        rows_by_experiment[experiment_dir.name] = rows
        all_rows.extend(rows)

    write_csv(all_rows, output_dir / "sft_curves.csv")
    write_markdown(rows_by_experiment, output_dir / "sft_report.md")
    plotted = plot_curves(rows_by_experiment, output_dir / "sft_validation_curves.png")

    print(f"Wrote {output_dir / 'sft_curves.csv'}")
    print(f"Wrote {output_dir / 'sft_report.md'}")
    if plotted:
        print(f"Wrote {output_dir / 'sft_validation_curves.png'}")
    else:
        print("matplotlib is not installed; skipped PNG plot.")


if __name__ == "__main__":
    main()
