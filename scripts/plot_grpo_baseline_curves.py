"""
绘制 GRPO baseline 对比曲线。

这个脚本不依赖 matplotlib，直接生成 SVG，适合在本地环境没有绘图库时使用。
输入：

- baseline summary：本地 Qwen baseline 的 first1000.summary.json
- GRPO metrics：训练过程中保存的 metrics.jsonl
- final full eval：可选，完整验证集最终结果

输出：

- grpo_baseline_curves.svg：两栏曲线图
- grpo_eval_curves.csv：eval step 曲线数据
- grpo_rollout_curves.csv：rollout 每步过程数据
- grpo_baseline_report.md：简短结果摘要
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """读取 metrics.jsonl。"""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def mean(values: list[float]) -> float:
    """安全求均值。"""
    return sum(values) / len(values) if values else 0.0


def collect_eval_points(metrics: list[dict[str, Any]]) -> list[dict[str, float]]:
    """提取每次 validation eval 的指标。"""
    rows = []
    for record in metrics:
        if "eval/reward" not in record:
            continue
        rows.append(
            {
                "step": float(record["grpo_step"]),
                "optimizer_step": float(record.get("optimizer_step", 0.0)),
                "num_examples": float(record.get("eval/num_examples", 0.0)),
                "reward": float(record["eval/reward"]),
                "answer_reward": float(record.get("eval/answer_reward", record["eval/reward"])),
                "format_reward": float(record.get("eval/format_reward", 0.0)),
            }
        )
    return rows


def collect_rollout_points(metrics: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    提取 rollout 过程指标。

    off-policy 配置下同一个 grpo_step 会有两个 rollout_epoch 训练日志，它们共享同一批
    rollout reward。为了画“每步过程”，这里按 grpo_step 去重后取均值。
    """
    grouped: dict[float, list[dict[str, Any]]] = {}
    for record in metrics:
        if "rollout/reward" not in record:
            continue
        grouped.setdefault(float(record["grpo_step"]), []).append(record)

    rows = []
    for step in sorted(grouped):
        group = grouped[step]
        rows.append(
            {
                "step": step,
                "reward": mean([float(item["rollout/reward"]) for item in group]),
                "answer_reward": mean([float(item.get("rollout/answer_reward", item["rollout/reward"])) for item in group]),
                "format_reward": mean([float(item.get("rollout/format_reward", 0.0)) for item in group]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """写 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def path_line(points: list[tuple[float, float]]) -> str:
    """把点序列转成 SVG polyline points。"""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def star_points(cx: float, cy: float, outer: float = 11.0, inner: float = 5.0) -> str:
    """生成五角星 SVG polygon points。"""
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return path_line(points)


def render_panel(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    baseline: float,
    baseline_label: str,
    eval_points: list[dict[str, float]],
    rollout_points: list[dict[str, float]],
    metric_key: str,
    line_color: str,
    rollout_color: str,
    final_full: dict[str, float] | None,
) -> str:
    """渲染单个子图。"""
    left = x + 62
    right = x + width - 28
    top = y + 58
    bottom = y + height - 58
    plot_w = right - left
    plot_h = bottom - top

    max_step = max(
        [point["step"] for point in eval_points] + [point["step"] for point in rollout_points] + [50.0]
    )

    y_values = [baseline]
    y_values.extend(point[metric_key] for point in eval_points)
    y_values.extend(point[metric_key] for point in rollout_points)
    if final_full is not None:
        y_values.append(final_full[metric_key])
    y_max = max(1.0, math.ceil((max(y_values) + 0.03) * 10) / 10)
    y_min = 0.0

    def sx(step: float) -> float:
        return left + (step / max_step) * plot_w

    def sy(value: float) -> float:
        return bottom - ((value - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<text x="{x + width / 2:.1f}" y="{y + 28:.1f}" class="panel-title" text-anchor="middle">{escape(title)}</text>',
        f'<text x="{x + 18:.1f}" y="{y + height / 2:.1f}" class="axis-label" text-anchor="middle" transform="rotate(-90 {x + 18:.1f} {y + height / 2:.1f})">{escape(y_label)}</text>',
        f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" class="axis"/>',
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" class="axis"/>',
    ]

    # 网格线和刻度。
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = sy(value)
        parts.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8:.1f}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    for step in range(0, int(max_step) + 1, 10):
        xx = sx(float(step))
        parts.append(f'<line x1="{xx:.1f}" y1="{bottom:.1f}" x2="{xx:.1f}" y2="{bottom + 5:.1f}" class="axis"/>')
        parts.append(f'<text x="{xx:.1f}" y="{bottom + 22:.1f}" class="tick" text-anchor="middle">{step}</text>')

    baseline_y = sy(baseline)
    parts.append(
        f'<line x1="{left:.1f}" y1="{baseline_y:.1f}" x2="{right:.1f}" y2="{baseline_y:.1f}" class="baseline"/>'
    )
    parts.append(
        f'<text x="{right - 4:.1f}" y="{baseline_y - 7:.1f}" class="legend" text-anchor="end">{escape(baseline_label)}</text>'
    )

    if rollout_points:
        rollout_xy = [(sx(point["step"]), sy(point[metric_key])) for point in rollout_points]
        parts.append(
            f'<polyline points="{path_line(rollout_xy)}" class="rollout" stroke="{rollout_color}"/>'
        )

    if eval_points:
        eval_xy = [(sx(point["step"]), sy(point[metric_key])) for point in eval_points]
        parts.append(f'<polyline points="{path_line(eval_xy)}" class="eval-line" stroke="{line_color}"/>')
        for xx, yy in eval_xy:
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.2" fill="{line_color}" stroke="white" stroke-width="1.5"/>')

    if final_full is not None:
        xx = sx(final_full["step"])
        yy = sy(final_full[metric_key])
        parts.append(
            f'<polygon points="{star_points(xx, yy)}" fill="#e91e63" stroke="#8a0036" stroke-width="1.1"/>'
        )

    legend_x = left + 10
    legend_y = bottom - 64
    parts.extend(
        [
            f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 26:.1f}" y2="{legend_y:.1f}" class="rollout" stroke="{rollout_color}"/>',
            f'<text x="{legend_x + 34:.1f}" y="{legend_y + 4:.1f}" class="legend">rollout batch</text>',
            f'<line x1="{legend_x:.1f}" y1="{legend_y + 22:.1f}" x2="{legend_x + 26:.1f}" y2="{legend_y + 22:.1f}" class="eval-line" stroke="{line_color}"/>',
            f'<circle cx="{legend_x + 13:.1f}" cy="{legend_y + 22:.1f}" r="4.5" fill="{line_color}" stroke="white" stroke-width="1.2"/>',
            f'<text x="{legend_x + 34:.1f}" y="{legend_y + 26:.1f}" class="legend">eval on 512 subset</text>',
            f'<line x1="{legend_x:.1f}" y1="{legend_y + 44:.1f}" x2="{legend_x + 26:.1f}" y2="{legend_y + 44:.1f}" class="baseline"/>',
            f'<text x="{legend_x + 34:.1f}" y="{legend_y + 48:.1f}" class="legend">local baseline on 1K</text>',
        ]
    )
    if final_full is not None:
        parts.extend(
            [
                f'<polygon points="{star_points(legend_x + 13, legend_y + 66, 8, 3.8)}" fill="#e91e63" stroke="#8a0036" stroke-width="1"/>',
                f'<text x="{legend_x + 34:.1f}" y="{legend_y + 70:.1f}" class="legend">final full eval</text>',
            ]
        )

    return "\n".join(parts)


def render_svg(
    baseline: dict[str, Any],
    eval_points: list[dict[str, float]],
    rollout_points: list[dict[str, float]],
    final_full: dict[str, float] | None,
) -> str:
    """渲染完整 SVG。"""
    width = 1180
    height = 560
    baseline_reward = float(baseline["mean_reward"])
    baseline_format = float(baseline["mean_format_reward"])

    final_payload = None
    if final_full is not None and eval_points:
        final_payload = {
            "step": eval_points[-1]["step"],
            "reward": float(final_full["eval/reward"]),
            "answer_reward": float(final_full.get("eval/answer_reward", final_full["eval/reward"])),
            "format_reward": float(final_full.get("eval/format_reward", 0.0)),
        }

    reward_panel = render_panel(
        x=44,
        y=78,
        width=520,
        height=410,
        title="Reward / Answer Accuracy",
        y_label="Accuracy",
        baseline=baseline_reward,
        baseline_label=f"baseline {baseline_reward:.1%}",
        eval_points=eval_points,
        rollout_points=rollout_points,
        metric_key="reward",
        line_color="#43a047",
        rollout_color="#a5d6a7",
        final_full=final_payload,
    )
    format_panel = render_panel(
        x=616,
        y=78,
        width=520,
        height=410,
        title="Format Accuracy",
        y_label="Accuracy",
        baseline=baseline_format,
        baseline_label=f"baseline {baseline_format:.1%}",
        eval_points=eval_points,
        rollout_points=rollout_points,
        metric_key="format_reward",
        line_color="#1e88e5",
        rollout_color="#90caf9",
        final_full=final_payload,
    )

    subtitle = (
        f"Baseline: Qwen2.5-Math-1.5B local first 1000; "
        f"GRPO eval: {int(eval_points[-1]['num_examples']) if eval_points else 0} examples per point"
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 22px Arial, sans-serif; fill: #111827; }}
    .subtitle {{ font: 13px Arial, sans-serif; fill: #4b5563; }}
    .panel-title {{ font: 700 18px Arial, sans-serif; fill: #111827; }}
    .axis-label {{ font: 14px Arial, sans-serif; fill: #374151; }}
    .tick {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .legend {{ font: 12px Arial, sans-serif; fill: #374151; }}
    .axis {{ stroke: #111827; stroke-width: 1.1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .baseline {{ stroke: #8b8b8b; stroke-width: 2.3; stroke-dasharray: 8 6; }}
    .eval-line {{ fill: none; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }}
    .rollout {{ fill: none; stroke-width: 2.1; stroke-linecap: round; stroke-linejoin: round; opacity: 0.88; }}
  </style>
  <rect x="20" y="24" width="1140" height="508" rx="12" fill="#ffffff" stroke="#e5e7eb"/>
  <text x="590" y="58" class="title" text-anchor="middle">GRPO: Qwen2.5-Math-1.5B Baseline vs Training Curves</text>
  <text x="590" y="80" class="subtitle" text-anchor="middle">{escape(subtitle)}</text>
  {reward_panel}
  {format_panel}
  <text x="590" y="526" class="axis-label" text-anchor="middle">GRPO Step</text>
</svg>
'''


def write_report(
    path: Path,
    baseline: dict[str, Any],
    eval_points: list[dict[str, float]],
    final_full: dict[str, Any] | None,
) -> None:
    """写一份简短 markdown 摘要。"""
    final_eval = eval_points[-1] if eval_points else {}
    lines = [
        "# GRPO baseline curve summary",
        "",
        "## Local baseline on first 1000",
        "",
        f"- reward / answer accuracy: {float(baseline['mean_reward']):.4f}",
        f"- format accuracy: {float(baseline['mean_format_reward']):.4f}",
        f"- examples: {int(baseline['num_examples'])}",
        "",
        "## Final subset eval during GRPO",
        "",
        f"- step: {int(final_eval.get('step', 0))}",
        f"- reward / answer accuracy: {float(final_eval.get('reward', 0.0)):.4f}",
        f"- format accuracy: {float(final_eval.get('format_reward', 0.0)):.4f}",
        f"- eval examples: {int(final_eval.get('num_examples', 0))}",
    ]
    if final_full is not None:
        lines.extend(
            [
                "",
                "## Final full validation eval",
                "",
                f"- reward / answer accuracy: {float(final_full['eval/reward']):.4f}",
                f"- format accuracy: {float(final_full['eval/format_reward']):.4f}",
                f"- examples: {int(final_full['num_examples'])}",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", default="output/local_qwen_baseline/first1000.summary.json")
    parser.add_argument("--metrics-path", default="output/grpo_50step/metrics.jsonl")
    parser.add_argument(
        "--final-full-summary",
        default="output/grpo_50step_tar/offpolicy_50step_fast/final_full_eval.summary.json",
    )
    parser.add_argument("--output-dir", default="output/grpo_50step/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = read_json(args.baseline_summary)
    metrics = read_metrics(args.metrics_path)
    eval_points = collect_eval_points(metrics)
    rollout_points = collect_rollout_points(metrics)

    final_full_path = Path(args.final_full_summary)
    final_full = read_json(final_full_path) if final_full_path.exists() else None

    write_csv(output_dir / "grpo_eval_curves.csv", eval_points)
    write_csv(output_dir / "grpo_rollout_curves.csv", rollout_points)
    (output_dir / "grpo_baseline_curves.svg").write_text(
        render_svg(baseline, eval_points, rollout_points, final_full),
        encoding="utf-8",
    )
    write_report(output_dir / "grpo_baseline_report.md", baseline, eval_points, final_full)


if __name__ == "__main__":
    main()
