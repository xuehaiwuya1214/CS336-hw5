"""
No-tool ICRL-lite prompt utilities.

本模块只负责数据输入和 prompt 构造，不包含任何 GRPO loss / reward /
advantage / optimizer 逻辑。ICRL-lite 实验只在 rollout 阶段把原始
r1_zero prompt 替换为 few-shot prompt；eval 阶段仍使用 zero-shot prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ICRLDemo:
    """一条 no-tool few-shot demonstration。"""

    id: str
    question: str
    reasoning: str
    answer: str
    category: str = ""


def load_icrl_demos(path: str | Path) -> list[ICRLDemo]:
    """读取 JSONL demo bank。"""
    demos: list[ICRLDemo] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        demos.append(
            ICRLDemo(
                id=str(record["id"]),
                question=str(record["question"]),
                reasoning=str(record["reasoning"]),
                answer=str(record["answer"]),
                category=str(record.get("category", "")),
            )
        )
    if not demos:
        raise ValueError(f"No ICRL demos loaded from {path}.")
    return demos


def num_icrl_shots_for_step(step: int) -> int:
    """
    固定 50-step ICRL-lite 退火 schedule。

    step 1-10: 3-shot
    step 11-20: 2-shot
    step 21-30: 1-shot
    step 31-50: 0-shot
    """
    if step <= 0:
        raise ValueError(f"step must be 1-indexed and positive, got {step}.")
    if step <= 10:
        return 3
    if step <= 20:
        return 2
    if step <= 30:
        return 1
    return 0


def select_icrl_demos_for_step(demos: list[ICRLDemo], step: int) -> list[ICRLDemo]:
    """
    按 step 选择 demonstrations。

    3-shot 使用全部 3 个；2-shot 使用后 2 个；1-shot 使用最后 1 个。
    """
    num_shots = num_icrl_shots_for_step(step)
    if num_shots == 0:
        return []
    if len(demos) < num_shots:
        raise ValueError(f"Need {num_shots} demos for step {step}, only got {len(demos)}.")
    return demos[-num_shots:]


def _format_demo(demo: ICRLDemo) -> str:
    """把一条 demo 格式化为 r1_zero 对话片段。"""
    return (
        f"User: {demo.question}\n"
        "Assistant: <think>\n"
        f"{demo.reasoning}\n"
        f"</think> <answer>{demo.answer}</answer>\n\n"
    )


def build_icrl_prompt(
    question: str,
    demos: list[ICRLDemo],
    r1_zero_template: str,
) -> str:
    """
    构造 rollout 使用的 ICRL-lite prompt。

    demos 为空时，严格退回原始 zero-shot r1_zero prompt，保证 0-shot 阶段
    与 baseline 的 prompt 构造一致。
    """
    if not demos:
        return r1_zero_template.format(question=question)

    marker = "User: {question}"
    if marker not in r1_zero_template:
        raise ValueError("r1_zero_template must contain 'User: {question}'.")

    prefix, suffix = r1_zero_template.split(marker, maxsplit=1)
    demo_block = "".join(_format_demo(demo) for demo in demos)
    return f"{prefix}{demo_block}User: {question}{suffix}"
