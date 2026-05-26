"""
Download hiyouga/math12k and export train/test splits as JSONL.

Example:
    python scripts/prepare_math12k.py \
        --output-dir data/math12k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def write_jsonl(split, output_path: Path) -> None:
    """Write a Hugging Face dataset split to JSONL with problem/answer fields."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in tqdm(split, desc=f"Writing {output_path.name}"):
            f.write(
                json.dumps(
                    {
                        "problem": example["problem"],
                        "answer": example["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/math12k")
    parser.add_argument("--dataset-name", default="hiyouga/math12k")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    output_dir = Path(args.output_dir)

    for split_name, split in dataset.items():
        write_jsonl(split, output_dir / f"{split_name}.jsonl")


if __name__ == "__main__":
    main()
