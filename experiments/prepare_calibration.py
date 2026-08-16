"""Materialize one deterministic Wikitext calibration corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "train"
DEFAULT_SEED = 1234
DEFAULT_EXAMPLES = 128


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(output: Path, *, seed: int, examples: int) -> None:
    if output.exists():
        raise RuntimeError(f"Calibration file already exists: {output}")

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    eligible = [idx for idx, text in enumerate(dataset["text"]) if text.strip()]
    if examples > len(eligible):
        raise ValueError(f"Requested {examples} examples, but only {len(eligible)} are non-empty.")
    selected = random.Random(seed).sample(eligible, examples)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for order, row_index in enumerate(selected):
            text = dataset[row_index]["text"]
            record = {
                "order": order,
                "dataset_id": DATASET_ID,
                "dataset_config": DATASET_CONFIG,
                "dataset_revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
                "row_index": row_index,
                "selection_seed": seed,
                "text_sha256": text_sha256(text),
                "text": text,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/calibration/wikitext_2_raw_v1_seed1234_n128.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    args = parser.parse_args()
    prepare(args.output, seed=args.seed, examples=args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
