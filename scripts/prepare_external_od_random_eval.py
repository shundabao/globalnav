#!/usr/bin/env python3
"""Create reproducible random evaluation samples from the external OD subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBSET_DIR = ROOT / "data" / "external_od_subsets"
DATASET_FILES = {
    "multiwoz24": (
        "multiwoz24_explicit_od.jsonl",
        "multiwoz24_clarification.jsonl",
    ),
    "sgd": (
        "sgd_explicit_od.jsonl",
        "sgd_clarification.jsonl",
    ),
    "atis": ("atis_explicit_single_pair.jsonl",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every eligible sample in each dataset instead of random sampling.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def pool_fingerprint(rows: list[dict[str, Any]]) -> str:
    sample_ids = sorted(row["sample_id"] for row in rows)
    content = "\n".join(sample_ids).encode()
    return hashlib.sha256(content).hexdigest()


def normalize_record(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    clarification = row["task_type"] == "clarification_resolution"
    gold = row["gold"]
    origin_variants = list(
        dict.fromkeys(
            value
            for value in (gold["origin"], gold.get("origin_surface"))
            if value
        )
    )
    destination_variants = list(
        dict.fromkeys(
            value
            for value in (gold["destination"], gold.get("destination_surface"))
            if value
        )
    )
    return {
        "sample_id": row["sample_id"],
        "dataset": dataset,
        "source_id": str(row["source_id"]),
        "turn_index": row.get("turn_index", 0),
        "domain": row.get("domain") or row.get("service") or row.get("intent", ""),
        "instruction": row["input_utterance"],
        "raw_utterance": row["input_utterance"],
        "gold_origin": gold["origin"],
        "gold_destination": gold["destination"],
        "gold_origin_variants": origin_variants,
        "gold_destination_variants": destination_variants,
        "expected_clarification": clarification,
        "missing_endpoint": gold.get("missing_endpoint"),
        "clarification_answer": (
            row.get("original_clarification", {}).get("user_answer", "")
            if clarification
            else ""
        ),
        "source_record": row,
    }


def main() -> None:
    args = parse_args()
    selected: list[dict[str, Any]] = []
    manifest_datasets: dict[str, Any] = {}

    for dataset, filenames in DATASET_FILES.items():
        pool: list[dict[str, Any]] = []
        for filename in filenames:
            pool.extend(read_jsonl(args.subset_dir / filename))
        pool.sort(key=lambda row: row["sample_id"])
        if not args.all and args.count_per_dataset > len(pool):
            raise ValueError(
                f"{dataset}: requested {args.count_per_dataset}, pool has {len(pool)}"
            )

        dataset_seed = stable_dataset_seed(args.seed, dataset)
        if args.all:
            chosen = pool
        else:
            chosen = random.Random(dataset_seed).sample(pool, args.count_per_dataset)
        normalized = [normalize_record(dataset, row) for row in chosen]
        selected.extend(normalized)
        manifest_datasets[dataset] = {
            "source_files": list(filenames),
            "pool_size": len(pool),
            "pool_sample_id_sha256": pool_fingerprint(pool),
            "derived_random_seed": None if args.all else dataset_seed,
            "selected_count": len(normalized),
            "task_type_counts": dict(
                Counter(row["source_record"]["task_type"] for row in normalized)
            ),
            "selected_sample_ids": [row["sample_id"] for row in normalized],
        }

    samples_path = args.output_dir / "samples.jsonl"
    write_jsonl(samples_path, selected)
    sample_bytes = samples_path.read_bytes()
    manifest = {
        "method": (
            "Full eligible pool, sorted by sample_id within each dataset."
            if args.all
            else "Uniform random sampling without replacement within each dataset."
        ),
        "base_seed": args.seed,
        "count_per_dataset": None if args.all else args.count_per_dataset,
        "total_selected": len(selected),
        "dataset_order": list(DATASET_FILES),
        "sample_file": samples_path.name,
        "sample_file_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "datasets": manifest_datasets,
        "warning": (
            "If model prompts are tuned after inspecting these results, treat these "
            "sample IDs as development data and exclude them from a later final test."
        ),
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
