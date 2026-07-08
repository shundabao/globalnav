#!/usr/bin/env python3
"""Evaluate a vision-language action follower on route annotation records."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe_nav.config import get_openai_api_key
from globe_nav.maps.streetview import GlobalStreetViewProvider


ACTIONS = ("forward", "left", "right", "stop")
MODEL_PRICES_PER_MILLION = {
    "gpt-5.4-mini": {"input_usd": 0.75, "output_usd": 4.50},
}
SYSTEM_PROMPT = """You are a street-navigation instruction follower.
Choose exactly one action for the current first-person street view:
forward, left, right, or stop.

Follow the active instruction. Choose forward when its maneuver is not due yet.
Choose left or right only when the instructed turn should be taken at the current
location. Choose stop only when the destination has been reached. The camera faces
the direction of travel before the action. Reply with exactly one action word."""


@dataclass(frozen=True)
class EvalCase:
    sample_id: str
    route_id: str
    route_index: int
    case_type: str
    node_index: int
    source_index: int
    target_index: int
    origin: str
    destination: str
    instruction: str
    distance_since_instruction_m: int
    lat: float
    lon: float
    heading: float
    gold_action: str


def load_records(path: Path, route_count: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < route_count:
        raise ValueError(f"requested {route_count} routes, but {path} contains {len(rows)}")
    return rows[:route_count]


def haversine_m(a: dict, b: dict) -> float:
    radius_m = 6_371_000.0
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlat = lat2 - lat1
    dlon = math.radians(b["lon"] - a["lon"])
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(h))


def bearing_deg(a: dict, b: dict) -> float:
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def cumulative_distances(nodes: list[dict]) -> list[float]:
    distances = [0.0]
    for previous, current in zip(nodes, nodes[1:]):
        distances.append(distances[-1] + haversine_m(previous, current))
    return distances


def choose_forward_control(
    nodes: list[dict],
    cumulative: list[float],
    source_index: int,
    target_index: int,
) -> int:
    candidates = [
        idx
        for idx in range(source_index, target_index)
        if nodes[idx].get("oracle_action") == "forward"
    ]
    if not candidates:
        raise ValueError(f"no forward control before node {target_index}")
    midpoint = (cumulative[source_index] + cumulative[target_index]) / 2.0
    return min(candidates, key=lambda idx: abs(cumulative[idx] - midpoint))


def build_cases(records: list[dict]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for route_index, record in enumerate(records):
        nodes = record["nodes"]
        cumulative = cumulative_distances(nodes)
        source_by_target = {
            node.get("auto_instruction_target"): idx
            for idx, node in enumerate(nodes)
            if node.get("annotation_instruction") and node.get("auto_instruction_target")
        }

        for target_index, target in enumerate(nodes):
            action = target.get("action_override") or target.get("oracle_action")
            if action not in ("left", "right", "stop"):
                continue
            source_index = source_by_target.get(target.get("node_id"))
            if source_index is None:
                continue
            source = nodes[source_index]
            instruction = source["annotation_instruction"].strip()
            control_index = choose_forward_control(
                nodes, cumulative, source_index, target_index
            )

            for case_type, node_index in (
                ("forward_control", control_index),
                ("keypoint", target_index),
            ):
                node = nodes[node_index]
                gold_action = (
                    node.get("action_override") or node.get("oracle_action")
                )
                if node_index > 0 and gold_action != "forward":
                    heading = bearing_deg(nodes[node_index - 1], node)
                else:
                    heading = float(node.get("heading_deg", 0.0))
                cases.append(
                    EvalCase(
                        sample_id=(
                            f"{record['id']}:{target['node_id']}:{case_type}"
                        ),
                        route_id=record["id"],
                        route_index=route_index,
                        case_type=case_type,
                        node_index=node_index,
                        source_index=source_index,
                        target_index=target_index,
                        origin=record["origin"],
                        destination=record["destination"],
                        instruction=instruction,
                        distance_since_instruction_m=round(
                            cumulative[node_index] - cumulative[source_index]
                        ),
                        lat=float(node["lat"]),
                        lon=float(node["lon"]),
                        heading=round(heading, 1),
                        gold_action=gold_action,
                    )
                )
    return cases


def image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_action(text: str) -> str:
    normalized = text.lower().strip().replace("-", "_")
    for token in re.findall(r"[a-z_]+", normalized):
        if token in ACTIONS:
            return token
    return ""


def usage_dict(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def evaluate_case(
    case: EvalCase,
    *,
    client: Any,
    provider: GlobalStreetViewProvider,
    model: str,
    retries: int,
) -> dict:
    started = time.monotonic()
    base = {
        **case.__dict__,
        "model": model,
        "vision": True,
    }
    try:
        image = provider.image_bytes(
            case.lat, case.lon, heading=case.heading
        )
        if not image:
            raise RuntimeError("street-view image unavailable")

        prompt = (
            f"Route: {case.origin} to {case.destination}\n"
            f"Active instruction: {case.instruction}\n"
            f"Distance travelled since this instruction: "
            f"{case.distance_since_instruction_m} meters\n"
            "What action should be taken now?"
        )
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": "none"},
                    max_output_tokens=16,
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": SYSTEM_PROMPT}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": image_data_url(image),
                                    "detail": "low",
                                },
                            ],
                        },
                    ],
                )
                raw_output = (response.output_text or "").strip()
                prediction = extract_action(raw_output)
                if not prediction:
                    raise ValueError(f"unparseable model output: {raw_output!r}")
                return {
                    **base,
                    "prediction": prediction,
                    "correct": prediction == case.gold_action,
                    "raw_output": raw_output,
                    "usage": usage_dict(response),
                    "latency_s": round(time.monotonic() - started, 3),
                    "error": "",
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise last_error or RuntimeError("model request failed")
    except Exception as exc:
        return {
            **base,
            "prediction": "",
            "correct": False,
            "raw_output": "",
            "usage": {},
            "latency_s": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def classification_metrics(
    gold: list[str], predictions: list[str], labels: tuple[str, ...] = ACTIONS
) -> dict:
    per_class = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predictions))
        fp = sum(g != label and p == label for g, p in zip(gold, predictions))
        fn = sum(g == label and p != label for g, p in zip(gold, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": sum(g == label for g in gold),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    accuracy = (
        sum(g == p for g, p in zip(gold, predictions)) / len(gold)
        if gold
        else 0.0
    )
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(labels)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(labels)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def summarize(
    results: list[dict],
    *,
    records: list[dict],
    cases: list[EvalCase],
    model: str,
    seed: int,
) -> dict:
    valid = [row for row in results if not row["error"]]
    gold = [row["gold_action"] for row in valid]
    predictions = [row["prediction"] for row in valid]

    keypoints = [row for row in valid if row["case_type"] == "keypoint"]
    turns = [row for row in keypoints if row["gold_action"] in ("left", "right")]
    route_keypoints: dict[str, list[dict]] = defaultdict(list)
    for row in keypoints:
        route_keypoints[row["route_id"]].append(row)
    route_successes = {
        route_id: bool(rows) and all(row["correct"] for row in rows)
        for route_id, rows in route_keypoints.items()
    }

    rng = random.Random(seed)
    random_predictions = [rng.choice(ACTIONS) for _ in gold]
    forward_predictions = ["forward"] * len(gold)
    totals = Counter()
    for row in valid:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] += value
    pricing = MODEL_PRICES_PER_MILLION.get(model)
    estimated_cost = None
    if pricing:
        estimated_cost = (
            totals["input_tokens"] * pricing["input_usd"]
            + totals["output_tokens"] * pricing["output_usd"]
        ) / 1_000_000

    per_route = []
    for record in records:
        rows = [row for row in valid if row["route_id"] == record["id"]]
        kp_rows = [row for row in rows if row["case_type"] == "keypoint"]
        per_route.append(
            {
                "route_id": record["id"],
                "origin": record["origin"],
                "destination": record["destination"],
                "evaluated_cases": len(rows),
                "keypoints": len(kp_rows),
                "keypoints_correct": sum(row["correct"] for row in kp_rows),
                "keypoint_accuracy": round(
                    sum(row["correct"] for row in kp_rows) / len(kp_rows), 4
                )
                if kp_rows
                else 0.0,
                "route_success": route_successes.get(record["id"], False),
            }
        )

    return {
        "created_at": int(time.time()),
        "model": model,
        "vision": True,
        "protocol": {
            "routes": len(records),
            "route_selection": "first N records in input JSONL",
            "cases": (
                "all annotated keypoints, paired with one distance-matched "
                "forward control from the same active-instruction interval"
            ),
            "teacher_forced": True,
            "image_detail": "low",
            "turn_camera_heading": "incoming heading before action",
            "seed": seed,
        },
        "requested_cases": len(cases),
        "completed_cases": len(valid),
        "failed_cases": len(results) - len(valid),
        "action_distribution": dict(Counter(gold)),
        "model_metrics": {
            **classification_metrics(gold, predictions),
            "keypoint_accuracy": round(
                sum(row["correct"] for row in keypoints) / len(keypoints), 4
            )
            if keypoints
            else 0.0,
            "turn_accuracy": round(
                sum(row["correct"] for row in turns) / len(turns), 4
            )
            if turns
            else 0.0,
            "route_success_rate": round(
                sum(route_successes.values()) / len(records), 4
            )
            if records
            else 0.0,
            "successful_routes": sum(route_successes.values()),
            "total_routes": len(records),
        },
        "baselines": {
            "always_forward": classification_metrics(gold, forward_predictions),
            "uniform_random": classification_metrics(gold, random_predictions),
        },
        "usage": dict(totals),
        "cost": {
            "estimated_usd": round(estimated_cost, 6)
            if estimated_cost is not None
            else None,
            "rates_per_million_tokens": pricing,
            "note": (
                "Estimate from reported API tokens; check the platform billing "
                "dashboard for the charged amount."
            ),
        },
        "per_route": per_route,
        "failed_samples": [
            {"sample_id": row["sample_id"], "error": row["error"]}
            for row in results
            if row["error"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/globnav_bench/instruction_annotations_auto.jsonl"),
    )
    parser.add_argument("--routes", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evals/gpt-5.4-mini_vision_20"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input, args.routes)
    cases = build_cases(records)
    if args.max_cases:
        cases = cases[: args.max_cases]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    existing = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["sample_id"]] = row
    pending = [case for case in cases if case.sample_id not in existing]

    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key(), timeout=90.0)
    provider = GlobalStreetViewProvider(size="640x640")
    write_lock = threading.Lock()
    completed = len(existing)
    print(
        f"model={args.model} routes={len(records)} cases={len(cases)} "
        f"resuming={completed} pending={len(pending)}",
        flush=True,
    )

    def run(case: EvalCase) -> dict:
        return evaluate_case(
            case,
            client=client,
            provider=provider,
            model=args.model,
            retries=args.retries,
        )

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {executor.submit(run, case): case for case in pending}
        for future in as_completed(futures):
            row = future.result()
            existing[row["sample_id"]] = row
            with write_lock:
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed += 1
            status = "ok" if not row["error"] else "ERROR"
            print(
                f"[{completed}/{len(cases)}] {status} {row['sample_id']} "
                f"gold={row['gold_action']} pred={row['prediction'] or '-'}",
                flush=True,
            )

    ordered = [existing[case.sample_id] for case in cases if case.sample_id in existing]
    summary = summarize(
        ordered,
        records=records,
        cases=cases,
        model=args.model,
        seed=args.seed,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["model_metrics"], indent=2), flush=True)
    print(f"wrote {results_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
