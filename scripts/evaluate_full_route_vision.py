#!/usr/bin/env python3
"""Run a sequential vision-language evaluation over every node of one route."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe_nav.config import get_openai_api_key
from globe_nav.maps.streetview import GlobalStreetViewProvider


ACTIONS = {"forward", "left", "right", "stop"}
ORDINALS = {"next": 1, "first": 1, "second": 2, "third": 3, "fourth": 4}
SYSTEM_PROMPT = """You are a sequential street-navigation agent.
At every route viewpoint, inspect the current first-person street image and choose
exactly one action: forward, left, right, or stop.

A physical intersection or traffic-light junction may remain visible across several
consecutive images while the agent approaches it. Count that physical location only
once, not once per image. "completed_relevant_events" means the number of distinct
instruction-relevant intersections or traffic-light locations fully passed before
the current action. Do not increment it merely because the same event appears larger
in a new frame. Turn only at the instructed decision location and stop only at the
destination.

Return only one JSON object with these fields:
action, event_type, event_phase, same_physical_event_as_previous,
completed_relevant_events, instruction_progress, reason.
event_type must be one of none, intersection, traffic_light, destination, other.
event_phase must be one of none, approaching, decision, passed.
instruction_progress must be one of continue, execute, complete.
Keep reason under 25 words."""


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


def image_data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    action = str(parsed.get("action", "")).lower().strip()
    if action not in ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    parsed["action"] = action
    parsed["completed_relevant_events"] = max(
        0, int(parsed.get("completed_relevant_events", 0))
    )
    parsed["same_physical_event_as_previous"] = bool(
        parsed.get("same_physical_event_as_previous", False)
    )
    return parsed


def usage_dict(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def load_route(path: Path, route_index: int) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[route_index]


def cumulative_distances(nodes: list[dict]) -> list[float]:
    result = [0.0]
    for previous, current in zip(nodes, nodes[1:]):
        result.append(result[-1] + haversine_m(previous, current))
    return result


def incoming_headings(nodes: list[dict]) -> list[float]:
    headings = [float(nodes[0].get("heading_deg", 0.0))]
    headings.extend(bearing_deg(nodes[i - 1], nodes[i]) for i in range(1, len(nodes)))
    return headings


def instruction_segments(route: dict) -> list[dict]:
    nodes = route["nodes"]
    target_to_index = {node["node_id"]: node["index"] for node in nodes}
    segments = []
    for node in nodes:
        instruction = node.get("annotation_instruction", "").strip()
        if not instruction:
            continue
        target_index = target_to_index.get(node.get("auto_instruction_target"))
        if target_index is None:
            continue
        match = re.search(
            r"\b(next|first|second|third|fourth)\b", instruction.lower()
        )
        segments.append(
            {
                "source_index": node["index"],
                "target_index": target_index,
                "instruction": instruction,
                "gold_action": nodes[target_index]["oracle_action"],
                "ordinal": ORDINALS.get(match.group(1)) if match else None,
            }
        )
    return segments


def prefetch_images(
    provider: GlobalStreetViewProvider,
    nodes: list[dict],
    headings: list[float],
    start: int,
    end: int,
    workers: int,
) -> dict[int, bytes]:
    images: dict[int, bytes] = {}

    def fetch(index: int) -> tuple[int, bytes]:
        node = nodes[index]
        data = provider.image_bytes(node["lat"], node["lon"], headings[index])
        if not data:
            raise RuntimeError(f"street-view image unavailable at node {index}")
        return index, data

    indices = range(max(0, start - 1), end)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch, index): index for index in indices}
        for future in as_completed(futures):
            index, data = future.result()
            images[index] = data
    return images


def active_segment(segments: list[dict], index: int) -> dict:
    candidates = [segment for segment in segments if segment["source_index"] <= index]
    if not candidates:
        raise ValueError(f"no active instruction at node {index}")
    return candidates[-1]


def evaluate_step(
    *,
    client: Any,
    model: str,
    route: dict,
    nodes: list[dict],
    images: dict[int, bytes],
    headings: list[float],
    cumulative: list[float],
    index: int,
    segment: dict,
    state_before: dict,
    retries: int,
) -> dict:
    node = nodes[index]
    new_instruction = index == segment["source_index"]
    distance_m = round(cumulative[index] - cumulative[segment["source_index"]])
    prompt = (
        f"Route: {route['origin']} to {route['destination']}\n"
        f"Active instruction: {segment['instruction']}\n"
        f"New instruction at this frame: {'yes' if new_instruction else 'no'}\n"
        f"Distance travelled since instruction: {distance_m} meters\n"
        f"Previous navigation state: {json.dumps(state_before, ensure_ascii=False)}\n"
        "The final image below is the current viewpoint, facing the direction of "
        "travel before taking the action. Decide the action now."
    )
    content: list[dict] = []
    if index > 0:
        content.extend(
            [
                {"type": "input_text", "text": "Immediately previous viewpoint:"},
                {
                    "type": "input_image",
                    "image_url": image_data_url(images[index - 1]),
                    "detail": "low",
                },
            ]
        )
    content.extend(
        [
            {"type": "input_text", "text": "Current viewpoint:"},
            {
                "type": "input_image",
                "image_url": image_data_url(images[index]),
                "detail": "low",
            },
            {"type": "input_text", "text": prompt},
        ]
    )

    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": "none"},
                max_output_tokens=180,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {"role": "user", "content": content},
                ],
            )
            raw_output = (response.output_text or "").strip()
            parsed = parse_json_object(raw_output)
            gold_action = node.get("action_override") or node["oracle_action"]
            state_after = {
                "completed_relevant_events": parsed[
                    "completed_relevant_events"
                ],
                "event_type": parsed.get("event_type", "none"),
                "event_phase": parsed.get("event_phase", "none"),
                "event_description": parsed.get("reason", ""),
                "previous_action": parsed["action"],
            }
            return {
                "route_id": route["id"],
                "node_index": index,
                "node_id": node["node_id"],
                "lat": node["lat"],
                "lon": node["lon"],
                "heading": round(headings[index], 1),
                "instruction": segment["instruction"],
                "instruction_source_index": segment["source_index"],
                "instruction_target_index": segment["target_index"],
                "new_instruction": new_instruction,
                "distance_since_instruction_m": distance_m,
                "gold_action": gold_action,
                "prediction": parsed["action"],
                "correct": parsed["action"] == gold_action,
                "model_state_before": state_before,
                "model_state_after": state_after,
                "model_interpretation": parsed,
                "raw_output": raw_output,
                "response_id": response.id,
                "usage": usage_dict(response),
                "latency_s": round(time.monotonic() - started, 3),
                "error": "",
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise last_error or RuntimeError("model request failed")


def summarize(route: dict, rows: list[dict], segments: list[dict], model: str) -> dict:
    usage = Counter()
    for row in rows:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, int):
                usage[key] += value

    keypoints = [row for row in rows if row["gold_action"] != "forward"]
    segment_results = []
    for segment in segments:
        interval = [
            row
            for row in rows
            if segment["source_index"] <= row["node_index"] <= segment["target_index"]
        ]
        target = next(
            (row for row in interval if row["node_index"] == segment["target_index"]),
            None,
        )
        premature = [
            row
            for row in interval
            if row["node_index"] < segment["target_index"]
            and row["prediction"] != "forward"
        ]
        segment_results.append(
            {
                **segment,
                "completed": target is not None,
                "target_prediction": target["prediction"] if target else None,
                "target_correct": target["correct"] if target else None,
                "model_completed_events_at_target": (
                    target["model_interpretation"].get(
                        "completed_relevant_events"
                    )
                    if target
                    else None
                ),
                "expected_completed_events_at_target": (
                    segment["ordinal"] - 1 if segment["ordinal"] else None
                ),
                "target_reason": (
                    target["model_interpretation"].get("reason", "")
                    if target
                    else ""
                ),
                "premature_nonforward_count": len(premature),
                "premature_nodes": [row["node_index"] for row in premature],
            }
        )

    count_decreases = []
    count_jumps = []
    duplicate_count_changes = []
    for previous, current in zip(rows, rows[1:]):
        if current["new_instruction"]:
            continue
        before = previous["model_state_after"]["completed_relevant_events"]
        after = current["model_state_after"]["completed_relevant_events"]
        if after < before:
            count_decreases.append(current["node_index"])
        if after > before + 1:
            count_jumps.append(current["node_index"])
        interpretation = current["model_interpretation"]
        if (
            interpretation.get("same_physical_event_as_previous")
            and after > before
            and interpretation.get("event_phase") != "passed"
        ):
            duplicate_count_changes.append(current["node_index"])

    input_price = 0.75
    output_price = 4.50
    estimated_cost = (
        usage["input_tokens"] * input_price
        + usage["output_tokens"] * output_price
    ) / 1_000_000
    return {
        "created_at": int(time.time()),
        "model": model,
        "route_id": route["id"],
        "origin": route["origin"],
        "destination": route["destination"],
        "total_route_nodes": len(route["nodes"]),
        "evaluated_nodes": len(rows),
        "complete": len(rows) == len(route["nodes"]),
        "overall_accuracy": round(
            sum(row["correct"] for row in rows) / len(rows), 4
        )
        if rows
        else 0.0,
        "keypoint_accuracy": round(
            sum(row["correct"] for row in keypoints) / len(keypoints), 4
        )
        if keypoints
        else 0.0,
        "prediction_distribution": dict(Counter(row["prediction"] for row in rows)),
        "gold_distribution": dict(Counter(row["gold_action"] for row in rows)),
        "count_diagnostics": {
            "count_decrease_nodes": count_decreases,
            "count_jump_nodes": count_jumps,
            "possible_duplicate_count_nodes": duplicate_count_changes,
        },
        "usage": dict(usage),
        "cost": {
            "estimated_usd": round(estimated_cost, 6),
            "rates_per_million_tokens": {
                "input_usd": input_price,
                "output_usd": output_price,
            },
        },
        "segments": segment_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/globnav_bench/instruction_annotations_auto.jsonl"),
    )
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evals/gpt-5.4-mini_full_route_489"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route = load_route(args.input, args.route_index)
    nodes = route["nodes"]
    segments = instruction_segments(route)
    cumulative = cumulative_distances(nodes)
    headings = incoming_headings(nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    rows = []
    if predictions_path.exists():
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for expected, row in enumerate(rows):
        if row["node_index"] != expected:
            raise ValueError("existing predictions are not a contiguous prefix")

    start = len(rows)
    end = len(nodes)
    if args.max_steps:
        end = min(end, args.max_steps)
    if start >= end:
        print(f"nothing to do: {start}/{end} nodes already evaluated")
    else:
        print(f"prefetching street views for nodes {start}..{end - 1}", flush=True)
        provider = GlobalStreetViewProvider(size="640x640")
        images = prefetch_images(
            provider, nodes, headings, start, end, args.workers
        )

        from openai import OpenAI

        client = OpenAI(api_key=get_openai_api_key(), timeout=120.0)
        for index in range(start, end):
            segment = active_segment(segments, index)
            if index == segment["source_index"]:
                state_before = {
                    "completed_relevant_events": 0,
                    "event_type": "none",
                    "event_phase": "none",
                    "event_description": "",
                    "previous_action": (
                        rows[-1]["prediction"] if rows else "none"
                    ),
                }
            else:
                state_before = rows[-1]["model_state_after"]
            try:
                row = evaluate_step(
                    client=client,
                    model=args.model,
                    route=route,
                    nodes=nodes,
                    images=images,
                    headings=headings,
                    cumulative=cumulative,
                    index=index,
                    segment=segment,
                    state_before=state_before,
                    retries=args.retries,
                )
            except Exception as exc:
                print(f"stopping at node {index}: {type(exc).__name__}: {exc}")
                break
            rows.append(row)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            parsed = row["model_interpretation"]
            print(
                f"[{index + 1}/{len(nodes)}] gold={row['gold_action']:<7} "
                f"pred={row['prediction']:<7} event={parsed.get('event_type')} "
                f"phase={parsed.get('event_phase')} "
                f"count={parsed.get('completed_relevant_events')}",
                flush=True,
            )

    summary = summarize(route, rows, segments, args.model)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in (
        "evaluated_nodes", "complete", "overall_accuracy",
        "keypoint_accuracy", "usage", "cost",
    )}, indent=2))
    print(f"wrote {predictions_path} and {summary_path}")


if __name__ == "__main__":
    main()
