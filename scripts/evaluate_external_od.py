#!/usr/bin/env python3
"""Evaluate GLOBALNAV on small external origin/destination benchmark samples.

The script normalizes 10 examples each from MultiWOZ 2.4, SGD, and ATIS,
runs the same clarifier and modular trip builder used by the web demo, and
reports both diagnostic metrics and a strict end-to-end success rate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from globe_nav.clarification import InstructionClarifier
from globe_nav.config import load_env
from globe_nav.planner.environment import MobilityEnvironment
from globe_nav.planner.trip_builder import ModularTripBuilder


MULTIWOZ_SELECTION = [
    ("SNG0073.json", 0, False),
    ("MUL0671.json", 1, False),
    ("MUL1489.json", 6, False),
    ("MUL1642.json", 0, False),
    ("PMUL0550.json", 0, False),
    ("SNG0263.json", 0, False),
    ("MUL1575.json", 3, True),
    ("PMUL1172.json", 2, True),
    ("MUL2162.json", 0, True),
    ("MUL0264.json", 3, True),
]

SGD_SELECTION = [
    ("27_00000", 2, "Trains_1", False),
    ("27_00075", 0, "Buses_3", False),
    ("27_00090", 0, "Buses_3", False),
    ("26_00019", 6, "Flights_4", False),
    ("26_00024", 4, "Flights_4", False),
    ("27_00093", 2, "Buses_3", False),
    ("27_00002", 0, "Trains_1", True),
    ("27_00070", 2, "Buses_3", True),
    ("26_00001", 2, "Flights_4", True),
    ("26_00006", 4, "Flights_4", True),
]

SGD_ENDPOINT_SLOTS = {
    "Flights_4": ("origin_airport", "destination_airport"),
    "Buses_3": ("from_city", "to_city"),
    "Trains_1": ("from", "to"),
}

ATIS_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=pfsv/atis&config=default&split=test&offset=0&length=100"
)
ATIS_SELECTION = [0, 1, 2, 4, 6, 11, 17, 18, 31, 77]

ALIASES = {
    "st johns": "saint johns",
    "st john s": "saint johns",
    "chi town": "chicago",
    "new york city": "new york",
    "nyc": "new york",
    "ny": "new york",
    "san fran": "san francisco",
    "sf": "san francisco",
    "sfo": "san francisco",
    "la": "los angeles",
    "lax": "los angeles",
    "sd": "san diego",
    "vegas": "las vegas",
    "cdmx": "mexico city",
    "atl": "atlanta",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def labels_dict(turn: dict) -> dict[str, str]:
    return dict(turn.get("turn_label", []))


def endpoint_domain(labels: dict[str, str]) -> tuple[str, str, str] | None:
    for domain in ("taxi", "train"):
        origin_slot = f"{domain}-departure"
        destination_slot = f"{domain}-destination"
        if origin_slot in labels or destination_slot in labels:
            return domain, origin_slot, destination_slot
    return None


def load_multiwoz_samples(path: Path) -> list[dict]:
    dialogues = json.loads(path.read_text(encoding="utf-8"))
    by_id = {dialogue["dialogue_idx"]: dialogue for dialogue in dialogues}
    samples = []

    for dialogue_id, turn_idx, needs_clarification in MULTIWOZ_SELECTION:
        dialogue = by_id[dialogue_id]
        turns = dialogue["dialogue"]
        turn = next(item for item in turns if item["turn_idx"] == turn_idx)
        current_labels = labels_dict(turn)
        domain_info = endpoint_domain(current_labels)
        if not domain_info:
            raise ValueError(f"No transport endpoint label in {dialogue_id} turn {turn_idx}")
        domain, origin_slot, destination_slot = domain_info

        later_labels: dict[str, str] = {}
        later_answers: dict[str, str] = {}
        for later in turns:
            if later["turn_idx"] <= turn_idx:
                continue
            for slot, value in labels_dict(later).items():
                later_labels.setdefault(slot, value)
                later_answers.setdefault(slot, later["usr"])

        origin = current_labels.get(origin_slot) or later_labels.get(origin_slot)
        destination = current_labels.get(destination_slot) or later_labels.get(destination_slot)
        if not origin or not destination:
            raise ValueError(f"Missing gold endpoint in {dialogue_id} turn {turn_idx}")

        missing_slot = None
        if origin_slot not in current_labels:
            missing_slot = origin_slot
        elif destination_slot not in current_labels:
            missing_slot = destination_slot
        derived_clarification = missing_slot is not None
        if needs_clarification != derived_clarification:
            raise ValueError(
                f"Clarification label mismatch in {dialogue_id} turn {turn_idx}: "
                f"selected={needs_clarification} derived={derived_clarification}"
            )

        answer = later_answers.get(missing_slot, "") if missing_slot else ""
        if needs_clarification and not answer:
            raise ValueError(f"No clarification answer in {dialogue_id} turn {turn_idx}")

        instruction = turn["usr"]
        if domain == "taxi":
            instruction = (
                "Travel context: local places without a city name are in Cambridge, "
                "United Kingdom.\n"
                f"User request: {turn['usr']}"
            )
        samples.append(
            {
                "sample_id": f"multiwoz24:{dialogue_id}:{turn_idx}",
                "dataset": "multiwoz24",
                "source_id": dialogue_id,
                "turn_index": turn_idx,
                "domain": domain,
                "instruction": instruction,
                "raw_utterance": turn["usr"],
                "gold_origin": origin,
                "gold_destination": destination,
                "expected_clarification": needs_clarification,
                "missing_endpoint": (
                    "origin" if missing_slot == origin_slot else
                    "destination" if missing_slot == destination_slot else None
                ),
                "clarification_answer": answer,
            }
        )
    return samples


def iter_sgd_dialogues(directory: Path) -> Iterable[dict]:
    for path in sorted(directory.glob("dialogues_*.json")):
        yield from json.loads(path.read_text(encoding="utf-8"))


def user_inform_values(frame: dict) -> dict[str, str]:
    values = {}
    for action in frame.get("actions", []):
        if action.get("act") == "INFORM" and action.get("slot") and action.get("values"):
            values[action["slot"]] = action["values"][0]
    return values


def find_sgd_frame(turn: dict, service: str) -> dict | None:
    return next((frame for frame in turn.get("frames", []) if frame["service"] == service), None)


def load_sgd_samples(directory: Path) -> list[dict]:
    dialogues = {dialogue["dialogue_id"]: dialogue for dialogue in iter_sgd_dialogues(directory)}
    samples = []

    for dialogue_id, turn_idx, service, needs_clarification in SGD_SELECTION:
        dialogue = dialogues[dialogue_id]
        turns = dialogue["turns"]
        turn = turns[turn_idx]
        frame = find_sgd_frame(turn, service)
        if turn["speaker"] != "USER" or frame is None:
            raise ValueError(f"Invalid SGD selection {dialogue_id} turn {turn_idx}")

        origin_slot, destination_slot = SGD_ENDPOINT_SLOTS[service]
        current = user_inform_values(frame)
        state = frame.get("state", {}).get("slot_values", {})
        origin = current.get(origin_slot) or next(iter(state.get(origin_slot, [])), "")
        destination = current.get(destination_slot) or next(iter(state.get(destination_slot, [])), "")

        missing_slot = None
        if origin_slot not in current:
            missing_slot = origin_slot
        elif destination_slot not in current:
            missing_slot = destination_slot
        derived_clarification = missing_slot is not None
        if needs_clarification != derived_clarification:
            raise ValueError(
                f"Clarification label mismatch in {dialogue_id} turn {turn_idx}: "
                f"selected={needs_clarification} derived={derived_clarification}"
            )

        answer = ""
        for later in turns[turn_idx + 1:]:
            if later["speaker"] != "USER":
                continue
            later_frame = find_sgd_frame(later, service)
            if not later_frame:
                continue
            later_values = user_inform_values(later_frame)
            if missing_slot and missing_slot in later_values:
                answer = later["utterance"]
                if missing_slot == origin_slot:
                    origin = later_values[missing_slot]
                else:
                    destination = later_values[missing_slot]
                break

        if not origin or not destination:
            raise ValueError(f"Missing SGD gold endpoint in {dialogue_id} turn {turn_idx}")
        if needs_clarification and not answer:
            raise ValueError(f"No SGD clarification answer in {dialogue_id} turn {turn_idx}")

        samples.append(
            {
                "sample_id": f"sgd:{dialogue_id}:{turn_idx}",
                "dataset": "sgd",
                "source_id": dialogue_id,
                "turn_index": turn_idx,
                "domain": service,
                "instruction": turn["utterance"],
                "raw_utterance": turn["utterance"],
                "gold_origin": origin,
                "gold_destination": destination,
                "expected_clarification": needs_clarification,
                "missing_endpoint": (
                    "origin" if missing_slot == origin_slot else
                    "destination" if missing_slot == destination_slot else None
                ),
                "clarification_answer": answer,
            }
        )
    return samples


def extract_bio_value(text: str, tags_text: str, target: str) -> str:
    tokens = text.split()
    tags = tags_text.split()
    if len(tokens) != len(tags):
        return ""
    pieces = []
    active = False
    for token, tag in zip(tokens, tags):
        label = tag[2:] if tag.startswith(("B-", "I-")) else ""
        if label == target:
            if tag.startswith("B-") and pieces:
                break
            pieces.append(token)
            active = True
        elif active:
            break
    return " ".join(pieces)


def load_atis_samples() -> list[dict]:
    response = requests.get(ATIS_ROWS_URL, timeout=30)
    response.raise_for_status()
    candidates = []
    for wrapped in response.json()["rows"]:
        row = wrapped["row"]
        origin = extract_bio_value(row["text"], row["ner"], "fromloc.city_name")
        destination = extract_bio_value(row["text"], row["ner"], "toloc.city_name")
        if origin and destination:
            candidates.append((row, origin, destination))

    by_id = {row["id"]: (row, origin, destination) for row, origin, destination in candidates}
    chosen = [by_id[row_id] for row_id in ATIS_SELECTION]
    return [
        {
            "sample_id": f"atis:test:{row['id']}",
            "dataset": "atis",
            "source_id": str(row["id"]),
            "turn_index": 0,
            "domain": row["intent"],
            "instruction": row["text"],
            "raw_utterance": row["text"],
            "gold_origin": origin,
            "gold_destination": destination,
            "expected_clarification": False,
            "missing_endpoint": None,
            "clarification_answer": "",
        }
        for row, origin, destination in chosen
    ]


def basic_normalize_place(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return re.sub(r"\s+", " ", value).strip()


def normalize_place(value: str) -> str:
    normalized = basic_normalize_place(value)
    return ALIASES.get(normalized, normalized)


def place_match(gold: str, predicted: str) -> bool:
    raw_gold = basic_normalize_place(gold)
    raw_pred = basic_normalize_place(predicted)
    if raw_gold and raw_pred:
        raw_gold_compact = raw_gold.replace(" ", "")
        raw_pred_compact = raw_pred.replace(" ", "")
        if raw_gold_compact in raw_pred_compact or raw_pred_compact in raw_gold_compact:
            return True

    gold_norm = normalize_place(gold)
    pred_norm = normalize_place(predicted)
    if not gold_norm or not pred_norm:
        return False
    gold_compact = gold_norm.replace(" ", "")
    pred_compact = pred_norm.replace(" ", "")
    if gold_compact in pred_compact or pred_compact in gold_compact:
        return True
    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())
    return len(gold_tokens & pred_tokens) / max(len(gold_tokens), 1) >= 0.8


def exact_place_match(gold: str, predicted: str) -> bool:
    gold_norm = normalize_place(gold)
    pred_norm = normalize_place(predicted)
    return bool(gold_norm and pred_norm and gold_norm == pred_norm)


def matches_any(
    gold: str,
    variants: list[str] | None,
    predicted: str,
    *,
    exact: bool = False,
) -> bool:
    matcher = exact_place_match if exact else place_match
    candidates = variants or [gold]
    return any(matcher(candidate, predicted) for candidate in candidates)


EMPTY_PREDICTED_ENDPOINTS = {
    "",
    "none",
    "null",
    "not mentioned",
    "unknown",
    "unspecified",
    "n/a",
    "na",
}


ORIGIN_TARGET_TERMS = (
    "origin",
    "origination",
    "departure",
    "depart",
    "departing",
    "pickup",
    "pick up",
    "picked up",
    "pick-up",
    "start",
    "starting",
    "source",
    "from where",
    "where from",
    "leave from",
    "leaving from",
    "coming from",
    "traveling from",
    "travelling from",
    "travel from",
    "from city",
    "from airport",
    "from station",
)

DESTINATION_TARGET_TERMS = (
    "destination",
    "dest",
    "arrival location",
    "dropoff",
    "drop off",
    "drop-off",
    "to where",
    "where to",
    "go to",
    "going to",
    "headed to",
    "heading to",
    "traveling to",
    "travelling to",
    "travel to",
    "take you",
    "to city",
    "to airport",
    "to station",
)


def endpoint_empty(value: Any) -> bool:
    normalized = basic_normalize_place(str(value or ""))
    return normalized in EMPTY_PREDICTED_ENDPOINTS


def endpoint_target_from_text(category: str, question: str) -> str:
    text = f"{category or ''} {question or ''}".lower()
    origin_score = sum(term in text for term in ORIGIN_TARGET_TERMS)
    destination_score = sum(term in text for term in DESTINATION_TARGET_TERMS)
    if origin_score > destination_score:
        return "origin"
    if destination_score > origin_score:
        return "destination"
    return "unknown"


def infer_clarification_target(
    origin: Any,
    destination: Any,
    questions: list[str] | None,
    categories: list[str] | None = None,
) -> str:
    """Infer which endpoint the system is asking for without an LLM judge."""
    origin_missing = endpoint_empty(origin)
    destination_missing = endpoint_empty(destination)
    if origin_missing and not destination_missing:
        return "origin"
    if destination_missing and not origin_missing:
        return "destination"

    questions = questions or []
    categories = categories or []
    pairs = list(zip(categories, questions))
    if not pairs:
        pairs = [("", question) for question in questions]
    for category, question in pairs:
        target = endpoint_target_from_text(category, question)
        if target != "unknown":
            return target
    return "unknown"


def first_question(result: Any) -> tuple[str, str]:
    if not result.ambiguities:
        return "", ""
    ambiguity = result.ambiguities[0]
    return ambiguity.category, ambiguity.question


def selected_option_verified(plan: Any) -> bool:
    for segment in plan.segments:
        option = next(
            (item for item in segment.options if item.option_id == segment.default_option_id),
            segment.options[0] if segment.options else None,
        )
        if option is None or not option.verified:
            return False
    return True


def evaluate_sample(sample: dict, model: str, cache_dir: str, route_check: str) -> dict:
    started = time.time()
    clarifier = InstructionClarifier(model=model)
    initial = clarifier.analyze(sample["instruction"])
    initial_asked = not initial.is_clear
    ask_act_correct = initial_asked == sample["expected_clarification"]
    initial_question_categories = [ambiguity.category for ambiguity in initial.ambiguities]
    predicted_clarification_target = (
        infer_clarification_target(
            initial.resolved_origin,
            initial.resolved_destination,
            initial.questions,
            initial_question_categories,
        )
        if initial_asked
        else "none"
    )
    clarification_target_correct = (
        True
        if not sample["expected_clarification"]
        else (
            initial_asked
            and predicted_clarification_target == sample.get("missing_endpoint")
        )
    )

    final = initial
    clarification_turns = 0
    should_supply_clarification_answer = bool(
        sample["expected_clarification"]
        and clarification_target_correct
        and sample.get("clarification_answer")
    )
    if not final.is_clear and should_supply_clarification_answer:
        category, question = first_question(final)
        if category:
            final = clarifier.answer_and_reparse(
                category,
                question,
                sample["clarification_answer"],
            )
            clarification_turns += 1

    origin_correct = matches_any(
        sample["gold_origin"],
        sample.get("gold_origin_variants"),
        final.resolved_origin or "",
    )
    destination_correct = matches_any(
        sample["gold_destination"],
        sample.get("gold_destination_variants"),
        final.resolved_destination or "",
    )
    origin_exact = matches_any(
        sample["gold_origin"],
        sample.get("gold_origin_variants"),
        final.resolved_origin or "",
        exact=True,
    )
    destination_exact = matches_any(
        sample["gold_destination"],
        sample.get("gold_destination_variants"),
        final.resolved_destination or "",
        exact=True,
    )
    intent_resolution_success = bool(final.is_clear and origin_correct and destination_correct)
    exact_pair_success = bool(final.is_clear and origin_exact and destination_exact)

    environment_acceptance_success = route_check == "none" and final.is_clear
    route_generated: bool | None = None
    route_error = ""
    route_summary: dict[str, Any] = {}
    if route_check == "geocode" and final.is_clear:
        try:
            environment = MobilityEnvironment(cache_dir=cache_dir, use_online=True)
            origin_waypoint = environment.geocode(final.resolved_origin or "")
            destination_waypoint = environment.geocode(final.resolved_destination or "")
            environment_acceptance_success = bool(origin_waypoint and destination_waypoint)
            route_summary = {
                "origin_geocoded": bool(origin_waypoint),
                "destination_geocoded": bool(destination_waypoint),
                "resolved_plan_origin": origin_waypoint.name if origin_waypoint else "",
                "resolved_plan_destination": destination_waypoint.name if destination_waypoint else "",
                "origin_coordinates": (
                    [origin_waypoint.lat, origin_waypoint.lon] if origin_waypoint else None
                ),
                "destination_coordinates": (
                    [destination_waypoint.lat, destination_waypoint.lon]
                    if destination_waypoint else None
                ),
            }
        except Exception as exc:
            route_error = f"{type(exc).__name__}: {exc}"
    elif route_check == "full" and final.is_clear:
        try:
            builder = ModularTripBuilder(
                env=MobilityEnvironment(cache_dir=cache_dir, use_online=True)
            )
            plan = builder.build(
                final.resolved_origin or "",
                final.resolved_destination or "",
                instruction=sample["instruction"],
            )
            route_generated = bool(
                plan.segments
                and all(segment.options for segment in plan.segments)
            )
            environment_acceptance_success = route_generated
            route_summary = {
                "resolved_plan_origin": plan.origin,
                "resolved_plan_destination": plan.destination,
                "segment_count": len(plan.segments),
                "segment_types": [segment.segment_type for segment in plan.segments],
                "option_counts": [len(segment.options) for segment in plan.segments],
                "available_mode_chains": [
                    [option.mode_chain for option in segment.options]
                    for segment in plan.segments
                ],
                "default_mode_chains": [
                    next(
                        (
                            option.mode_chain
                            for option in segment.options
                            if option.option_id == segment.default_option_id
                        ),
                        segment.options[0].mode_chain if segment.options else "",
                    )
                    for segment in plan.segments
                ],
                "default_total_min": round(plan.default_total_min, 1),
                "default_options_verified": selected_option_verified(plan),
            }
        except Exception as exc:
            route_error = f"{type(exc).__name__}: {exc}"

    strict_end_to_end_success = bool(
        ask_act_correct
        and clarification_target_correct
        and intent_resolution_success
        and environment_acceptance_success
    )
    system_completion_success = bool(
        final.is_clear and environment_acceptance_success
    )

    return {
        **sample,
        "model": model,
        "predicted_initial": {
            "origin": initial.resolved_origin,
            "destination": initial.resolved_destination,
            "is_clear": initial.is_clear,
            "questions": initial.questions,
            "question_categories": initial_question_categories,
        },
        "predicted_final": {
            "origin": final.resolved_origin,
            "destination": final.resolved_destination,
            "is_clear": final.is_clear,
            "questions": final.questions,
        },
        "predicted_clarification_target": predicted_clarification_target,
        "metrics": {
            "initial_asked": initial_asked,
            "ask_act_correct": ask_act_correct,
            "clarification_target_correct": clarification_target_correct,
            "clarification_turns": clarification_turns,
            "origin_correct": origin_correct,
            "destination_correct": destination_correct,
            "origin_exact": origin_exact,
            "destination_exact": destination_exact,
            "exact_pair_success": exact_pair_success,
            "intent_resolution_success": intent_resolution_success,
            "environment_acceptance_success": environment_acceptance_success,
            "route_generated": route_generated,
            "system_completion_success": system_completion_success,
            "strict_end_to_end_success": strict_end_to_end_success,
        },
        "route": route_summary,
        "route_error": route_error,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(row["metrics"][key]) for row in rows) / len(rows) * 100, 1)


def rate_or_none(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return rate(rows, key)


def rescore_prediction(row: dict, route_check: str = "none") -> dict:
    initial = row.get("predicted_initial", {})
    stored_final = row.get("predicted_final", {})
    metrics = row["metrics"]
    initial_asked = not bool(initial.get("is_clear"))
    predicted_clarification_target = (
        infer_clarification_target(
            initial.get("origin"),
            initial.get("destination"),
            initial.get("questions") or [],
            initial.get("question_categories") or [],
        )
        if initial_asked
        else "none"
    )
    clarification_target_correct = (
        True
        if not row["expected_clarification"]
        else (
            initial_asked
            and predicted_clarification_target == row.get("missing_endpoint")
        )
    )
    should_use_clarified_final = bool(
        row["expected_clarification"]
        and clarification_target_correct
        and metrics.get("clarification_turns", 0) > 0
    )
    final = stored_final if should_use_clarified_final else {
        "origin": initial.get("origin"),
        "destination": initial.get("destination"),
        "is_clear": initial.get("is_clear"),
        "questions": initial.get("questions") or [],
    }
    origin_correct = matches_any(
        row["gold_origin"],
        row.get("gold_origin_variants"),
        final.get("origin") or "",
    )
    destination_correct = matches_any(
        row["gold_destination"],
        row.get("gold_destination_variants"),
        final.get("destination") or "",
    )
    origin_exact = matches_any(
        row["gold_origin"],
        row.get("gold_origin_variants"),
        final.get("origin") or "",
        exact=True,
    )
    destination_exact = matches_any(
        row["gold_destination"],
        row.get("gold_destination_variants"),
        final.get("destination") or "",
        exact=True,
    )
    intent_resolution_success = bool(
        final.get("is_clear") and origin_correct and destination_correct
    )
    environment_acceptance_success = (
        bool(final.get("is_clear"))
        if route_check == "none"
        else bool(metrics["environment_acceptance_success"])
    )
    row["predicted_final"] = final
    row["predicted_clarification_target"] = predicted_clarification_target
    metrics.update(
        {
            "initial_asked": initial_asked,
            "ask_act_correct": initial_asked == row["expected_clarification"],
            "clarification_target_correct": clarification_target_correct,
            "clarification_turns": (
                metrics.get("clarification_turns", 0)
                if should_use_clarified_final
                else 0
            ),
            "origin_correct": origin_correct,
            "destination_correct": destination_correct,
            "origin_exact": origin_exact,
            "destination_exact": destination_exact,
            "exact_pair_success": bool(
                final.get("is_clear") and origin_exact and destination_exact
            ),
            "intent_resolution_success": intent_resolution_success,
            "environment_acceptance_success": environment_acceptance_success,
            "system_completion_success": bool(
                final.get("is_clear") and environment_acceptance_success
            ),
            "strict_end_to_end_success": bool(
                initial_asked == row["expected_clarification"]
                and clarification_target_correct
                and intent_resolution_success
                and environment_acceptance_success
            ),
        }
    )
    return row


def summarize(rows: list[dict], model: str, route_check: str) -> dict:
    groups = {"overall": rows}
    for dataset in sorted({row["dataset"] for row in rows}):
        groups[dataset] = [row for row in rows if row["dataset"] == dataset]

    summary_groups = {}
    for name, group in groups.items():
        group_ambiguous = [row for row in group if row["expected_clarification"]]
        summary_groups[name] = {
            "count": len(group),
            "ask_act_accuracy": rate(group, "ask_act_correct"),
            "clarification_target_accuracy": rate_or_none(
                group_ambiguous, "clarification_target_correct"
            ),
            "origin_accuracy": rate(group, "origin_correct"),
            "destination_accuracy": rate(group, "destination_correct"),
            "exact_origin_accuracy": rate(group, "origin_exact"),
            "exact_destination_accuracy": rate(group, "destination_exact"),
            "exact_pair_accuracy": rate(group, "exact_pair_success"),
            "intent_resolution_accuracy": rate(group, "intent_resolution_success"),
            "environment_acceptance_rate": rate(group, "environment_acceptance_success"),
            "system_completion_rate": rate(group, "system_completion_success"),
            "strict_end_to_end_success_rate": rate(group, "strict_end_to_end_success"),
            "full_route_generation_rate": (
                rate(group, "route_generated") if route_check == "full" else None
            ),
        }

    explicit = [row for row in rows if not row["expected_clarification"]]
    ambiguous = [row for row in rows if row["expected_clarification"]]
    if route_check == "none":
        strict_definition = (
            "correct ask/act decision AND, for clarification examples, the "
            "system asks for the gold missing endpoint before receiving the "
            "gold clarification answer AND correct final "
            "origin/destination AND a clear final parser state; geocoding "
            "and routing are not checked"
        )
        completion_definition = (
            "system reaches a clear parser state; geocoding and routing are not checked"
        )
    else:
        strict_definition = (
            "correct ask/act decision AND, for clarification examples, the "
            "system asks for the gold missing endpoint before receiving the "
            "gold clarification answer AND correct final "
            "origin/destination AND the environment accepts/geocodes the endpoints"
        )
        completion_definition = (
            "system reaches a clear intent accepted by the environment, regardless of "
            "whether it asked an unnecessary clarification"
        )

    return {
        "model": model,
        "sample_count": len(rows),
        "route_check": route_check,
        "metric_definition": {
            "strict_end_to_end_success": strict_definition,
            "system_completion_success": completion_definition,
        },
        "groups": summary_groups,
        "clarification_breakdown": {
            "explicit_count": len(explicit),
            "ambiguous_count": len(ambiguous),
            "explicit_no_ask_accuracy": rate(explicit, "ask_act_correct"),
            "ambiguous_ask_accuracy": rate(ambiguous, "ask_act_correct"),
            "ambiguous_target_accuracy": rate(
                ambiguous, "clarification_target_correct"
            ),
            "ambiguous_resolution_accuracy": rate(ambiguous, "intent_resolution_success"),
        },
        "failures": [
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "instruction": row["raw_utterance"],
                "gold_origin": row["gold_origin"],
                "gold_destination": row["gold_destination"],
                "missing_endpoint": row.get("missing_endpoint"),
                "predicted_clarification_target": row.get(
                    "predicted_clarification_target"
                ),
                "initial_questions": row.get("predicted_initial", {}).get(
                    "questions", []
                ),
                "predicted_origin": row["predicted_final"]["origin"],
                "predicted_destination": row["predicted_final"]["destination"],
                "metrics": row["metrics"],
                "route_error": row["route_error"],
            }
            for row in rows
            if not row["metrics"]["strict_end_to_end_success"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiwoz",
        type=Path,
        help="Path to MultiWOZ 2.4 test_dials_manually-modified.json",
    )
    parser.add_argument(
        "--sgd",
        type=Path,
        help="Path to the SGD test directory",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        help="Use an existing normalized samples JSONL instead of source datasets",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evals/external_od_gpt5mini",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "data/cache"),
    )
    parser.add_argument(
        "--route-check",
        choices=("none", "geocode", "full"),
        default="geocode",
        help=(
            "Environment check after parsing: none, endpoint geocoding, or the full "
            "live ModularTripBuilder pipeline"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Normalize and save samples without calling the model",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Evaluate only the selected normalized sample ID; may be repeated",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Recompute deterministic metrics from existing predictions without API calls",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent model requests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed sample IDs from an existing predictions.jsonl.",
    )
    args = parser.parse_args()

    load_env()
    if args.samples_file:
        samples = [
            json.loads(line)
            for line in args.samples_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if not args.multiwoz or not args.sgd:
            parser.error("--multiwoz and --sgd are required unless --samples-file is used")
        samples = (
            load_multiwoz_samples(args.multiwoz)
            + load_sgd_samples(args.sgd)
            + load_atis_samples()
        )
    if not args.samples_file and len(samples) != 30:
        raise ValueError(f"Expected 30 samples, got {len(samples)}")
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [sample for sample in samples if sample["sample_id"] in selected]
        missing = selected - {sample["sample_id"] for sample in samples}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "samples.jsonl", samples)
    if args.prepare_only:
        print(f"Prepared {len(samples)} samples at {args.output_dir / 'samples.jsonl'}")
        return 0
    if args.rescore_only:
        predictions_path = args.output_dir / "predictions.jsonl"
        predictions = [
            rescore_prediction(json.loads(line), route_check=args.route_check)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        write_jsonl(predictions_path, predictions)
        summary = summarize(predictions, args.model, route_check=args.route_check)
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary["groups"], indent=2, ensure_ascii=False))
        return 0

    predictions_path = args.output_dir / "predictions.jsonl"
    completed: dict[str, dict] = {}
    if args.resume and predictions_path.exists():
        samples_by_id = {sample["sample_id"]: sample for sample in samples}
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["sample_id"] not in samples_by_id:
                    continue
                row = {**row, **samples_by_id[row["sample_id"]]}
                row = rescore_prediction(row, route_check=args.route_check)
                completed[row["sample_id"]] = row
        print(f"Resuming with {len(completed)} completed samples.", flush=True)

    def run_one(sample: dict) -> dict:
        try:
            return evaluate_sample(
                sample,
                model=args.model,
                cache_dir=args.cache_dir,
                route_check=args.route_check,
            )
        except Exception as exc:
            return {
                **sample,
                "model": args.model,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "predicted_initial": {
                    "origin": "",
                    "destination": "",
                    "is_clear": False,
                    "questions": [],
                    "question_categories": [],
                },
                "predicted_final": {
                    "origin": "",
                    "destination": "",
                    "is_clear": False,
                    "questions": [],
                },
                "predicted_clarification_target": "unknown",
                "metrics": {
                    "initial_asked": False,
                    "ask_act_correct": False,
                    "clarification_target_correct": False,
                    "clarification_turns": 0,
                    "origin_correct": False,
                    "destination_correct": False,
                    "origin_exact": False,
                    "destination_exact": False,
                    "exact_pair_success": False,
                    "intent_resolution_success": False,
                    "environment_acceptance_success": False,
                    "route_generated": False if args.route_check == "full" else None,
                    "system_completion_success": False,
                    "strict_end_to_end_success": False,
                },
                "route": {},
                "route_error": "",
                "elapsed_seconds": 0.0,
            }

    pending = [sample for sample in samples if sample["sample_id"] not in completed]
    print(
        f"Evaluating {len(pending)} pending samples with {args.workers} worker(s).",
        flush=True,
    )
    done_count = len(completed)
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {executor.submit(run_one, sample): sample for sample in pending}
        for future in as_completed(futures):
            sample = futures[future]
            result = future.result()
            completed[sample["sample_id"]] = result
            done_count += 1
            ordered_partial = [
                completed[item["sample_id"]]
                for item in samples
                if item["sample_id"] in completed
            ]
            write_jsonl(predictions_path, ordered_partial)
            print(
                f"[{done_count:03d}/{len(samples)}] {sample['sample_id']} "
                f"intent={result['metrics']['intent_resolution_success']} "
                f"strict_e2e={result['metrics']['strict_end_to_end_success']} "
                f"elapsed={result.get('elapsed_seconds', 0):.1f}s",
                flush=True,
            )

    predictions = [completed[sample["sample_id"]] for sample in samples]
    write_jsonl(predictions_path, predictions)
    summary = summarize(predictions, args.model, route_check=args.route_check)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["groups"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
