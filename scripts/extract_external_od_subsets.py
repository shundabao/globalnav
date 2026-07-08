#!/usr/bin/env python3
"""Build deterministic OD evaluation subsets from original dataset annotations.

No language model or manual sample list is used. The script only filters original
test examples using slot labels, dialogue acts, span annotations, and BIO tags.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MULTIWOZ_DOMAINS = {
    "taxi": ("taxi-departure", "taxi-destination"),
    "train": ("train-departure", "train-destination"),
}
MULTIWOZ_REQUEST_NAMES = {
    "taxi-departure": "Depart",
    "taxi-destination": "Dest",
    "train-departure": "Depart",
    "train-destination": "Dest",
}
SGD_ENDPOINTS = {
    "Buses_3": ("from_city", "to_city"),
    "Flights_4": ("origin_airport", "destination_airport"),
    "Trains_1": ("from", "to"),
}
EMPTY_VALUES = {None, "", "not mentioned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiwoz-test", type=Path, required=True)
    parser.add_argument("--multiwoz-acts", type=Path, required=True)
    parser.add_argument("--sgd-test-dir", type=Path, required=True)
    parser.add_argument(
        "--atis-dir",
        type=Path,
        required=True,
        help="Directory containing Hugging Face rows API JSON chunks.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def literal_value_in_utterance(value: str, utterance: str) -> bool:
    """Conservative lexical filter; it does not create or alter a gold label."""
    return normalized_text(value) in normalized_text(utterance)


def same_endpoint(origin: str, destination: str) -> bool:
    return normalized_text(origin) == normalized_text(destination)


def multiwoz_requested(
    acts: Any,
    domain: str,
    missing_slot: str,
) -> bool:
    if not isinstance(acts, dict):
        return False
    expected = MULTIWOZ_REQUEST_NAMES[missing_slot].lower()
    for act_name, slot_values in acts.items():
        if act_name.lower() != f"{domain}-request":
            continue
        if any(pair[0].lower() == expected for pair in slot_values):
            return True
    return False


def extract_multiwoz(
    test_path: Path,
    acts_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dialogues = read_json(test_path)
    dialogue_acts = read_json(acts_path)
    explicit_candidates = 0
    clarification_candidates = 0
    explicit: list[dict[str, Any]] = []
    clarification: list[dict[str, Any]] = []

    for dialogue in dialogues:
        source_id = dialogue["dialogue_idx"]
        act_id = source_id.removesuffix(".json")
        turns = dialogue["dialogue"]
        for index, turn in enumerate(turns):
            delta = dict(turn["turn_label"])
            state = dict(turn["belief_state"])

            for domain, (origin_slot, destination_slot) in MULTIWOZ_DOMAINS.items():
                endpoint_slots = {origin_slot, destination_slot}
                present = {
                    slot for slot in endpoint_slots if delta.get(slot) not in EMPTY_VALUES
                }

                if present == endpoint_slots:
                    explicit_candidates += 1
                    origin = delta[origin_slot]
                    destination = delta[destination_slot]
                    if same_endpoint(origin, destination):
                        continue
                    lexical_evidence = {
                        "origin": literal_value_in_utterance(origin, turn["usr"]),
                        "destination": literal_value_in_utterance(
                            destination, turn["usr"]
                        ),
                    }
                    if not all(lexical_evidence.values()):
                        continue
                    explicit.append(
                        {
                            "sample_id": f"multiwoz24:{source_id}:{index}:explicit",
                            "dataset": "MultiWOZ 2.4",
                            "split": "test",
                            "task_type": "explicit_od",
                            "source_id": source_id,
                            "turn_index": index,
                            "domain": domain,
                            "input_utterance": turn["usr"],
                            "gold": {
                                "origin": origin,
                                "destination": destination,
                                "origin_slot": origin_slot,
                                "destination_slot": destination_slot,
                            },
                            "selection_evidence": {
                                "both_slots_in_original_turn_label": True,
                                "gold_values_literal_in_input": lexical_evidence,
                            },
                            "raw": {"turn": turn},
                        }
                    )
                    continue

                if len(present) != 1 or index + 1 >= len(turns):
                    continue
                known_slot = next(iter(present))
                missing_slot = next(iter(endpoint_slots - present))
                if state.get(missing_slot) not in EMPTY_VALUES:
                    continue

                answer_turn = turns[index + 1]
                answer_delta = dict(answer_turn["turn_label"])
                if answer_delta.get(missing_slot) in EMPTY_VALUES:
                    continue
                system_acts = dialogue_acts.get(act_id, {}).get(str(index + 1), {})
                if not multiwoz_requested(system_acts, domain, missing_slot):
                    continue

                clarification_candidates += 1
                if not literal_value_in_utterance(delta[known_slot], turn["usr"]):
                    continue
                if not literal_value_in_utterance(
                    answer_delta[missing_slot], answer_turn["usr"]
                ):
                    continue

                final_state = dict(answer_turn["belief_state"])
                values = {
                    origin_slot: final_state[origin_slot],
                    destination_slot: final_state[destination_slot],
                }
                value_utterances = {
                    slot: (
                        answer_turn["usr"] if slot in answer_delta else turn["usr"]
                    )
                    for slot in endpoint_slots
                }
                if any(
                    not literal_value_in_utterance(values[slot], value_utterances[slot])
                    for slot in endpoint_slots
                ):
                    continue
                if same_endpoint(values[origin_slot], values[destination_slot]):
                    continue
                clarification.append(
                    {
                        "sample_id": (
                            f"multiwoz24:{source_id}:{index}:clarification"
                        ),
                        "dataset": "MultiWOZ 2.4",
                        "split": "test",
                        "task_type": "clarification_resolution",
                        "source_id": source_id,
                        "turn_index": index,
                        "domain": domain,
                        "input_utterance": turn["usr"],
                        "gold": {
                            "origin": values[origin_slot],
                            "destination": values[destination_slot],
                            "origin_slot": origin_slot,
                            "destination_slot": destination_slot,
                            "missing_endpoint": (
                                "origin"
                                if missing_slot == origin_slot
                                else "destination"
                            ),
                        },
                        "original_clarification": {
                            "system": answer_turn["sys"],
                            "user_answer": answer_turn["usr"],
                        },
                        "selection_evidence": {
                            "missing_slot_absent_from_belief_state": True,
                            "system_request_act": {
                                "slot": missing_slot,
                                "raw_dialogue_acts": system_acts,
                            },
                            "answer_in_next_original_turn_label": True,
                            "gold_endpoints_from_answer_turn_belief_state": True,
                            "gold_values_literal_in_respective_user_turns": True,
                        },
                        "raw": {
                            "initial_turn": turn,
                            "answer_turn": answer_turn,
                        },
                    }
                )

    stats = {
        "source_dialogues": len(dialogues),
        "annotation_candidates": {
            "explicit_od": explicit_candidates,
            "clarification_resolution": clarification_candidates,
        },
        "strict_subset": {
            "explicit_od": len(explicit),
            "clarification_resolution": len(clarification),
        },
        "strict_subset_by_domain": {
            "explicit_od": dict(Counter(row["domain"] for row in explicit)),
            "clarification_resolution": dict(
                Counter(row["domain"] for row in clarification)
            ),
        },
    }
    return explicit, clarification, stats


def iter_sgd_dialogues(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("dialogues_*.json")):
        yield from read_json(path)


def sgd_frame(turn: dict[str, Any], service: str) -> dict[str, Any] | None:
    return next(
        (frame for frame in turn.get("frames", []) if frame["service"] == service),
        None,
    )


def sgd_informs(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        action["slot"]: action
        for action in frame.get("actions", [])
        if action.get("act") == "INFORM"
        and action.get("slot")
        and (action.get("canonical_values") or action.get("values"))
    }


def sgd_action_value(action: dict[str, Any], key: str) -> str:
    values = action.get(key) or action.get("values") or []
    return values[0]


def extract_sgd(
    test_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    explicit: list[dict[str, Any]] = []
    clarification: list[dict[str, Any]] = []
    source_dialogues = 0

    for dialogue in iter_sgd_dialogues(test_directory):
        source_dialogues += 1
        turns = dialogue["turns"]
        for index, turn in enumerate(turns):
            if turn["speaker"] != "USER":
                continue
            for service, (origin_slot, destination_slot) in SGD_ENDPOINTS.items():
                frame = sgd_frame(turn, service)
                if frame is None:
                    continue
                informs = sgd_informs(frame)
                endpoint_slots = {origin_slot, destination_slot}
                present = endpoint_slots & informs.keys()
                span_slots = {slot["slot"] for slot in frame.get("slots", [])}

                if present == endpoint_slots and endpoint_slots <= span_slots:
                    origin_value = sgd_action_value(
                        informs[origin_slot], "canonical_values"
                    )
                    destination_value = sgd_action_value(
                        informs[destination_slot], "canonical_values"
                    )
                    if same_endpoint(origin_value, destination_value):
                        continue
                    explicit.append(
                        {
                            "sample_id": (
                                f"sgd:{dialogue['dialogue_id']}:{index}:"
                                f"{service}:explicit"
                            ),
                            "dataset": "Schema-Guided Dialogue",
                            "split": "test",
                            "task_type": "explicit_od",
                            "source_id": dialogue["dialogue_id"],
                            "turn_index": index,
                            "service": service,
                            "input_utterance": turn["utterance"],
                            "gold": {
                                "origin": origin_value,
                                "destination": destination_value,
                                "origin_surface": sgd_action_value(
                                    informs[origin_slot], "values"
                                ),
                                "destination_surface": sgd_action_value(
                                    informs[destination_slot], "values"
                                ),
                                "origin_slot": origin_slot,
                                "destination_slot": destination_slot,
                            },
                            "selection_evidence": {
                                "both_original_inform_actions": True,
                                "both_original_span_annotations": True,
                            },
                            "raw": {"turn": turn, "service_frame": frame},
                        }
                    )
                    continue

                if len(present) != 1 or index + 2 >= len(turns):
                    continue
                known_slot = next(iter(present))
                missing_slot = next(iter(endpoint_slots - present))
                if frame.get("state", {}).get("slot_values", {}).get(missing_slot):
                    continue
                if known_slot not in span_slots:
                    continue

                system_turn = turns[index + 1]
                answer_turn = turns[index + 2]
                if (
                    system_turn["speaker"] != "SYSTEM"
                    or answer_turn["speaker"] != "USER"
                ):
                    continue
                system_frame = sgd_frame(system_turn, service)
                answer_frame = sgd_frame(answer_turn, service)
                if system_frame is None or answer_frame is None:
                    continue
                requested = any(
                    action.get("act") == "REQUEST"
                    and action.get("slot") == missing_slot
                    for action in system_frame.get("actions", [])
                )
                answer_informs = sgd_informs(answer_frame)
                answer_span_slots = {
                    slot["slot"] for slot in answer_frame.get("slots", [])
                }
                if (
                    not requested
                    or missing_slot not in answer_informs
                    or missing_slot not in answer_span_slots
                ):
                    continue

                actions = {
                    known_slot: informs[known_slot],
                    missing_slot: answer_informs[missing_slot],
                }
                for endpoint_slot in endpoint_slots:
                    if endpoint_slot in answer_informs:
                        actions[endpoint_slot] = answer_informs[endpoint_slot]
                origin_value = sgd_action_value(
                    actions[origin_slot], "canonical_values"
                )
                destination_value = sgd_action_value(
                    actions[destination_slot], "canonical_values"
                )
                if same_endpoint(origin_value, destination_value):
                    continue
                clarification.append(
                    {
                        "sample_id": (
                            f"sgd:{dialogue['dialogue_id']}:{index}:"
                            f"{service}:clarification"
                        ),
                        "dataset": "Schema-Guided Dialogue",
                        "split": "test",
                        "task_type": "clarification_resolution",
                        "source_id": dialogue["dialogue_id"],
                        "turn_index": index,
                        "service": service,
                        "input_utterance": turn["utterance"],
                        "gold": {
                            "origin": origin_value,
                            "destination": destination_value,
                            "origin_surface": sgd_action_value(
                                actions[origin_slot], "values"
                            ),
                            "destination_surface": sgd_action_value(
                                actions[destination_slot], "values"
                            ),
                            "origin_slot": origin_slot,
                            "destination_slot": destination_slot,
                            "missing_endpoint": (
                                "origin"
                                if missing_slot == origin_slot
                                else "destination"
                            ),
                        },
                        "original_clarification": {
                            "system": system_turn["utterance"],
                            "user_answer": answer_turn["utterance"],
                        },
                        "selection_evidence": {
                            "missing_slot_absent_from_state": True,
                            "original_system_request_action": missing_slot,
                            "answer_original_inform_action": True,
                            "answer_turn_endpoint_updates_override_initial_values": True,
                            "original_span_annotations": {
                                "known_endpoint": True,
                                "answer_endpoint": True,
                            },
                        },
                        "raw": {
                            "initial_turn": turn,
                            "system_turn": system_turn,
                            "answer_turn": answer_turn,
                        },
                    }
                )

    stats = {
        "source_dialogues": source_dialogues,
        "strict_subset": {
            "explicit_od": len(explicit),
            "clarification_resolution": len(clarification),
        },
        "strict_subset_by_service": {
            "explicit_od": dict(Counter(row["service"] for row in explicit)),
            "clarification_resolution": dict(
                Counter(row["service"] for row in clarification)
            ),
        },
    }
    return explicit, clarification, stats


def atis_spans(text: str, ner: str) -> list[dict[str, Any]]:
    tokens = text.split()
    tags = ner.split()
    if len(tokens) != len(tags):
        raise ValueError(f"ATIS token/tag length mismatch: {text}")
    spans: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        tag = tags[index]
        if not tag.startswith("B-"):
            index += 1
            continue
        slot = tag[2:]
        end = index + 1
        while end < len(tokens) and tags[end] == f"I-{slot}":
            end += 1
        spans.append(
            {
                "slot": slot,
                "value": " ".join(tokens[index:end]),
                "token_start": index,
                "token_end": end,
            }
        )
        index = end
    return spans


def load_atis_rows(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(
        directory.glob("*.json"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        rows.extend(item["row"] for item in payload["rows"])
    return rows


def extract_atis(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_rows = load_atis_rows(directory)
    all_annotated: list[dict[str, Any]] = []
    unambiguous: list[dict[str, Any]] = []
    multi_endpoint: list[dict[str, Any]] = []

    for row in source_rows:
        spans = atis_spans(row["text"], row["ner"])
        origins = [span for span in spans if span["slot"].startswith("fromloc.")]
        destinations = [span for span in spans if span["slot"].startswith("toloc.")]
        if not origins or not destinations:
            continue
        record = {
            "sample_id": f"atis:{row['id']}:explicit",
            "dataset": "ATIS",
            "split": "test",
            "task_type": "explicit_od",
            "source_id": row["id"],
            "intent": row["intent"],
            "input_utterance": row["text"],
            "gold": {
                "origins": origins,
                "destinations": destinations,
            },
            "selection_evidence": {
                "original_bio_fromloc_spans": len(origins),
                "original_bio_toloc_spans": len(destinations),
            },
            "raw": row,
        }
        all_annotated.append(record)
        if len(origins) == 1 and len(destinations) == 1:
            record["gold"]["origin"] = origins[0]["value"]
            record["gold"]["destination"] = destinations[0]["value"]
            record["gold"]["origin_slot"] = origins[0]["slot"]
            record["gold"]["destination_slot"] = destinations[0]["slot"]
            unambiguous.append(record)
        else:
            multi_endpoint.append(record)

    city_pairs = sum(
        row["gold"]["origin_slot"] == "fromloc.city_name"
        and row["gold"]["destination_slot"] == "toloc.city_name"
        for row in unambiguous
    )
    stats = {
        "source_rows": len(source_rows),
        "rows_with_fromloc_and_toloc": len(all_annotated),
        "strict_single_pair_subset": len(unambiguous),
        "strict_single_pair_city_to_city": city_pairs,
        "multi_endpoint_stress_subset": len(multi_endpoint),
        "strict_subset_by_slot_pair": {
            f"{origin} -> {destination}": count
            for (origin, destination), count in sorted(
                Counter(
                    (
                        row["gold"]["origin_slot"],
                        row["gold"]["destination_slot"],
                    )
                    for row in unambiguous
                ).items()
            )
        },
    }
    return unambiguous, multi_endpoint, stats


def main() -> None:
    args = parse_args()
    mw_explicit, mw_clarification, mw_stats = extract_multiwoz(
        args.multiwoz_test,
        args.multiwoz_acts,
    )
    sgd_explicit, sgd_clarification, sgd_stats = extract_sgd(args.sgd_test_dir)
    atis_explicit, atis_multi, atis_stats = extract_atis(args.atis_dir)

    outputs = {
        "multiwoz24_explicit_od.jsonl": mw_explicit,
        "multiwoz24_clarification.jsonl": mw_clarification,
        "sgd_explicit_od.jsonl": sgd_explicit,
        "sgd_clarification.jsonl": sgd_clarification,
        "atis_explicit_single_pair.jsonl": atis_explicit,
        "atis_multi_endpoint_stress.jsonl": atis_multi,
    }
    for filename, rows in outputs.items():
        write_jsonl(args.output_dir / filename, rows)

    summary = {
        "method": (
            "Deterministic filtering of original test annotations; no LLM, "
            "manual sample IDs, or generated labels."
        ),
        "recommended_primary_evaluation": {
            "explicit_od": {
                "MultiWOZ 2.4": len(mw_explicit),
                "Schema-Guided Dialogue": len(sgd_explicit),
                "ATIS": len(atis_explicit),
                "total": len(mw_explicit) + len(sgd_explicit) + len(atis_explicit),
            },
            "clarification_resolution": {
                "MultiWOZ 2.4": len(mw_clarification),
                "Schema-Guided Dialogue": len(sgd_clarification),
                "ATIS": 0,
                "total": len(mw_clarification) + len(sgd_clarification),
            },
        },
        "datasets": {
            "MultiWOZ 2.4": mw_stats,
            "Schema-Guided Dialogue": sgd_stats,
            "ATIS": atis_stats,
        },
        "files": {filename: len(rows) for filename, rows in outputs.items()},
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
