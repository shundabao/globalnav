"""Schema helpers for GlobNav-Bench JSONL examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = '0.2'

SPLITS = {
    'planning_feasibility',
    'intent_clarification',
    'route_option_generation',
    'hybrid_follower',
    'stress_test',
}

CLARIFICATION_TYPES = {
    None,
    'missing_origin',
    'missing_destination',
    'ambiguous_place',
    'ambiguous_airport',
    'transport_preference',
    'time_constraint',
}

JSON_SCHEMA: dict[str, Any] = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    '$id': 'https://globalnav.example.org/globnav-bench.schema.json',
    'title': 'GlobNav-Bench example',
    'type': 'object',
    'required': [
        'schema_version', 'id', 'split', 'language', 'instruction',
        'categories', 'gold_intent', 'clarification',
        'route_annotation', 'follower_annotation', 'metadata',
    ],
    'properties': {
        'schema_version': {'const': SCHEMA_VERSION},
        'id': {'type': 'string'},
        'split': {'enum': sorted(SPLITS)},
        'language': {'type': 'string'},
        'instruction': {'type': 'string'},
        'categories': {'type': 'array', 'items': {'type': 'string'}},
        'gold_intent': {'type': 'object'},
        'clarification': {'type': 'object'},
        'route_annotation': {'type': 'object'},
        'follower_annotation': {'type': ['object', 'null']},
        'metadata': {'type': 'object'},
        'annotation_status': {'type': 'string'},
    },
}


def validate_example(example: dict[str, Any]) -> list[str]:
    """Return schema errors; empty list means the example is usable."""
    errors: list[str] = []
    for key in JSON_SCHEMA['required']:
        if key not in example:
            errors.append(f'missing required field: {key}')

    if example.get('schema_version') != SCHEMA_VERSION:
        errors.append(f'unsupported schema_version: {example.get("schema_version")!r}')
    if example.get('split') not in SPLITS:
        errors.append(f'unknown split: {example.get("split")!r}')
    if not example.get('id'):
        errors.append('id must be non-empty')
    if not example.get('instruction'):
        errors.append('instruction must be non-empty')
    if not isinstance(example.get('categories'), list) or not example.get('categories'):
        errors.append('categories must be a non-empty list')

    gold = example.get('gold_intent') or {}
    if not isinstance(gold, dict):
        errors.append('gold_intent must be an object')
    else:
        if 'origin' not in gold:
            errors.append('gold_intent.origin missing')
        if 'destination' not in gold:
            errors.append('gold_intent.destination missing')

    clarification = example.get('clarification') or {}
    if not isinstance(clarification, dict):
        errors.append('clarification must be an object')
    else:
        needs = clarification.get('needs_clarification')
        if not isinstance(needs, bool):
            errors.append('clarification.needs_clarification must be boolean')
        ctype = clarification.get('gold_question_type')
        if ctype not in CLARIFICATION_TYPES:
            errors.append(f'unknown clarification type: {ctype!r}')
        if needs and not clarification.get('gold_question'):
            errors.append('clarification.gold_question required when needs_clarification=true')

    route = example.get('route_annotation') or {}
    if not isinstance(route, dict):
        errors.append('route_annotation must be an object')
    else:
        if 'expected_segment_order' not in route:
            errors.append('route_annotation.expected_segment_order missing')
        if 'segments' not in route:
            errors.append('route_annotation.segments missing')

    if example.get('split') == 'hybrid_follower':
        follower = example.get('follower_annotation')
        if not isinstance(follower, dict):
            errors.append('hybrid_follower examples require follower_annotation object')
        elif not follower.get('system_simulation_oracle'):
            errors.append('follower_annotation.system_simulation_oracle missing')

    return errors


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def write_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False, sort_keys=True) + '\n')


def validation_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    failures = []
    for ex in examples:
        by_split[ex.get('split', '<missing>')] = by_split.get(ex.get('split', '<missing>'), 0) + 1
        errors = validate_example(ex)
        if errors:
            failures.append({'id': ex.get('id'), 'errors': errors})
    return {
        'schema_version': SCHEMA_VERSION,
        'example_count': len(examples),
        'by_split': by_split,
        'valid_count': len(examples) - len(failures),
        'failure_count': len(failures),
        'failures': failures[:50],
    }
