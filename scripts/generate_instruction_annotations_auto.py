#!/usr/bin/env python3
"""Generate route-grounded instruction annotation records automatically."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe_nav.bench.sampler import CITY_SEEDS, POI
from globe_nav.env.transport import Waypoint
from globe_nav.gui.app import _route_nodes_for_selection
from globe_nav.maps.geocoder import haversine_km
from globe_nav.maps.geo import line_interpolate
from globe_nav.planner.models import DetailedLeg, RouteStep
from globe_nav.planner.segment_models import MicroLeg, ModularTripPlan, SegmentOption, TripSegment
from globe_nav.planner.trip_builder import ModularTripBuilder


VAGUE_PLACE_TOKENS = (
    ' city ',
    ' district',
    'downtown',
    'city centre',
    'city center',
)

GROUND_MODES = {'walk', 'drive'}
KEYPOINT_ACTIONS = {'left', 'right', 'u_turn', 'turn_around', 'stop'}
ORDINALS = {1: 'next', 2: 'second', 3: 'third', 4: 'fourth'}
GENERIC_CUE_NAMES = {
    'depart',
    'arrive',
    'turn',
    'continue',
    'new name',
    'fork',
    'roundabout',
    'exit roundabout',
}


def is_concrete_poi(poi: POI) -> bool:
    query = f' {poi.query.lower()} '
    return not any(token in query for token in VAGUE_PLACE_TOKENS)


def candidate_pairs(seed: int) -> list[tuple[object, POI, POI]]:
    pairs = []
    for city in CITY_SEEDS:
        pois = [poi for poi in city.pois if is_concrete_poi(poi)]
        if len(pois) < 2:
            continue
        for origin in pois:
            for dest in pois:
                if origin != dest:
                    pairs.append((city, origin, dest))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def waypoint_from_poi(poi: POI) -> Waypoint:
    return Waypoint(
        name=poi.query,
        lat=poi.lat,
        lon=poi.lon,
        place_type=poi.poi_type,
    )


def fallback_leg(origin: POI, dest: POI, mode: str) -> DetailedLeg:
    distance_km = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)
    speed_kmh = 35.0 if mode == 'drive' else 5.0
    duration_min = max(distance_km / speed_kmh * 60.0, 1.0)
    return DetailedLeg(
        mode=mode,
        from_name=origin.query,
        to_name=dest.query,
        from_lat=origin.lat,
        from_lon=origin.lon,
        to_lat=dest.lat,
        to_lon=dest.lon,
        distance_km=distance_km,
        duration_min=duration_min,
        steps=[RouteStep(f'{mode} from {origin.query} to {dest.query}')],
        geometry=line_interpolate(origin.lat, origin.lon, dest.lat, dest.lon, n=80),
        verified=False,
        note='linear fallback',
    )


def option_from_leg(seg_id: str, leg: DetailedLeg) -> SegmentOption:
    signature = (
        f'{leg.mode}:{leg.from_name}->{leg.to_name}:'
        f'{leg.from_lat:.5f},{leg.from_lon:.5f}->{leg.to_lat:.5f},{leg.to_lon:.5f}'
    )
    opt_id = f'{seg_id}_{hashlib.md5(signature.encode()).hexdigest()[:8]}'
    names = {'walk': 'walk', 'drive': 'drive'}
    return SegmentOption(
        option_id=opt_id,
        label=names.get(leg.mode, leg.mode.title()),
        mode_chain=leg.mode,
        micro_legs=[MicroLeg.from_detailed(leg)],
        duration_min=max(leg.duration_min, 1.0),
        distance_km=leg.distance_km,
        verified=leg.verified,
        is_recommended=True,
        tooltip=f'{leg.duration_min:.0f} min · {leg.distance_km:.1f} km',
    )


def clean_route_cue(cue: str) -> str:
    text = (cue or '').strip()
    m = re.match(
        r'^(depart|arrive|turn|continue|new name|fork|roundabout|exit roundabout) on (.+)$',
        text,
        flags=re.I,
    )
    if m and m.group(2).strip().lower() in GENERIC_CUE_NAMES:
        return m.group(1).lower()
    return text


def clean_leg_steps(leg: DetailedLeg) -> None:
    for step in leg.steps:
        step.instruction = clean_route_cue(step.instruction)


def build_local_plan(
    builder: ModularTripBuilder,
    origin: POI,
    dest: POI,
    *,
    mode: str = 'drive',
    allow_fallback: bool = False,
) -> ModularTripPlan:
    a = waypoint_from_poi(origin)
    b = waypoint_from_poi(dest)
    title = f'{origin.query} → {dest.query}'
    leg = builder.env.osrm.route(
        a.lat, a.lon, a.name,
        b.lat, b.lon, b.name,
        mode=mode,
    )
    if leg is None:
        if not allow_fallback:
            raise ValueError(f'no OSRM {mode} route')
        leg = fallback_leg(origin, dest, mode)
    clean_leg_steps(leg)
    option = option_from_leg('seg_1', leg)
    segment = TripSegment(
        segment_id='seg_1',
        title=title,
        segment_type='local',
        from_name=a.name,
        to_name=b.name,
        options=[option],
        default_option_id=option.option_id,
        description=f'{mode} route from OSRM/OpenStreetMap',
    )
    plan = ModularTripPlan(
        origin=origin.query,
        destination=dest.query,
        segments=[segment],
        instruction=f'Annotate route from {origin.query} to {dest.query}',
    )
    return plan


def choose_ground_option(plan: ModularTripPlan) -> dict[str, str]:
    selections = {}
    for seg in plan.segments:
        pure_drive = [
            opt for opt in seg.options
            if opt.micro_legs and all(m.mode == 'drive' for m in opt.micro_legs)
        ]
        pure_walk = [
            opt for opt in seg.options
            if opt.micro_legs and all(m.mode == 'walk' for m in opt.micro_legs)
        ]
        any_ground = [
            opt for opt in seg.options
            if opt.micro_legs and any(m.mode in GROUND_MODES for m in opt.micro_legs)
        ]
        chosen = (pure_drive or pure_walk or any_ground or seg.options)[0]
        selections[seg.segment_id] = chosen.option_id
    return selections


def route_distance_m(nodes: list[dict], start_idx: int, end_idx: int) -> int:
    total = 0.0
    for i in range(start_idx, end_idx):
        a, b = nodes[i], nodes[i + 1]
        total += haversine_km(a['lat'], a['lon'], b['lat'], b['lon']) * 1000
    return int(round(total))


def road_from_cue(cue: str) -> str:
    cue = clean_route_cue(cue)
    patterns = [
        r'(?:turn|continue|depart|arrive|new name|fork|roundabout|exit roundabout) on (.+)$',
        r'onto (.+)$',
        r'toward (.+)$',
    ]
    for pat in patterns:
        m = re.search(pat, cue, flags=re.I)
        if m:
            road = m.group(1).strip()
            if road.lower() in GENERIC_CUE_NAMES:
                return ''
            return road[:80]
    return ''


def ordinal_for_distance(distance_m: int) -> str:
    if distance_m < 90:
        n = 1
    elif distance_m < 220:
        n = 2
    elif distance_m < 420:
        n = 3
    else:
        n = 4
    return ORDINALS[n]


def action_phrase(action: str) -> str:
    return {
        'left': 'turn left',
        'right': 'turn right',
        'u_turn': 'make a U-turn',
        'turn_around': 'turn around',
        'stop': 'stop',
    }.get(action, action.replace('_', ' '))


def instruction_for_target(nodes: list[dict], source_idx: int, target_idx: int, rng: random.Random) -> str:
    source = nodes[source_idx]
    target = nodes[target_idx]
    action = target['oracle_action']
    distance_m = route_distance_m(nodes, source_idx, target_idx)
    road = road_from_cue(target.get('route_cue', ''))
    ordinal = ordinal_for_distance(distance_m)
    phrase = action_phrase(action)

    if action == 'stop':
        dest = target.get('to') or 'the destination'
        templates = [
            f'Continue until you reach {dest}, then stop.',
            f'Keep following the route to {dest} and stop there.',
            f'Go straight toward the destination and stop when you arrive.',
        ]
        return rng.choice(templates)

    if action in ('u_turn', 'turn_around'):
        if distance_m < 120:
            return f'At the next safe point, {phrase}.'
        return f'Continue for about {distance_m} meters, then {phrase}.'

    if road:
        templates = [
            f'Continue to {road}, then {phrase}.',
            f'At the {ordinal} intersection, {phrase} onto {road}.',
            f'Go straight for about {distance_m} meters, then {phrase} on {road}.',
        ]
    else:
        place = 'traffic light' if rng.random() < 0.45 else 'intersection'
        templates = [
            f'At the {ordinal} {place}, {phrase}.',
            f'Continue for about {distance_m} meters, then {phrase}.',
            f'Go straight until the {ordinal} cross street, then {phrase}.',
        ]
    return rng.choice(templates)


def apply_auto_instructions(nodes: list[dict], seed: int) -> int:
    rng = random.Random(seed)
    for node in nodes:
        node['annotation_instruction'] = ''
        node['action_override'] = ''
        node['auto_instruction_target'] = ''

    keypoint_indices = [
        i for i, node in enumerate(nodes)
        if node.get('oracle_action') in KEYPOINT_ACTIONS
    ]
    if not keypoint_indices:
        return 0

    trigger_indices = [0]
    trigger_indices.extend(
        idx + 1 for idx in keypoint_indices
        if idx + 1 < len(nodes) and nodes[idx].get('oracle_action') != 'stop'
    )
    trigger_indices = sorted(set(trigger_indices))

    annotated = 0
    for source_idx in trigger_indices:
        future = [idx for idx in keypoint_indices if idx > source_idx]
        if not future:
            continue
        target_idx = future[0]
        nodes[source_idx]['annotation_instruction'] = instruction_for_target(
            nodes, source_idx, target_idx, rng,
        )
        nodes[source_idx]['auto_instruction_target'] = nodes[target_idx].get('node_id', '')
        nodes[source_idx]['auto_target_action'] = nodes[target_idx].get('oracle_action', '')
        annotated += 1
    return annotated


def build_record(
    idx: int,
    city,
    origin: POI,
    dest: POI,
    plan: ModularTripPlan,
    selections: dict[str, str],
    nodes: list[dict],
    annotated_count: int,
    sample_every: int,
) -> dict:
    now = int(time.time())
    return {
        'id': f'instr_auto_{idx:05d}_{city.city_id}',
        'schema_version': 'instruction_nodes_v1_auto',
        'annotator': 'auto',
        'created_at': now,
        'city': city.city,
        'country': city.country,
        'origin': plan.origin,
        'destination': plan.destination,
        'origin_input': origin.query,
        'destination_input': dest.query,
        'route_instruction': plan.instruction,
        'selections': selections,
        'sample_every': sample_every,
        'ground_only': True,
        'node_count': len(nodes),
        'annotated_count': annotated_count,
        'keypoint_count': sum(1 for n in nodes if n.get('is_keypoint')),
        'auto_annotation': {
            'policy': 'place instruction at route start and immediately after each keypoint; describe the next future keypoint action',
            'empty_instruction_meaning': 'follow route/default forward',
            'origin_destination_source': 'concrete POIs from CITY_SEEDS',
        },
        'total': plan.compute_total(selections),
        'plan': plan.to_dict(),
        'route_geometry': plan.path_for_selections(selections),
        'nodes': nodes,
    }


def generate(args: argparse.Namespace) -> list[dict]:
    builder = ModularTripBuilder()
    builder.env.geocoder.use_online = not args.offline
    records = []
    failures = []

    pairs = candidate_pairs(args.seed)
    if args.shuffle_reuse:
        pairs = pairs * max(1, (args.count // max(len(pairs), 1)) + 1)

    for city, origin, dest in pairs:
        if len(records) >= args.count:
            break
        try:
            plan = build_local_plan(
                builder,
                origin,
                dest,
                mode=args.route_mode,
                allow_fallback=args.allow_fallback,
            )
            selections = choose_ground_option(plan)
            nodes = _route_nodes_for_selection(
                plan,
                selections,
                sample_every=args.sample_every,
                ground_only=True,
                max_nodes=args.max_nodes,
            )
            if len(nodes) < args.min_nodes:
                raise ValueError(f'too few nodes: {len(nodes)}')
            annotated_count = apply_auto_instructions(nodes, args.seed + len(records) + 17)
            if annotated_count < args.min_annotations:
                raise ValueError(f'too few auto instructions: {annotated_count}')
            record = build_record(
                len(records) + 1,
                city,
                origin,
                dest,
                plan,
                selections,
                nodes,
                annotated_count,
                args.sample_every,
            )
            records.append(record)
            print(
                f'[{len(records):02d}/{args.count}] {city.city}: '
                f'{origin.label} -> {dest.label} '
                f'nodes={len(nodes)} annotated={annotated_count}',
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - batch generator should skip bad pairs.
            failures.append((city.city, origin.label, dest.label, str(exc)))
            if args.verbose:
                print(f'skip {city.city}: {origin.label}->{dest.label}: {exc}', flush=True)

    if len(records) < args.count:
        raise RuntimeError(
            f'generated {len(records)} records, requested {args.count}; '
            f'failures={len(failures)}'
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--max-nodes', type=int, default=1600)
    parser.add_argument('--min-nodes', type=int, default=8)
    parser.add_argument('--min-annotations', type=int, default=1)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--route-mode', choices=['drive', 'walk'], default='drive')
    parser.add_argument('--allow-fallback', action='store_true')
    parser.add_argument('--shuffle-reuse', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--out',
        default='data/globnav_bench/instruction_annotations_auto.jsonl',
    )
    args = parser.parse_args()

    records = generate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    print(f'wrote {len(records)} records to {out}')


if __name__ == '__main__':
    main()
