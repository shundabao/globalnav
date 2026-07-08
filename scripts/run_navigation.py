#!/usr/bin/env python3
"""GLOBALNAV CLI — LLM understands intent; environment plans concrete routes."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.clarification import InstructionClarifier
from globe_nav.config import DEFAULT_MODEL, load_env
from globe_nav.planner.display import format_options
from globe_nav.planner.environment import MobilityEnvironment


def run(args):
    load_env()

    clarifier = InstructionClarifier(model=args.model)
    env = MobilityEnvironment(cache_dir=args.cache_dir, use_online=not args.offline)

    def answer_fn(question, options):
        if options:
            print(f'  Options: {", ".join(options)}')
        if args.auto_answer and options:
            return options[0]
        return input(f'  {question}\n  > ').strip()

    # 1) LLM: origin + destination only
    result = clarifier.analyze(args.instruction, args.origin, args.destination)
    while not result.is_clear:
        if args.no_clarify:
            print(json.dumps({'status': 'clarifying', 'questions': result.questions}, ensure_ascii=False))
            return 1
        amb = result.ambiguities[0]
        if args.verbose:
            print(f'[CLARIFY] {amb.question}')
        answer = answer_fn(amb.question, amb.options)
        result = clarifier.answer_and_reparse(amb.category, amb.question, answer)

    origin = result.resolved_origin
    destination = result.resolved_destination

    if args.verbose:
        print('=== LLM parsed (intent only) ===')
        print(f'  origin:      {origin}')
        print(f'  destination: {destination}')
        print()

    # 2) Environment: enumerate all feasible multi-modal options
    if args.verbose:
        print('=== Querying OSRM / OpenFlights / OSM transit ... ===\n')

    options = env.plan_all_options(origin, destination)

    # 3) Output
    text = format_options(options)
    print(text)

    payload = {
        'status': 'ok',
        'origin': origin,
        'destination': destination,
        'option_count': len(options),
        'options': [o.to_dict() for o in options],
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'Saved to {args.output}')
    elif args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


def main():
    parser = argparse.ArgumentParser(description='GLOBALNAV: environment-driven multi-modal routing')
    parser.add_argument('-i', '--instruction', required=True)
    parser.add_argument('--origin', help='Optional origin override')
    parser.add_argument('--destination', '-d', help='Optional destination override')
    parser.add_argument('--offline', action='store_true', help='Disable online geocoding')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--json', action='store_true', help='Also print full JSON')
    parser.add_argument('--auto-answer', action='store_true')
    parser.add_argument('--no-clarify', action='store_true')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
