#!/usr/bin/env python3
"""Run VELMA-style global instruction follower."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.config import DEFAULT_MODEL, load_env
from globe_nav.follower.agent import InstructionFollowerAgent


def main():
    parser = argparse.ArgumentParser(description='GLOBALNAV Instruction Follower (VELMA-style)')
    parser.add_argument('-i', '--instruction', required=True, help='Full navigation instruction')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--rule-based', action='store_true', help='No LLM for actions (demo)')
    parser.add_argument('--origin', help='Explicit origin (skips LLM clarify when used with --destination)')
    parser.add_argument('--destination', help='Explicit destination')
    parser.add_argument('--no-llm', action='store_true', help='Skip all LLM calls (offline decompose + rule actions)')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-o', '--output', help='Save trajectory JSON')
    args = parser.parse_args()

    load_env()
    use_llm = not args.no_llm
    agent = InstructionFollowerAgent(
        model=args.model,
        use_online_maps=not args.offline,
        rule_based=args.rule_based or args.no_llm,
        streetview=not args.offline,
    )

    print('=== Preparing trip (parse + environment route) ===')
    prep = agent.prepare(
        args.instruction,
        origin=args.origin,
        destination=args.destination,
        use_llm=use_llm,
    )
    if prep.get('status') == 'clarifying':
        print(json.dumps(prep, indent=2, ensure_ascii=False))
        return 1

    print(f"Origin: {prep['origin']}")
    print(f"Destination: {prep['destination']}")
    print(f"Execution legs: {prep['execution_legs']}")
    print(json.dumps(prep['decomposed'], indent=2, ensure_ascii=False))
    print('\n=== Following instruction step-by-step ===\n')

    result = agent.run(max_steps=args.max_steps, verbatim=args.verbose)

    print(f"\n=== Done: success={result.success}, steps={result.steps} ===")
    payload = {
        'success': result.success,
        'steps': result.steps,
        'decomposed': result.decomposed,
        'trajectory': result.trajectory,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f'Saved to {args.output}')
    elif args.verbose:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str)[:8000])

    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
