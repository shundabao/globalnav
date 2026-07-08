#!/usr/bin/env python3
"""Generate a GlobNav-Bench pilot JSONL split."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.bench.sampler import GlobNavBenchSampler, summarize_examples
from globe_nav.bench.schema import JSON_SCHEMA, validation_report, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate GlobNav-Bench pilot examples.')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--out',
        default='data/globnav_bench/pilot_500.jsonl',
        help='Output JSONL path.',
    )
    parser.add_argument(
        '--schema-out',
        default='data/globnav_bench/schema.json',
        help='Output JSON schema path.',
    )
    parser.add_argument(
        '--report-out',
        default='data/globnav_bench/pilot_500_validation_report.json',
        help='Output validation report path.',
    )
    args = parser.parse_args()

    sampler = GlobNavBenchSampler(seed=args.seed)
    examples = sampler.sample(args.count)
    write_jsonl(args.out, examples)

    schema_path = Path(args.schema_out)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(JSON_SCHEMA, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    report = validation_report(examples)
    report['summary'] = summarize_examples(examples)
    report['generation'] = {
        'seed': args.seed,
        'count': args.count,
        'note': (
            'Pilot examples are generated from curated global seed places and '
            'machine-validated against schema/OpenFlights where applicable. '
            'They are ready for human annotation review in the benchmark UI.'
        ),
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f'wrote {len(examples)} examples to {args.out}')
    print(f'wrote schema to {args.schema_out}')
    print(f'wrote report to {args.report_out}')
    print(json.dumps(report['summary'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
