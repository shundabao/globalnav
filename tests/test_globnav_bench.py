"""Tests for GlobNav-Bench sampling and annotation endpoints."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.bench.sampler import GlobNavBenchSampler
from globe_nav.bench.schema import read_jsonl, validate_example, validation_report
from globe_nav.gui.app import create_app


class TestGlobNavBenchSampler(unittest.TestCase):
    def test_sampler_generates_valid_examples(self):
        examples = GlobNavBenchSampler(seed=11).sample(40)
        self.assertEqual(len(examples), 40)
        self.assertFalse([err for ex in examples for err in validate_example(ex)])
        splits = {ex['split'] for ex in examples}
        self.assertIn('planning_feasibility', splits)
        self.assertIn('intent_clarification', splits)
        self.assertIn('hybrid_follower', splits)

    def test_pilot_500_exists_and_validates(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'globnav_bench', 'pilot_500.jsonl',
        )
        examples = read_jsonl(path)
        report = validation_report(examples)
        self.assertEqual(report['example_count'], 500)
        self.assertEqual(report['failure_count'], 0)
        self.assertEqual(report['by_split']['planning_feasibility'], 150)
        self.assertEqual(report['by_split']['intent_clarification'], 200)
        self.assertEqual(report['by_split']['route_option_generation'], 50)
        self.assertEqual(report['by_split']['hybrid_follower'], 75)


class TestGlobNavBenchApi(unittest.TestCase):
    def test_bench_examples_endpoint(self):
        app = create_app(offline=True)
        client = app.test_client()
        response = client.get('/api/bench/examples?limit=2&split=planning_feasibility')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreaterEqual(payload['total'], 2)
        self.assertEqual(len(payload['examples']), 2)
        self.assertEqual(payload['examples'][0]['split'], 'planning_feasibility')

    def test_bench_review_requires_example_id(self):
        app = create_app(offline=True)
        client = app.test_client()
        response = client.post('/api/bench/review', json={'review_status': 'approved'})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
