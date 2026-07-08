"""Tests for planner and LLM clarification."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.clarification import InstructionClarifier
from globe_nav.llm_parser import LLMInstructionParser, ParsedTrip
from globe_nav.maps.geocoder import Geocoder
from globe_nav.planner.flights import FlightGraph
from globe_nav.planner.models import DetailedLeg, RouteOption

MOCK_INTENT = {
    'origin': 'University of Technology Sydney',
    'destination': 'Jiuzhaigou, Sichuan, China',
    'needs_clarification': False,
    'questions': [],
}


class TestGeocoder(unittest.TestCase):
    def test_uts(self):
        wp = Geocoder(use_online=False).geocode('UTS')
        self.assertIsNotNone(wp)


class TestFlightGraph(unittest.TestCase):
    def test_no_direct_syd_ctu(self):
        fg = FlightGraph()
        fg.ensure_loaded()
        self.assertFalse(fg.has_direct('SYD', 'CTU'))

    def test_one_stop_syd_ctu(self):
        fg = FlightGraph()
        paths = fg.find_paths('SYD', 'CTU', max_stops=1)
        self.assertTrue(len(paths) > 0)
        self.assertEqual(len(paths[0]), 3)


class TestLLMIntentOnly(unittest.TestCase):
    @patch('globe_nav.llm_parser.LLMClient')
    def test_no_legs_from_llm(self, mock_client_cls):
        mock_client_cls.return_value.chat_json.return_value = MOCK_INTENT
        clarifier = InstructionClarifier()
        result = clarifier.analyze('我在UTS，要去四川九寨沟')
        self.assertTrue(result.is_clear)
        self.assertEqual(len(result.legs), 0)
        self.assertIn('Technology Sydney', result.resolved_origin)


class TestRouteOption(unittest.TestCase):
    def test_total_time(self):
        opt = RouteOption('o1', legs=[
            DetailedLeg('walk', 'A', 'B', 0, 0, 0, 0, 1, 10),
            DetailedLeg('fly', 'B', 'C', 0, 0, 0, 0, 100, 60),
        ])
        self.assertEqual(opt.total_duration_min, 70)


@unittest.skipUnless(os.environ.get('OPENAI_API_KEY'), 'needs API key')
class TestLivePlanner(unittest.TestCase):
    def setUp(self):
        from globe_nav.config import load_env
        load_env()

    def test_uts_jiuzhaigou_options(self):
        from globe_nav.planner.environment import MobilityEnvironment
        env = MobilityEnvironment(use_online=False)
        opts = env.plan_all_options('UTS', 'Jiuzhaigou')
        self.assertGreater(len(opts), 1)
        self.assertTrue(any('fly' in o.mode_chain for o in opts))
        has_osrm = any(
            any(l.mode in ('walk', 'drive') and l.steps for l in o.legs)
            for o in opts
        )
        self.assertTrue(has_osrm)


if __name__ == '__main__':
    unittest.main()
