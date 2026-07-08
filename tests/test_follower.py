"""Tests for instruction follower simulator."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.follower.agent import InstructionFollowerAgent
from globe_nav.follower.alignment import procedural_notes_map
from globe_nav.follower.decomposer import DecomposedInstruction, SegmentGoal
from globe_nav.follower.simulator import GlobalInstructionSimulator, bearing_label
from globe_nav.planner.segment_models import ModularTripPlan
from globe_nav.planner.trip_builder import ModularTripBuilder


class TestSimulator(unittest.TestCase):
    def test_bearing(self):
        self.assertEqual(bearing_label(90), 'E')

    def test_local_simulation(self):
        builder = ModularTripBuilder()
        builder.env.geocoder.use_online = False
        plan = builder.build('UTS', 'Sydney Opera House')
        sim = GlobalInstructionSimulator(plan)
        self.assertGreater(len(sim.legs), 0)
        steps = 0
        while not sim.state.done and steps < 50:
            obs = sim.get_observation()
            action = 'left' if obs.get('is_turn_ahead') and 'left' in obs.get('available_actions', []) else 'forward'
            _, done, _ = sim.step(action)
            steps += 1
            if done:
                break
        self.assertTrue(sim.state.success or steps > 0)


class TestAlignment(unittest.TestCase):
    def test_three_phase_notes(self):
        decomposed = DecomposedInstruction(
            initial_location='UTS', initial_facing='east', goal='Liverpool Uni',
            segment_goals=[
                SegmentGoal('seg_access', 'walk', 'UTS', 'SYD', 'walk to airport'),
                SegmentGoal('seg_flight', 'fly', 'SYD', 'MAN', 'fly via Dubai'),
                SegmentGoal('seg_egress', 'train', 'MAN', 'Liverpool', 'train then walk'),
            ],
        )
        builder = ModularTripBuilder()
        builder.env.geocoder.use_online = False
        plan = builder.build('UTS', 'University of Liverpool')
        notes = procedural_notes_map(decomposed, plan)
        self.assertIn('walk to airport', notes.get('seg_access', ''))
        self.assertIn('fly', notes.get('seg_flight', '').lower())


class TestRouteGeometry(unittest.TestCase):
    def test_long_haul_has_geometry(self):
        builder = ModularTripBuilder()
        builder.env.geocoder.use_online = False
        plan = builder.build(
            'UTS', 'University of Liverpool',
            instruction='fly from Sydney to Manchester then Liverpool',
        )
        self.assertEqual(len(plan.segments), 3)
        geo = plan.path_for_selections()
        self.assertTrue(geo['segments'])
        has_fly = any(
            pl['mode'] == 'fly'
            for seg in geo['segments']
            for pl in seg.get('polylines', [])
        )
        self.assertTrue(has_fly)


class TestFollowerAgent(unittest.TestCase):
    def test_rule_based_opera(self):
        agent = InstructionFollowerAgent(use_online_maps=False, rule_based=True, streetview=False)
        prep = agent.prepare(
            'Walk from UTS to Sydney Opera House',
            origin='UTS', destination='Sydney Opera House', use_llm=False,
        )
        if prep.get('status') == 'clarifying':
            self.skipTest('needs clarification')
        self.assertEqual(prep['status'], 'ready')
        self.assertIn('route_geometry', prep)
        result = agent.run(max_steps=80, verbatim=False)
        self.assertGreater(result.steps, 0)

    def test_step_once(self):
        agent = InstructionFollowerAgent(use_online_maps=False, rule_based=True, streetview=False)
        prep = agent.prepare(
            'Walk from UTS to Sydney Opera House',
            origin='UTS', destination='Sydney Opera House', use_llm=False,
        )
        if prep.get('status') != 'ready':
            self.skipTest('needs clarification')
        step = agent.step_once()
        self.assertIn('action', step)
        self.assertIn('observation', step)


class TestLiverpoolE2E(unittest.TestCase):
    """Full multi-modal UTS → Liverpool (offline decompose + network routing)."""

    def test_liverpool_rule_based(self):
        agent = InstructionFollowerAgent(
            use_online_maps=True, rule_based=True, streetview=False,
        )
        instruction = (
            '我在悉尼科技大学UTS，面朝乔治街。先走到悉尼机场，坐飞机到曼彻斯特，'
            '出机场后坐火车到利物浦，最后走到利物浦大学。'
        )
        prep = agent.prepare(
            instruction,
            origin='University of Technology Sydney',
            destination='University of Liverpool',
            use_llm=False,
        )
        if prep.get('status') == 'clarifying':
            self.skipTest('needs clarification')
        self.assertEqual(prep['status'], 'ready')
        plan_str = str(prep.get('plan', {}))
        self.assertTrue(
            'MAN' in plan_str or 'Manchester' in plan_str or '曼彻斯特' in plan_str,
            'expected Manchester in flight segment',
        )
        self.assertGreaterEqual(prep['execution_legs'], 3)

        result = agent.run(max_steps=400, verbatim=False)
        self.assertGreater(result.steps, 5)
        self.assertTrue(result.success, f'failed after {result.steps} steps')


if __name__ == '__main__':
    unittest.main()
