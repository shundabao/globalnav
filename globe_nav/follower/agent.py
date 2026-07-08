"""VELMA-style instruction follower for global multi-modal navigation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from globe_nav.clarification import InstructionClarifier
from globe_nav.config import DEFAULT_MODEL
from globe_nav.follower.decomposer import DecomposedInstruction, InstructionDecomposer
from globe_nav.follower.simulator import GlobalInstructionSimulator
from globe_nav.follower.verbalizer import (
    action_space_for_mode,
    build_init_prompt,
    observations_to_str,
)
from globe_nav.llm_client import LLMClient
from globe_nav.maps.streetview import GlobalStreetViewProvider
from globe_nav.planner.segment_models import ModularTripPlan
from globe_nav.planner.trip_builder import ModularTripBuilder


@dataclass
class FollowerResult:
    success: bool
    steps: int
    trajectory: list[dict] = field(default_factory=list)
    decomposed: Optional[dict] = None


class InstructionFollowerAgent:
    """Global instruction follower inspired by VELMA."""

    def __init__(
        self, model: str = DEFAULT_MODEL, use_online_maps: bool = True,
        rule_based: bool = False, streetview: bool = True,
    ):
        self.model = model
        self.clarifier = InstructionClarifier(model=model)
        self.decomposer = InstructionDecomposer(model=model)
        self.builder = ModularTripBuilder()
        self.builder.env.geocoder.use_online = use_online_maps
        self.llm = LLMClient(model=model, max_tokens=32)
        self.rule_based = rule_based
        self.streetview_provider = GlobalStreetViewProvider(enabled=streetview)
        self.sim: Optional[GlobalInstructionSimulator] = None
        self.plan: Optional[ModularTripPlan] = None
        self.decomposed: Optional[DecomposedInstruction] = None
        self.init_prompt: str = ''
        self.action_lines: list[str] = []
        self.selections: dict[str, str] = {}
        self.instruction: str = ''

    def prepare(
        self, instruction: str, selections: Optional[dict] = None,
        origin: Optional[str] = None, destination: Optional[str] = None,
        use_llm: bool = True,
    ) -> dict:
        self.instruction = instruction
        self.selections = dict(selections or {})

        resolved_origin = origin
        resolved_destination = destination

        if use_llm and not (origin and destination):
            clar = self.clarifier.analyze(instruction, origin=origin, destination=destination)
            if not clar.is_clear:
                return {
                    'status': 'clarifying',
                    'questions': [
                        {'category': a.category, 'question': a.question, 'options': a.options}
                        for a in clar.ambiguities
                    ],
                }
            resolved_origin = clar.resolved_origin or origin
            resolved_destination = clar.resolved_destination or destination
        else:
            offline = self.decomposer.decompose_offline(instruction, origin or '', destination or '')
            resolved_origin = origin or offline.initial_location or 'origin'
            resolved_destination = destination or offline.goal or 'destination'

        if use_llm:
            self.decomposed = self.decomposer.decompose(
                instruction, resolved_origin, resolved_destination,
            )
        else:
            self.decomposed = self.decomposer.decompose_offline(
                instruction, resolved_origin, resolved_destination,
            )

        self.plan = self.builder.build(
            resolved_origin, resolved_destination, instruction,
        )
        if not self.selections:
            for seg in self.plan.segments:
                self.selections[seg.segment_id] = seg.default_option_id

        self.sim = GlobalInstructionSimulator(
            self.plan, self.decomposed, self.selections, self.streetview_provider,
        )
        self.init_prompt = build_init_prompt(instruction, self.decomposed, self.plan)
        self.action_lines = []

        init_obs = self.sim.get_observation()
        return {
            'status': 'ready',
            'origin': self.plan.origin,
            'destination': self.plan.destination,
            'decomposed': self.decomposed.to_dict(),
            'plan': self.plan.to_dict(),
            'selections': self.selections,
            'route_geometry': self.plan.path_for_selections(self.selections),
            'execution_legs': len(self.sim.legs),
            'streetview_status': self.streetview_provider.status_dict(),
            'initial_observation': init_obs,
        }

    def update_selections(self, selections: dict[str, str]) -> dict:
        """Rebuild simulator after user picks segment options."""
        if not self.plan or not self.decomposed:
            raise RuntimeError('Call prepare() first')
        self.selections.update(selections)
        self.sim = GlobalInstructionSimulator(
            self.plan, self.decomposed, self.selections, self.streetview_provider,
        )
        self.action_lines = []
        return {
            'route_geometry': self.plan.path_for_selections(self.selections),
            'execution_legs': len(self.sim.legs),
            'initial_observation': self.sim.get_observation(),
        }

    def step_once(self) -> dict:
        """Execute one action step; for interactive GUI playback."""
        if not self.sim:
            raise RuntimeError('Call prepare() first')
        if self.sim.state.done:
            return {
                'done': True,
                'success': self.sim.state.success,
                'observation': self.sim.get_observation(),
            }

        obs_before = dict(self.sim.get_observation())
        prompt = self._build_prompt()
        action = self._query_action(prompt, obs_before)
        obs_after, done, info = self.sim.step(action)

        n = len([x for x in self.action_lines if x and x[0].isdigit()]) + 1
        self.action_lines.append(f'{n}. {action}')
        if not done:
            self.action_lines.append(observations_to_str(obs_after))

        return {
            'step': n - 1,
            'action': action,
            'observation': obs_before,
            'observation_after': obs_after if not done else obs_before,
            'done': done,
            'success': self.sim.state.success if done else False,
            'info': info,
        }

    def _build_prompt(self) -> str:
        obs = self.sim.get_observation()
        mode = obs.get('mode', 'walk')
        base = self.init_prompt
        for old, new in [
            (action_space_for_mode('walk'), action_space_for_mode(mode)),
        ]:
            if old in base:
                base = base.replace(old, new, 1)
                break
        lines = [base]
        lines.extend(self.action_lines)
        if not self.sim.state.done:
            lines.append(observations_to_str(obs))
            n = len([l for l in self.action_lines if l and l[0].isdigit()]) + 1
            lines.append(f'{n}.')
        return '\n'.join(lines)

    def run(self, max_steps: int = 500, verbatim: bool = False) -> FollowerResult:
        if not self.sim:
            raise RuntimeError('Call prepare() first')

        trajectory = []
        step = 0

        while step < max_steps and not self.sim.state.done:
            result = self.step_once()
            obs_before = result['observation']

            if verbatim:
                print(f'Step {step}: {result["action"]}')
                print(f'  {observations_to_str(obs_before)}')

            trajectory.append({
                'step': step, 'action': result['action'],
                'observation': obs_before,
                'done': result['done'], 'info': result.get('info', {}),
            })

            if result['done']:
                break
            step += 1

        return FollowerResult(
            success=self.sim.state.success,
            steps=len(trajectory),
            trajectory=trajectory,
            decomposed=self.decomposed.to_dict() if self.decomposed else None,
        )

    def _query_action(self, prompt: str, obs: dict) -> str:
        if self.rule_based:
            return self._rule_policy(obs)
        system = (
            'You are a global navigation agent. Reply with exactly ONE action from the Action Space. '
            'Use forward/cruise to progress; left/right at turns; stop only at the final destination.'
        )
        text = self.llm.chat_text(system, prompt, max_tokens=16)
        return self._extract_action(text, obs.get('available_actions', ['forward']))

    def _rule_policy(self, obs: dict) -> str:
        actions = obs.get('available_actions', ['forward'])
        if obs.get('is_turn_ahead'):
            if 'left' in actions:
                return 'left'
            if 'right' in actions:
                return 'right'
        for p in ('forward', 'cruise', 'takeoff', 'board', 'depart', 'land', 'arrive', 'dock'):
            if p in actions:
                return p
        return actions[0] if actions else 'stop'

    @staticmethod
    def _extract_action(text: str, valid: list[str]) -> str:
        text = text.lower().strip()
        for a in valid:
            if a in text.split():
                return a
        for a in valid:
            if a in text:
                return a
        for tok in re.findall(r'[a-z_]+', text):
            if tok in valid:
                return tok
        return valid[0] if valid else 'forward'
