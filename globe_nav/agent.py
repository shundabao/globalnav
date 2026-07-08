"""Global navigation agent — LLM-driven instruction parsing and action selection."""

from dataclasses import dataclass, field
from typing import Callable, Optional

from globe_nav.clarification import ClarificationResult, InstructionClarifier
from globe_nav.config import DEFAULT_MODEL
from globe_nav.env.global_env import GlobalNavEnv
from globe_nav.llm_client import LLMClient


@dataclass
class NavigationSession:
    instruction: str
    clarification: ClarificationResult
    trajectory: list[dict] = field(default_factory=list)
    success: bool = False
    status: str = 'pending'


class GlobalNavAgent:
    """
    LLM-first agent:
    1. LLM parses instruction → multi-modal leg plan
    2. LLM asks clarifying questions when ambiguous
    3. LLM selects navigation actions step-by-step
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        use_online_maps: bool = True,
        cache_dir: str = 'data/cache',
        max_steps: int = 200,
        query_func: Optional[Callable[[str], str]] = None,
    ):
        self.model = model
        self.env = GlobalNavEnv(use_online_maps=use_online_maps, cache_dir=cache_dir)
        self.clarifier = InstructionClarifier(model=model)
        self.llm = LLMClient(model=model, max_tokens=64)
        self.query_func = query_func or self._llm_action_policy
        self.max_steps = max_steps
        self.session: Optional[NavigationSession] = None

    def start(
        self,
        instruction: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        modes: Optional[list[str]] = None,
    ) -> ClarificationResult:
        result = self.clarifier.analyze(instruction, origin, destination, modes)
        self.session = NavigationSession(instruction=instruction, clarification=result)
        self.session.status = 'navigating' if result.is_clear else 'clarifying'
        return result

    def answer_clarification(self, category: str, answer: str) -> ClarificationResult:
        if not self.session:
            raise RuntimeError('No active session. Call start() first.')
        amb = next((a for a in self.session.clarification.ambiguities if a.category == category), None)
        question = amb.question if amb else category
        result = self.clarifier.answer_and_reparse(category, question, answer)
        self.session.clarification = result
        self.session.status = 'navigating' if result.is_clear else 'clarifying'
        return result

    def navigate(self, verbatim: bool = False) -> dict:
        if not self.session:
            raise RuntimeError('No active session. Call start() first.')

        clar = self.session.clarification
        if not clar.is_clear:
            return {
                'status': 'clarifying',
                'questions': clar.questions,
                'ambiguities': [
                    {'category': a.category, 'question': a.question, 'options': a.options}
                    for a in clar.ambiguities
                ],
            }

        task = {
            'instruction': self.session.instruction,
            'origin': clar.resolved_origin,
            'destination': clar.resolved_destination,
            'resolved_origin': clar.resolved_origin,
            'resolved_destination': clar.resolved_destination,
            'modes': clar.resolved_modes,
            'legs': clar.legs,
        }

        try:
            self.env.reset(task)
        except ValueError as e:
            self.session.status = 'failed'
            return {'status': 'failed', 'error': str(e)}

        if verbatim:
            print('=== Planned legs ===')
            for i, leg in enumerate(clar.legs, 1):
                print(f'  {i}. [{leg["mode"]}] {leg["from"]} → {leg["to"]}')
                if leg.get('description'):
                    print(f'     {leg["description"]}')
            print()

        step = 0
        while step < self.max_steps:
            obs = self.env.get_observation()
            prompt = self._build_prompt(obs)
            action = self.query_func(prompt)
            action = self._extract_action(action, obs['available_actions'])

            if verbatim:
                print(f'Step {step}: {action} | [{obs["mode"]}] {obs["instruction"]}')

            obs, done, info = self.env.step(action)
            self.session.trajectory.append({
                'step': step, 'action': action, 'observation': obs, 'done': done, 'info': info,
            })

            if done:
                self.session.success = info.get('success', False)
                self.session.status = 'done'
                return self._build_result(info)

            step += 1

        self.session.status = 'failed'
        return {'status': 'failed', 'error': 'max_steps exceeded', 'trajectory': self.session.trajectory}

    def run_interactive(
        self,
        instruction: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        modes: Optional[list[str]] = None,
        answer_fn: Optional[Callable[[str, list[str]], str]] = None,
        verbatim: bool = False,
    ) -> dict:
        result = self.start(instruction, origin, destination, modes)

        if verbatim:
            print('=== LLM parsed trip ===')
            print(f'  origin:      {result.resolved_origin}')
            print(f'  destination: {result.resolved_destination}')
            for i, leg in enumerate(result.legs, 1):
                print(f'  leg {i}: [{leg["mode"]}] {leg["from"]} → {leg["to"]}')
            if result.questions:
                print(f'  needs clarification: {result.questions}')
            print()

        while not result.is_clear:
            if verbatim:
                for amb in result.ambiguities:
                    print(f'[CLARIFY] {amb.question}')
                    if amb.options:
                        print(f'  Options: {amb.options}')

            amb = result.ambiguities[0]
            if answer_fn:
                answer = answer_fn(amb.question, amb.options)
            elif amb.options:
                answer = amb.options[0]
            else:
                raise RuntimeError(f'Instruction unclear: {amb.question}. Provide answer_fn.')

            result = self.answer_clarification(amb.category, answer)
            if verbatim:
                print(f'[ANSWER] {answer}')

        return self.navigate(verbatim=verbatim)

    def _llm_action_policy(self, prompt: str) -> str:
        return self.llm.chat_text(
            system='You are a navigation agent. Reply with exactly one action from the Available actions list.',
            user=prompt,
            max_tokens=16,
        )

    def _build_prompt(self, obs: dict) -> str:
        actions = ', '.join(obs['available_actions'])
        legs_summary = '; '.join(
            f'{l["mode"]}: {l["from"]}→{l["to"]}' for l in obs.get('leg_plan', [])
        )
        return (
            f'You are a global navigation agent (current mode: {obs["mode"]}).\n'
            f'Task: {obs["task_instruction"]}\n'
            f'Full trip legs: {legs_summary}\n'
            f'Current segment: {obs["segment"]}\n'
            f'Location: {obs["location"]} ({obs["lat"]:.4f}, {obs["lon"]:.4f})\n'
            f'Step instruction: {obs["instruction"]}\n'
            f'Progress: {obs["progress"]}\n'
            f'Total trip: {obs["total_distance_km"]:.0f} km, ETA {obs["total_eta"]}\n'
            f'Available actions: {actions}\n'
            f'Reply with one action only.'
        )

    def _extract_action(self, output: str, valid_actions: list[str]) -> str:
        output = output.lower().strip()
        for action in valid_actions:
            if action in output.split():
                return action
        for action in valid_actions:
            if action in output:
                return action
        # fallback: follow route
        for preferred in ('forward', 'cruise', 'takeoff', 'board', 'depart', 'land', 'arrive', 'dock'):
            if preferred in valid_actions:
                return preferred
        return valid_actions[0] if valid_actions else 'stop'

    def _build_result(self, info: dict) -> dict:
        clar = self.session.clarification
        return {
            'status': 'done',
            'success': info.get('success', False),
            'origin': clar.resolved_origin,
            'destination': clar.resolved_destination,
            'modes': clar.resolved_modes,
            'legs': clar.legs,
            'total_distance_km': sum(s.distance_km for s in self.env.segments),
            'segments': [
                {
                    'mode': s.mode.value,
                    'from': s.from_waypoint.name,
                    'to': s.to_waypoint.name,
                    'distance_km': round(s.distance_km, 1),
                    'eta': s.eta_display,
                    'description': s.description,
                    'instructions': s.instructions,
                }
                for s in self.env.segments
            ],
            'steps': len(self.session.trajectory),
            'trajectory': self.session.trajectory,
        }
