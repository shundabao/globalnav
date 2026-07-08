"""Decompose long procedural navigation instructions into structured segment goals."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.config import DEFAULT_MODEL
from globe_nav.llm_client import LLMClient


def _extract_phase_notes(text: str, keywords: tuple[str, ...]) -> str:
    """Keep sentences that mention any keyword."""
    import re
    parts = re.split(r'[。.!?\n;；]', text)
    hits = [p.strip() for p in parts if p.strip() and any(k in p.lower() or k in p for k in keywords)]
    return ' '.join(hits[:3])


def _guess_origin(instruction: str) -> str:
    lower = instruction.lower()
    if 'uts' in lower or '悉尼科技' in instruction:
        return 'University of Technology Sydney'
    if 'times square' in lower or '时代广场' in instruction:
        return 'Times Square, New York'
    return ''


def _guess_destination(instruction: str) -> str:
    lower = instruction.lower()
    if '利物浦大学' in instruction or 'liverpool university' in lower:
        return 'University of Liverpool'
    if '歌剧院' in instruction or 'opera house' in lower:
        return 'Sydney Opera House'
    if '九寨沟' in instruction:
        return 'Jiuzhaigou National Park'
    return ''

SYSTEM = """You decompose a long navigation instruction into ordered segment goals for simulation.

The environment will handle routing; you extract what the USER wants at each phase.

Return JSON:
{
  "initial_pose": {
    "location": "place name",
    "facing": "compass direction or landmark (e.g. east toward Broadway)"
  },
  "goal": "final destination",
  "segment_goals": [
    {
      "segment_id": "seg_1",
      "mode_hint": "walk|drive|bus|train|fly",
      "from_hint": "start of this phase",
      "to_hint": "end of this phase",
      "procedural_notes": "turn left at X, face Y, etc."
    }
  ],
  "summary": "one line trip summary"
}

Rules:
- Preserve turn-by-turn cues (left, right, facing) in procedural_notes
- Split at mode changes: walk to airport, fly to city, train to campus, etc.
- For intercontinental trips use exactly 3 logical phases matching the planner:
  seg_access (local to departure airport), seg_flight (air segment), seg_egress (arrival to final goal)
- Use segment_id values: seg_access, seg_flight, seg_egress for long trips
"""


@dataclass
class SegmentGoal:
    segment_id: str
    mode_hint: str
    from_hint: str
    to_hint: str
    procedural_notes: str = ''


@dataclass
class DecomposedInstruction:
    initial_location: str
    initial_facing: str
    goal: str
    segment_goals: list[SegmentGoal] = field(default_factory=list)
    summary: str = ''
    raw_instruction: str = ''

    def to_dict(self) -> dict:
        return {
            'initial_pose': {'location': self.initial_location, 'facing': self.initial_facing},
            'goal': self.goal,
            'summary': self.summary,
            'segment_goals': [
                {
                    'segment_id': g.segment_id,
                    'mode_hint': g.mode_hint,
                    'from_hint': g.from_hint,
                    'to_hint': g.to_hint,
                    'procedural_notes': g.procedural_notes,
                }
                for g in self.segment_goals
            ],
        }


class InstructionDecomposer:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = LLMClient(model=model, max_tokens=1200)

    def decompose_offline(
        self, instruction: str, origin: str = '', destination: str = '',
    ) -> DecomposedInstruction:
        """Rule-based 3-phase split when LLM unavailable."""
        text = instruction
        origin = origin or _guess_origin(text)
        destination = destination or _guess_destination(text)
        lower = text.lower()
        facing = 'unknown'
        for token, label in (
            ('乔治街', 'east along George Street'),
            ('george street', 'east along George Street'),
            ('broadway', 'toward Broadway'),
            ('面朝北', 'north'), ('面朝南', 'south'),
            ('面朝东', 'east'), ('面朝西', 'west'),
        ):
            if token in lower or token in text:
                facing = label
                break

        is_long = any(k in lower or k in text for k in (
            'fly', 'flight', '飞机', '坐飞', '机场', 'airport', 'train', '火车',
        ))
        goals: list[SegmentGoal] = []
        if is_long:
            access_note = _extract_phase_notes(text, ('walk', '走', '步行', 'taxi', 'central', '中央'))
            flight_note = _extract_phase_notes(text, ('fly', 'flight', '飞机', '曼彻斯特', 'manchester', 'dubai', '迪拜'))
            egress_note = _extract_phase_notes(text, ('train', '火车', 'liverpool', '利物浦', '大学'))
            goals = [
                SegmentGoal('seg_access', 'walk', origin, 'departure airport', access_note or text[:120]),
                SegmentGoal('seg_flight', 'fly', 'departure airport', 'arrival airport', flight_note),
                SegmentGoal('seg_egress', 'train', 'arrival airport', destination, egress_note),
            ]
        else:
            goals = [
                SegmentGoal('seg_1', 'walk', origin, destination, text[:200]),
            ]

        return DecomposedInstruction(
            initial_location=origin,
            initial_facing=facing,
            goal=destination,
            segment_goals=goals,
            summary=f'{origin} → {destination}',
            raw_instruction=instruction,
        )

    def decompose(self, instruction: str, origin: str = '', destination: str = '') -> DecomposedInstruction:
        user = f'Instruction:\n{instruction}'
        if origin:
            user += f'\nKnown origin: {origin}'
        if destination:
            user += f'\nKnown destination: {destination}'

        try:
            raw = self.llm.chat_json(SYSTEM, user)
        except Exception:
            return self.decompose_offline(instruction, origin, destination)
        goals = []
        for i, g in enumerate(raw.get('segment_goals', [])):
            goals.append(SegmentGoal(
                segment_id=g.get('segment_id', f'seg_{i + 1}'),
                mode_hint=g.get('mode_hint', 'walk'),
                from_hint=g.get('from_hint', ''),
                to_hint=g.get('to_hint', ''),
                procedural_notes=g.get('procedural_notes', ''),
            ))

        pose = raw.get('initial_pose', {})
        return DecomposedInstruction(
            initial_location=pose.get('location', origin),
            initial_facing=pose.get('facing', 'unknown'),
            goal=raw.get('goal', destination),
            segment_goals=goals,
            summary=raw.get('summary', ''),
            raw_instruction=instruction,
        )
