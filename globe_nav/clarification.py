"""LLM clarifies origin/destination only; routing is environment's job."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.config import DEFAULT_MODEL
from globe_nav.llm_parser import LLMInstructionParser, ParsedTrip


@dataclass
class Ambiguity:
    category: str
    description: str
    question: str
    options: list[str] = field(default_factory=list)


@dataclass
class ClarificationResult:
    is_clear: bool
    ambiguities: list[Ambiguity] = field(default_factory=list)
    resolved_origin: Optional[str] = None
    resolved_destination: Optional[str] = None
    resolved_modes: list[str] = field(default_factory=list)
    legs: list[dict] = field(default_factory=list)
    parsed_trip: Optional[ParsedTrip] = None

    @property
    def questions(self) -> list[str]:
        return [a.question for a in self.ambiguities]


class InstructionClarifier:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.parser = LLMInstructionParser(model=model)
        self._qa_history: list[dict] = []
        self._last_instruction: str = ''

    def analyze(
        self,
        instruction: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        modes: Optional[list[str]] = None,
    ) -> ClarificationResult:
        self._last_instruction = instruction
        self._qa_history = []

        hint = instruction
        if origin or destination:
            extras = []
            if origin:
                extras.append(f'Start: {origin}')
            if destination:
                extras.append(f'Destination: {destination}')
            hint = f'{instruction}\n\n[Hints: {"; ".join(extras)}]'

        parsed = self.parser.parse(hint)
        if origin:
            parsed.origin = origin
        if destination:
            parsed.destination = destination

        return self._to_result(parsed)

    def apply_answer(self, result: ClarificationResult, category: str, answer: str) -> ClarificationResult:
        amb = next((a for a in result.ambiguities if a.category == category), None)
        question = amb.question if amb else category
        return self.answer_and_reparse(category, question, answer)

    def answer_and_reparse(self, category: str, question: str, answer: str) -> ClarificationResult:
        self._qa_history.append({'category': category, 'question': question, 'answer': answer})
        parsed = self.parser.parse(self._last_instruction, qa_history=self._qa_history)
        return self._to_result(parsed)

    def _to_result(self, parsed: ParsedTrip) -> ClarificationResult:
        instruction = self._last_instruction
        ambiguities = [
            Ambiguity(
                category=q.get('category', 'clarification'),
                description='LLM-detected ambiguity',
                question=q.get('question', ''),
                options=q.get('options', []),
            )
            for q in parsed.questions
            if not self._already_specified_in_instruction(instruction, q.get('category', ''))
        ]
        if parsed.origin and parsed.destination and not ambiguities:
            parsed.needs_clarification = False
        is_clear = (
            not parsed.needs_clarification
            and not ambiguities
            and bool(parsed.origin)
            and bool(parsed.destination)
        )
        return ClarificationResult(
            is_clear=is_clear,
            ambiguities=ambiguities,
            resolved_origin=parsed.origin,
            resolved_destination=parsed.destination,
            parsed_trip=parsed,
        )

    @staticmethod
    def _already_specified_in_instruction(instruction: str, category: str) -> bool:
        text = instruction.lower()
        cat = (category or '').lower()
        airport_kw = (
            '机场', 'airport', 'syd', 'kingsford', '悉尼机场', 'sydney airport',
            '曼彻斯特机场', 'manchester airport', 'man ',
        )
        campus_kw = (
            '利物浦大学', 'liverpool university', 'university of liverpool',
            '大学', 'campus', '主校区',
        )
        if 'airport' in cat or cat == 'airport':
            return any(k in instruction or k in text for k in airport_kw)
        if 'university' in cat or 'campus' in cat:
            return any(k in instruction or k in text for k in campus_kw)
        return False
