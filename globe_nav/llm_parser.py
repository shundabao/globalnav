"""LLM parses user intent only — transport planning is done by MobilityEnvironment."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.config import DEFAULT_MODEL
from globe_nav.llm_client import LLMClient

SYSTEM_PROMPT = """You parse travel instructions into origin and destination ONLY.

Do NOT plan routes or choose transport modes — the navigation environment does that.

Rules:
1. Extract the most specific origin and destination place names.
2. Set needs_clarification=true ONLY when origin or destination is genuinely missing or ambiguous.
3. Do NOT ask about airports if the user already named one (e.g. 悉尼机场, Sydney Airport, SYD).
4. Do NOT ask about university campus if the user already named one (e.g. 利物浦大学, University of Liverpool).
5. Procedural waypoints (central station, facing direction) are NOT ambiguities — ignore them for clarification.
6. Return valid JSON only.

JSON schema:
{
  "origin": "full place name with city/country if known",
  "destination": "full place name with city/country if known",
  "needs_clarification": false,
  "questions": [
    {"category": "snake_case_id", "question": "question text", "options": ["opt1", "opt2"]}
  ]
}
"""


@dataclass
class ParsedTrip:
    origin: str
    destination: str
    needs_clarification: bool = False
    questions: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # legs are NOT from LLM anymore
    @property
    def legs(self) -> list:
        return []

    @property
    def modes(self) -> list[str]:
        return []


class LLMInstructionParser:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = LLMClient(model=model, max_tokens=512)

    def parse(self, instruction: str, qa_history: Optional[list[dict]] = None) -> ParsedTrip:
        user_parts = [f'User instruction:\n{instruction}']
        if qa_history:
            user_parts.append('\nClarification history:')
            for item in qa_history:
                user_parts.append(f"Q ({item['category']}): {item['question']}")
                user_parts.append(f"A: {item['answer']}")

        raw = self.llm.chat_json(SYSTEM_PROMPT, '\n'.join(user_parts))
        questions = raw.get('questions', [])
        if questions and isinstance(questions[0], str):
            questions = [{'category': f'q{i}', 'question': q, 'options': []}
                         for i, q in enumerate(questions)]

        return ParsedTrip(
            origin=raw.get('origin', ''),
            destination=raw.get('destination', ''),
            needs_clarification=bool(raw.get('needs_clarification', False)),
            questions=questions,
            raw=raw,
        )
