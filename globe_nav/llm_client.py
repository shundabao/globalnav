"""OpenAI client wrapper for GLOBALNAV."""

import json
import time
from typing import Any, Optional

from globe_nav.config import DEFAULT_MODEL, get_openai_api_key


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install 'openai>=1.0'") from e
        self.client = OpenAI(api_key=get_openai_api_key())

    def _completion_kwargs(self, max_tokens: int) -> dict:
        """Return token/decoding arguments supported by the selected model."""
        model = self.model.lower()
        if model.startswith(('gpt-5', 'o1', 'o3', 'o4')):
            return {
                'max_completion_tokens': max_tokens,
                'reasoning_effort': 'minimal',
            }
        return {
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }

    def chat_json(self, system: str, user: str, max_tokens: Optional[int] = None) -> dict:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                    response_format={'type': 'json_object'},
                    **self._completion_kwargs(max_tokens or self.max_tokens),
                )
                text = resp.choices[0].message.content or '{}'
                return json.loads(text)
            except json.JSONDecodeError:
                return {}
            except Exception as e:
                if attempt == 2:
                    raise
                print(f'LLM error (retry {attempt + 1}): {e}')
                time.sleep(2)
        return {}

    def chat_text(self, system: str, user: str, max_tokens: int = 32) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                    **self._completion_kwargs(max_tokens),
                )
                return (resp.choices[0].message.content or '').strip()
            except Exception as e:
                if attempt == 2:
                    raise
                print(f'LLM error (retry {attempt + 1}): {e}')
                time.sleep(2)
        return ''
