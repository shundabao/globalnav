"""LLM + heuristic default selection per trip segment."""

import json
from typing import Optional

from globe_nav.config import DEFAULT_MODEL
from globe_nav.planner.segment_models import ModularTripPlan, TripSegment


class SegmentRecommender:
    """Pick default option per segment; LLM when available, else fastest."""

    def __init__(self, model: str = DEFAULT_MODEL, use_llm: bool = True):
        self.model = model
        self.use_llm = use_llm

    def apply_defaults(self, plan: ModularTripPlan) -> None:
        picks = self._llm_recommend(plan) if self.use_llm else {}
        for seg in plan.segments:
            if seg.segment_id in picks:
                oid = picks[seg.segment_id]
                if any(o.option_id == oid for o in seg.options):
                    seg.default_option_id = oid
                    for o in seg.options:
                        o.is_recommended = o.option_id == oid
                    continue
            self._heuristic_default(seg)

    def _heuristic_default(self, seg: TripSegment) -> None:
        if not seg.options:
            return
        if seg.segment_type == 'flight':
            # prefer direct flight (fewest micro legs)
            best = min(seg.options, key=lambda o: (len(o.micro_legs), o.duration_min))
        else:
            # prefer drive for airport access if similar time, else fastest
            drive = next((o for o in seg.options if o.label == '驾车'), None)
            fastest = min(seg.options, key=lambda o: o.duration_min)
            best = drive if drive and drive.duration_min < fastest.duration_min * 1.5 else fastest
        seg.default_option_id = best.option_id
        for o in seg.options:
            o.is_recommended = o.option_id == best.option_id

    def _llm_recommend(self, plan: ModularTripPlan) -> dict[str, str]:
        try:
            from globe_nav.llm_client import LLMClient
            client = LLMClient(model=self.model, max_tokens=400)
            summary = []
            for seg in plan.segments:
                opts = [
                    f"{o.option_id}: {o.label} ({o.duration_min:.0f}min)"
                    for o in seg.options[:8]
                ]
                summary.append({
                    'segment_id': seg.segment_id,
                    'title': seg.title,
                    'type': seg.segment_type,
                    'options': opts,
                })
            raw = client.chat_json(
                system=(
                    'Pick one default option_id per segment for this trip. '
                    'Prefer realistic choices: drive/taxi to airport, valid flights, '
                    'train/bus for last mile in China when available. Return JSON: '
                    '{"selections": {"segment_id": "option_id", ...}}'
                ),
                user=json.dumps({
                    'trip': f'{plan.origin} → {plan.destination}',
                    'segments': summary,
                }, ensure_ascii=False),
            )
            return raw.get('selections', {})
        except Exception:
            return {}
