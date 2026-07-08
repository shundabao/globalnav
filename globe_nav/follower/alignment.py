"""Align LLM-decomposed segment goals with planner segment IDs."""

from __future__ import annotations

from globe_nav.follower.decomposer import DecomposedInstruction
from globe_nav.planner.segment_models import ModularTripPlan


def procedural_notes_map(
    decomposed: DecomposedInstruction, plan: ModularTripPlan,
) -> dict[str, str]:
    """Map planner segment_id → merged procedural notes from decomposer goals."""
    goals = decomposed.segment_goals
    if not goals:
        return {}

    if len(plan.segments) == 1:
        notes = ' '.join(g.procedural_notes for g in goals if g.procedural_notes)
        return {plan.segments[0].segment_id: notes.strip()}

    access_notes: list[str] = []
    flight_notes: list[str] = []
    egress_notes: list[str] = []
    past_fly = False

    for g in goals:
        mh = (g.mode_hint or '').lower()
        note = (g.procedural_notes or '').strip()
        if not note:
            continue
        if 'fly' in mh or 'flight' in mh or 'plane' in mh or '飞机' in note:
            past_fly = True
            flight_notes.append(note)
        elif past_fly:
            egress_notes.append(note)
        else:
            access_notes.append(note)

    result: dict[str, str] = {}
    for seg in plan.segments:
        sid = seg.segment_id
        if sid == 'seg_flight' or seg.segment_type == 'flight':
            result[sid] = ' '.join(flight_notes)
        elif sid == 'seg_egress' or (seg == plan.segments[-1] and len(plan.segments) > 1):
            result[sid] = ' '.join(egress_notes)
        elif sid == 'seg_access' or seg == plan.segments[0]:
            result[sid] = ' '.join(access_notes)
        else:
            idx = plan.segments.index(seg)
            result[sid] = goals[idx].procedural_notes if idx < len(goals) else ''
    return result
