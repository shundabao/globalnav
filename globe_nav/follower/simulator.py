"""Step-by-step global navigation simulator (VELMA-style state transitions)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from typing import TYPE_CHECKING

from globe_nav.follower.alignment import procedural_notes_map
from globe_nav.follower.decomposer import DecomposedInstruction
from globe_nav.planner.segment_models import ModularTripPlan, SegmentOption

if TYPE_CHECKING:
    from globe_nav.maps.streetview import GlobalStreetViewProvider

# Per-mode action spaces
ACTION_SPACES = {
    'walk': ['forward', 'left', 'right', 'turn_around', 'stop'],
    'drive': ['forward', 'left', 'right', 'u_turn', 'stop'],
    'bus': ['board', 'forward', 'cruise', 'arrive', 'stop'],
    'train': ['board', 'forward', 'cruise', 'arrive', 'stop'],
    'tram': ['board', 'forward', 'cruise', 'arrive', 'stop'],
    'fly': ['takeoff', 'cruise', 'land', 'taxi', 'stop'],
    'maritime': ['depart', 'cruise', 'dock', 'stop'],
}


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_label(deg: float) -> str:
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[int((deg + 22.5) / 45) % 8]


@dataclass
class SimPoint:
    lat: float
    lon: float
    instruction: str = ''
    is_turn: bool = False


@dataclass
class ExecutionLeg:
    mode: str
    from_name: str
    to_name: str
    points: list[SimPoint] = field(default_factory=list)
    phase_labels: list[str] = field(default_factory=list)
    segment_id: str = ''
    procedural_notes: str = ''


@dataclass
class SimulatorState:
    leg_index: int = 0
    point_index: int = 0
    phase_index: int = 0
    heading: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    actions: list[str] = field(default_factory=list)
    done: bool = False
    success: bool = False


class GlobalInstructionSimulator:
    """
    Executes a modular trip plan step-by-step.
    Walk/drive: advance along OSRM step points; fly/train: discrete phases.
    """

    def __init__(
        self, plan: ModularTripPlan, decomposed: Optional[DecomposedInstruction] = None,
        selections: Optional[dict[str, str]] = None,
        streetview: Optional['GlobalStreetViewProvider'] = None,
    ):
        self.plan = plan
        self.decomposed = decomposed
        self.selections = selections or {}
        self.streetview = streetview
        self.legs: list[ExecutionLeg] = self._build_execution_legs()
        self.state = SimulatorState(actions=['init'])
        if self.legs:
            self._init_position()

    def _build_execution_legs(self) -> list[ExecutionLeg]:
        legs: list[ExecutionLeg] = []
        goal_map = (
            procedural_notes_map(self.decomposed, self.plan)
            if self.decomposed else {}
        )

        for seg in self.plan.segments:
            oid = self.selections.get(seg.segment_id, seg.default_option_id)
            opt: SegmentOption = next(
                (o for o in seg.options if o.option_id == oid),
                seg.options[0] if seg.options else None,
            )
            if not opt:
                continue
            for micro in opt.micro_legs:
                el = ExecutionLeg(
                    mode=micro.mode,
                    from_name=micro.from_name,
                    to_name=micro.to_name,
                    segment_id=seg.segment_id,
                    procedural_notes=goal_map.get(seg.segment_id, ''),
                )
                el.points = self._points_from_micro(micro)
                el.phase_labels = self._phases_for_mode(micro.mode)
                legs.append(el)
        return legs

    def _points_from_micro(self, micro) -> list[SimPoint]:
        points = []
        geom = getattr(micro, 'geometry', None) or []
        steps = micro.steps or []
        if geom:
            n_steps = len(steps) or 1
            for i, (lat, lon) in enumerate(geom):
                step_idx = min(int(i * n_steps / max(len(geom), 1)), n_steps - 1)
                instr = steps[step_idx] if steps else ''
                is_turn = any(k in instr.lower() for k in ('turn', 'left', 'right', 'u-turn'))
                points.append(SimPoint(lat, lon, instr, is_turn))
        elif steps:
            for s in steps:
                is_turn = any(k in s.lower() for k in ('turn', 'left', 'right'))
                points.append(SimPoint(0, 0, s, is_turn))
        else:
            points.append(SimPoint(0, 0, f'{micro.mode} {micro.from_name} → {micro.to_name}'))
        return points

    @staticmethod
    def _phases_for_mode(mode: str) -> list[str]:
        m = {
            'walk': ['forward'] * 3 + ['stop'],
            'drive': ['forward'] * 3 + ['stop'],
            'bus': ['board', 'cruise', 'arrive', 'stop'],
            'train': ['board', 'cruise', 'arrive', 'stop'],
            'tram': ['board', 'cruise', 'arrive', 'stop'],
            'fly': ['takeoff', 'cruise', 'land', 'stop'],
            'maritime': ['depart', 'cruise', 'dock', 'stop'],
        }
        return m.get(mode, ['forward', 'stop'])

    def _init_position(self) -> None:
        leg = self.legs[0]
        if leg.points and leg.points[0].lat != 0:
            self.state.lat = leg.points[0].lat
            self.state.lon = leg.points[0].lon
            if len(leg.points) > 1:
                self.state.heading = bearing_deg(
                    leg.points[0].lat, leg.points[0].lon,
                    leg.points[1].lat, leg.points[1].lon,
                )
        if self.decomposed and self.decomposed.initial_facing:
            self._apply_facing_hint(self.decomposed.initial_facing)

    def _apply_facing_hint(self, facing: str) -> None:
        f = facing.lower()
        for label, deg in [('north', 0), ('east', 90), ('south', 180), ('west', 270),
                           ('ne', 45), ('se', 135), ('sw', 225), ('nw', 315)]:
            if label in f:
                self.state.heading = deg
                break

    @property
    def current_leg(self) -> Optional[ExecutionLeg]:
        if self.state.leg_index < len(self.legs):
            return self.legs[self.state.leg_index]
        return None

    def get_action_space(self) -> list[str]:
        leg = self.current_leg
        if not leg or self.state.done:
            return ['stop']
        if leg.mode in ('walk', 'drive'):
            return ACTION_SPACES[leg.mode]
        return ACTION_SPACES.get(leg.mode, ['forward', 'stop'])

    def get_observation(self) -> dict:
        leg = self.current_leg
        if not leg or self.state.done:
            return {'done': True, 'message': 'Navigation finished'}

        pt = leg.points[min(self.state.point_index, len(leg.points) - 1)] if leg.points else SimPoint(0, 0, '')
        next_pt = leg.points[min(self.state.point_index + 1, len(leg.points) - 1)] if leg.points else pt

        if leg.mode in ('walk', 'drive'):
            if pt.lat and next_pt.lat:
                self.state.heading = bearing_deg(pt.lat, pt.lon, next_pt.lat, next_pt.lon)
                self.state.lat, self.state.lon = pt.lat, pt.lon
        elif leg.points and any(p.lat for p in leg.points):
            phase_frac = self.state.phase_index / max(len(leg.phase_labels) - 1, 1)
            idx = min(int(phase_frac * (len(leg.points) - 1)), len(leg.points) - 1)
            pt = leg.points[idx]
            if pt.lat:
                self.state.lat, self.state.lon = pt.lat, pt.lon
                if idx < len(leg.points) - 1 and leg.points[idx + 1].lat:
                    nxt = leg.points[idx + 1]
                    self.state.heading = bearing_deg(pt.lat, pt.lon, nxt.lat, nxt.lon)

        seg_total = len(self.plan.segments)
        seg_idx = self._plan_segment_index()

        obs = {
            'done': False,
            'leg_index': self.state.leg_index + 1,
            'leg_total': len(self.legs),
            'mode': leg.mode,
            'segment_index': seg_idx + 1,
            'segment_total': seg_total,
            'segment_id': leg.segment_id,
            'from': leg.from_name,
            'to': leg.to_name,
            'lat': self.state.lat,
            'lon': self.state.lon,
            'heading_deg': round(self.state.heading, 1),
            'facing': bearing_label(self.state.heading),
            'osm_instruction': pt.instruction,
            'is_turn_ahead': pt.is_turn,
            'procedural_notes': leg.procedural_notes,
            'progress': f'point {self.state.point_index + 1}/{max(len(leg.points), 1)}',
            'available_actions': self.get_action_space(),
            'initial_facing': self.decomposed.initial_facing if self.decomposed else '',
        }
        if leg.mode in ('walk', 'drive'):
            if self.streetview and self.streetview.enabled:
                self.streetview.enrich_observation(obs)
            else:
                obs['streetview'] = {
                    'available': False, 'reason': 'disabled', 'message': 'Street View is disabled',
                }
        elif leg.mode in ('fly', 'train', 'bus', 'tram', 'maritime'):
            obs['streetview'] = {
                'available': False,
                'reason': f'{leg.mode}_phase',
                'message': f'No Street View for the {leg.mode} phase',
                'phase': leg.phase_labels[min(self.state.phase_index, len(leg.phase_labels) - 1)]
                if leg.phase_labels else '',
            }
        return obs

    def _plan_segment_index(self) -> int:
        if not self.current_leg:
            return 0
        for i, seg in enumerate(self.plan.segments):
            if seg.segment_id == self.current_leg.segment_id:
                return i
        return 0

    def step(self, action: str) -> tuple[dict, bool, dict]:
        action = action.lower().strip().replace(' ', '_')
        leg = self.current_leg
        if not leg:
            self.state.done = True
            return self.get_observation(), True, {'success': True}

        self.state.actions.append(action)

        if action == 'stop':
            at_end = self.state.leg_index >= len(self.legs) - 1
            self.state.done = True
            self.state.success = at_end
            return self.get_observation(), True, {'success': at_end, 'reason': 'stop'}

        if leg.mode in ('walk', 'drive'):
            info = self._step_ground(action, leg)
        else:
            info = self._step_phased(action, leg)

        done = self.state.done
        return self.get_observation(), done, info

    def _step_ground(self, action: str, leg: ExecutionLeg) -> dict:
        if action == 'forward':
            if self.state.point_index < len(leg.points) - 1:
                self.state.point_index += 1
            else:
                return self._advance_leg()
        elif action in ('left', 'right'):
            delta = 90 if action == 'left' else -90
            self.state.heading = (self.state.heading + delta) % 360
            if leg.points and self.state.point_index < len(leg.points) - 1:
                self.state.point_index += 1
        elif action in ('turn_around', 'u_turn'):
            self.state.heading = (self.state.heading + 180) % 360
        return {'action_applied': action}

    def _step_phased(self, action: str, leg: ExecutionLeg) -> dict:
        phases = leg.phase_labels
        expected = phases[min(self.state.phase_index, len(phases) - 1)]
        preferred = {
            'takeoff': 'takeoff', 'board': 'board', 'depart': 'depart',
            'cruise': 'cruise', 'forward': 'forward',
            'land': 'land', 'arrive': 'arrive', 'dock': 'dock', 'taxi': 'taxi',
        }
        if action in preferred.values() or action == expected:
            self.state.phase_index += 1
            if self.state.phase_index >= len(phases) - 1:
                return self._advance_leg()
        return {'action_applied': action, 'expected': expected}

    def _advance_leg(self) -> dict:
        self.state.leg_index += 1
        self.state.point_index = 0
        self.state.phase_index = 0
        if self.state.leg_index >= len(self.legs):
            self.state.done = True
            self.state.success = True
            return {'success': True, 'reason': 'reached_goal'}
        return {'segment_advanced': True}
