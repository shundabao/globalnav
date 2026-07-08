"""Modular trip segment models for interactive GUI planning."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.planner.models import DetailedLeg


def _bounds_from_points(points: list[list[float]]) -> Optional[dict]:
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return {
        'south': min(lats), 'north': max(lats),
        'west': min(lons), 'east': max(lons),
    }


def format_duration(minutes: float) -> str:
    if minutes < 1:
        return '< 1 min'
    if minutes < 60:
        return f'{max(1, round(minutes))} min'
    h = minutes / 60
    if h < 24:
        return f'{h:.1f} h'
    return f'{int(h // 24)}d {h % 24:.0f}h'


@dataclass
class MicroLeg:
    mode: str
    from_name: str
    to_name: str
    distance_km: float
    duration_min: float
    steps: list[str] = field(default_factory=list)
    note: str = ''
    geometry: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_detailed(cls, leg: DetailedLeg) -> 'MicroLeg':
        return cls(
            mode=leg.mode,
            from_name=leg.from_name,
            to_name=leg.to_name,
            distance_km=leg.distance_km,
            duration_min=leg.duration_min,
            steps=[s.instruction for s in leg.steps[:30]],
            note=leg.note,
            geometry=list(leg.geometry),
        )


@dataclass
class SegmentOption:
    option_id: str
    label: str
    mode_chain: str
    micro_legs: list[MicroLeg] = field(default_factory=list)
    duration_min: float = 0.0
    distance_km: float = 0.0
    verified: bool = True
    is_recommended: bool = False
    tooltip: str = ''

    def to_dict(self) -> dict:
        return {
            'option_id': self.option_id,
            'label': self.label,
            'mode_chain': self.mode_chain,
            'duration_min': round(self.duration_min, 1),
            'duration_display': format_duration(self.duration_min),
            'distance_km': round(self.distance_km, 1),
            'verified': self.verified,
            'is_recommended': self.is_recommended,
            'tooltip': self.tooltip or f'{format_duration(self.duration_min)} · {self.distance_km:.1f} km',
            'micro_legs': [
                {
                    'mode': m.mode,
                    'from': m.from_name,
                    'to': m.to_name,
                    'duration_display': format_duration(m.duration_min),
                    'distance_km': round(m.distance_km, 1),
                    'steps': m.steps,
                    'note': m.note,
                    'geometry': [[lat, lon] for lat, lon in m.geometry],
                }
                for m in self.micro_legs
            ],
        }


@dataclass
class TripSegment:
    segment_id: str
    title: str
    segment_type: str  # local | flight
    from_name: str
    to_name: str
    options: list[SegmentOption] = field(default_factory=list)
    default_option_id: str = ''
    description: str = ''

    def to_dict(self) -> dict:
        return {
            'segment_id': self.segment_id,
            'title': self.title,
            'segment_type': self.segment_type,
            'from': self.from_name,
            'to': self.to_name,
            'description': self.description,
            'default_option_id': self.default_option_id,
            'options': [o.to_dict() for o in self.options],
        }


@dataclass
class ModularTripPlan:
    origin: str
    destination: str
    segments: list[TripSegment] = field(default_factory=list)
    instruction: str = ''

    @property
    def default_total_display(self) -> str:
        return format_duration(self.default_total_min)

    @property
    def default_total_min(self) -> float:
        total = 0.0
        for seg in self.segments:
            opt = next((o for o in seg.options if o.option_id == seg.default_option_id), None)
            if opt:
                total += opt.duration_min
            elif seg.options:
                total += seg.options[0].duration_min
        return total

    def to_dict(self) -> dict:
        return {
            'origin': self.origin,
            'destination': self.destination,
            'instruction': self.instruction,
            'default_total_min': round(self.default_total_min, 1),
            'default_total_display': format_duration(self.default_total_min),
            'segments': [s.to_dict() for s in self.segments],
        }

    def path_for_selections(self, selections: Optional[dict[str, str]] = None) -> dict:
        """Full route polylines grouped by segment for map display."""
        selections = selections or {}
        segments_out = []
        all_points: list[list[float]] = []
        for seg in self.segments:
            oid = selections.get(seg.segment_id, seg.default_option_id)
            opt = next((o for o in seg.options if o.option_id == oid),
                         seg.options[0] if seg.options else None)
            if not opt:
                continue
            seg_polylines = []
            for micro in opt.micro_legs:
                geom = [[lat, lon] for lat, lon in micro.geometry if lat or lon]
                if geom:
                    seg_polylines.append({
                        'mode': micro.mode,
                        'from': micro.from_name,
                        'to': micro.to_name,
                        'geometry': geom,
                    })
                    all_points.extend(geom)
            segments_out.append({
                'segment_id': seg.segment_id,
                'title': seg.title,
                'polylines': seg_polylines,
            })
        return {'segments': segments_out, 'bounds': _bounds_from_points(all_points)}

    def compute_total(self, selections: dict[str, str]) -> dict:
        legs = []
        total_min = 0.0
        total_km = 0.0
        for seg in self.segments:
            oid = selections.get(seg.segment_id, seg.default_option_id)
            opt = next((o for o in seg.options if o.option_id == oid), seg.options[0] if seg.options else None)
            if not opt:
                continue
            total_min += opt.duration_min
            total_km += opt.distance_km
            legs.extend(opt.micro_legs)
        return {
            'total_min': round(total_min, 1),
            'total_display': format_duration(total_min),
            'total_km': round(total_km, 1),
            'selections': selections,
            'leg_count': len(legs),
        }
