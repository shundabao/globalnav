"""Data models for concrete multi-modal route options."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouteStep:
    instruction: str
    distance_m: float = 0.0
    duration_s: float = 0.0


@dataclass
class DetailedLeg:
    mode: str
    from_name: str
    to_name: str
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    distance_km: float
    duration_min: float
    steps: list[RouteStep] = field(default_factory=list)
    geometry: list[tuple[float, float]] = field(default_factory=list)
    verified: bool = True
    note: str = ''

    def summary(self) -> str:
        return f'[{self.mode}] {self.from_name} → {self.to_name} ({self.distance_km:.1f} km, {self.duration_min:.0f} min)'


@dataclass
class RouteOption:
    option_id: str
    legs: list[DetailedLeg] = field(default_factory=list)
    description: str = ''

    @property
    def total_distance_km(self) -> float:
        return sum(l.distance_km for l in self.legs)

    @property
    def total_duration_min(self) -> float:
        return sum(l.duration_min for l in self.legs)

    @property
    def total_duration_display(self) -> str:
        m = self.total_duration_min
        if m < 60:
            return f'{m:.0f} min'
        h = m / 60
        if h < 24:
            return f'{h:.1f} h'
        return f'{int(h // 24)}d {h % 24:.0f}h'

    @property
    def mode_chain(self) -> str:
        return ' → '.join(l.mode for l in self.legs)

    def to_dict(self) -> dict:
        return {
            'option_id': self.option_id,
            'description': self.description,
            'mode_chain': self.mode_chain,
            'total_distance_km': round(self.total_distance_km, 1),
            'total_duration': self.total_duration_display,
            'legs': [
                {
                    'mode': l.mode,
                    'from': l.from_name,
                    'to': l.to_name,
                    'distance_km': round(l.distance_km, 1),
                    'duration_min': round(l.duration_min, 0),
                    'verified': l.verified,
                    'note': l.note,
                    'steps': [s.instruction for s in l.steps[:20]],
                    'step_count': len(l.steps),
                }
                for l in self.legs
            ],
        }
