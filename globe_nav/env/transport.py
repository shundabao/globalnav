"""Transport modes for global navigation."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Canonical modes — LLM must output one of these per leg
ALL_MODES = ('walk', 'drive', 'bus', 'train', 'fly', 'maritime')


class TransportMode(str, Enum):
    WALK = 'walk'
    DRIVE = 'drive'
    BUS = 'bus'
    TRAIN = 'train'
    FLY = 'fly'
    MARITIME = 'maritime'

    @classmethod
    def from_str(cls, value: str) -> 'TransportMode':
        aliases = {
            'car': 'drive', 'taxi': 'drive', 'rideshare': 'drive',
            'plane': 'fly', 'flight': 'fly',
            'rail': 'train', 'metro': 'train', 'subway': 'train',
            'hsr': 'train', 'high_speed_rail': 'train',
            'ship': 'maritime', 'ferry': 'maritime', 'boat': 'maritime',
            'foot': 'walk',
        }
        v = aliases.get(value.lower().strip(), value.lower().strip())
        return cls(v)

    @property
    def display_name(self) -> str:
        return {
            'walk': 'Walking',
            'drive': 'Driving',
            'bus': 'Bus',
            'train': 'Train',
            'fly': 'Flying',
            'maritime': 'Maritime',
        }[self.value]

    @property
    def speed_kmh(self) -> float:
        return {
            'walk': 5.0,
            'drive': 70.0,
            'bus': 50.0,
            'train': 200.0,
            'fly': 850.0,
            'maritime': 35.0,
        }[self.value]

    @property
    def action_space(self) -> list[str]:
        spaces = {
            'walk': ['forward', 'left', 'right', 'turn_around', 'stop'],
            'drive': ['forward', 'left', 'right', 'u_turn', 'merge', 'exit', 'stop'],
            'bus': ['board', 'depart', 'cruise', 'transfer', 'arrive', 'stop'],
            'train': ['board', 'depart', 'cruise', 'transfer', 'arrive', 'stop'],
            'fly': ['takeoff', 'climb', 'cruise', 'descend', 'land', 'taxi', 'stop'],
            'maritime': ['depart', 'cruise', 'enter_channel', 'dock', 'stop'],
        }
        return spaces[self.value]


@dataclass
class Waypoint:
    name: str
    lat: float
    lon: float
    place_type: str = 'city'
    iata: Optional[str] = None
    port_code: Optional[str] = None

    def coords(self) -> tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class RouteSegment:
    mode: TransportMode
    from_waypoint: Waypoint
    to_waypoint: Waypoint
    distance_km: float
    duration_hours: float
    instructions: list[str]
    geometry: list[tuple[float, float]]
    description: str = ''

    @property
    def eta_display(self) -> str:
        if self.duration_hours < 1:
            return f'{int(self.duration_hours * 60)} min'
        if self.duration_hours < 24:
            return f'{self.duration_hours:.1f} h'
        days = int(self.duration_hours // 24)
        hours = self.duration_hours % 24
        return f'{days}d {hours:.0f}h'
