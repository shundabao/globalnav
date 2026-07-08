"""Flight connectivity from OpenFlights route database."""

import csv
import io
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from globe_nav.maps.geocoder import haversine_km
from globe_nav.maps.geo import great_circle_points
from globe_nav.planner.models import DetailedLeg, RouteStep

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'openflights')
ROUTES_URL = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat'
AIRPORTS_URL = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'

# Exclude heliports / military — they break "nearest airport" for city centers
EXCLUDE_NAME = (
    'heliport', 'Heliport', 'Seaplane', 'Balloon', 'Air Base', 'Airbase',
    'NAS ', 'AAF', 'NOLF', 'Airstrip', 'Ultralight',
)

# Fallback hubs when OpenFlights has no path
GLOBAL_HUBS = ('DXB', 'LHR', 'SIN', 'HKG', 'LAX', 'FRA', 'DOH', 'ICN', 'PVG', 'AMS')


@dataclass
class Airport:
    iata: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    route_count: int = 0


class FlightGraph:
    CRUISE_KMH = 850.0
    LAYOVER_MIN = 90.0

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.airports: dict[str, Airport] = {}
        self.routes: dict[str, set[str]] = defaultdict(set)
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_airports()
        self._load_routes()
        self._count_routes()
        self._loaded = True

    def _load_airports(self) -> None:
        path = os.path.join(self.data_dir, 'airports.dat')
        if not os.path.exists(path):
            self._download(AIRPORTS_URL, path)
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = self._parse_csv_line(line)
                if len(parts) < 8:
                    continue
                iata = parts[4].strip()
                if not iata or iata == '\\N' or len(iata) != 3:
                    continue
                name = parts[1]
                if any(ex in name for ex in EXCLUDE_NAME):
                    continue
                try:
                    self.airports[iata] = Airport(
                        iata=iata,
                        name=name,
                        city=parts[2],
                        country=parts[3],
                        lat=float(parts[6]),
                        lon=float(parts[7]),
                    )
                except (ValueError, IndexError):
                    continue

    def _load_routes(self) -> None:
        path = os.path.join(self.data_dir, 'routes.dat')
        if not os.path.exists(path):
            self._download(ROUTES_URL, path)
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = self._parse_csv_line(line)
                if len(parts) < 5:
                    continue
                src, dst = parts[2].strip(), parts[4].strip()
                if src and dst and src in self.airports and dst in self.airports:
                    self.routes[src].add(dst)

    def _count_routes(self) -> None:
        inbound: dict[str, int] = defaultdict(int)
        for src, dsts in self.routes.items():
            for dst in dsts:
                inbound[dst] += 1
        for iata, ap in self.airports.items():
            ap.route_count = len(self.routes.get(iata, set())) + inbound.get(iata, 0)

    @staticmethod
    def _parse_csv_line(line: str) -> list[str]:
        return next(csv.reader(io.StringIO(line)))

    @staticmethod
    def _download(url: str, path: str) -> None:
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(resp.text)

    def nearest_commercial_airports(
        self, lat: float, lon: float, n: int = 3, max_km: float = 200,
    ) -> list[Airport]:
        """Nearest airports with scheduled routes (excludes heliports)."""
        self.ensure_loaded()
        ranked = []
        for ap in self.airports.values():
            if ap.route_count < 10:
                continue
            d = haversine_km(lat, lon, ap.lat, ap.lon)
            if d <= max_km:
                ranked.append((d, -ap.route_count, ap))
        ranked.sort(key=lambda x: (x[0], x[1]))
        seen = set()
        result = []
        for _, _, ap in ranked:
            if ap.iata in seen:
                continue
            seen.add(ap.iata)
            result.append(ap)
            if len(result) >= n:
                break
        return result

    def has_direct(self, src: str, dst: str) -> bool:
        self.ensure_loaded()
        return dst in self.routes.get(src, set())

    def find_paths(self, src: str, dst: str, max_stops: int = 2) -> list[list[str]]:
        self.ensure_loaded()
        if src not in self.airports or dst not in self.airports:
            return []
        if src == dst:
            return [[src]]

        results: list[list[str]] = []
        seen: set[tuple] = set()

        def add(path: list[str]) -> None:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                results.append(path)

        if self.has_direct(src, dst):
            add([src, dst])

        if max_stops >= 1:
            for mid in self.routes.get(src, set()):
                if self.has_direct(mid, dst):
                    add([src, mid, dst])

        if max_stops >= 2:
            for m1 in self.routes.get(src, set()):
                for m2 in self.routes.get(m1, set()):
                    if self.has_direct(m2, dst) and len({src, m1, m2, dst}) == 4:
                        add([src, m1, m2, dst])

        results.sort(key=lambda p: (len(p), self._path_distance(p)))
        return results[:8]

    def _path_distance(self, path: list[str]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            a, b = self.airports[path[i]], self.airports[path[i + 1]]
            total += haversine_km(a.lat, a.lon, b.lat, b.lon)
        return total

    def estimate_via_hubs(self, src: str, dst: str) -> list[list[str]]:
        """Last-resort: route through major global hubs (estimated, not in OpenFlights)."""
        self.ensure_loaded()
        if src not in self.airports or dst not in self.airports:
            return []
        results = []
        for hub in GLOBAL_HUBS:
            if hub in (src, dst) or hub not in self.airports:
                continue
            results.append([src, hub, dst])
        return results[:3]

    def build_flight_legs(self, path: list[str], estimated: bool = False) -> list[DetailedLeg]:
        legs = []
        for i in range(len(path) - 1):
            a = self.airports[path[i]]
            b = self.airports[path[i + 1]]
            dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
            fly_min = dist / self.CRUISE_KMH * 60 + 40
            note = (
                f'Estimated via hub (not in OpenFlights DB): {a.iata}-{b.iata}'
                if estimated else f'OpenFlights: {a.iata}-{b.iata} route on record'
            )
            legs.append(DetailedLeg(
                mode='fly',
                from_name=f'{a.name} ({a.iata})',
                to_name=f'{b.name} ({b.iata})',
                from_lat=a.lat, from_lon=a.lon,
                to_lat=b.lat, to_lon=b.lon,
                distance_km=dist,
                duration_min=fly_min,
                steps=[RouteStep(f'Fly {a.iata} → {b.iata}')],
                geometry=great_circle_points(a.lat, a.lon, b.lat, b.lon, n=32),
                verified=not estimated,
                note=note,
            ))
            if i < len(path) - 2:
                legs[-1].duration_min += self.LAYOVER_MIN
                legs[-1].note += f'; {self.LAYOVER_MIN:.0f} min layover at {b.iata}'
        return legs

    # backward compat
    def nearest_airports(self, lat: float, lon: float, n: int = 3, max_km: float = 150) -> list[Airport]:
        return self.nearest_commercial_airports(lat, lon, n=n, max_km=max_km)
