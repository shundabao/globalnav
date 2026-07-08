"""Real road routing via OSRM (OpenStreetMap)."""

import time
from typing import Optional

import requests

from globe_nav.maps.cache import DiskCache
from globe_nav.planner.models import DetailedLeg, RouteStep

OSRM_BASE = 'https://router.project-osrm.org'
PROFILE = {'walk': 'foot', 'drive': 'driving'}
WALK_SPEED_KMH = 5.0
MAX_REALISTIC_WALK_KMH = 7.0


class OSRMClient:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache = DiskCache(cache_dir)
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        self._last_request = time.time()

    def route(
        self,
        from_lat: float, from_lon: float, from_name: str,
        to_lat: float, to_lon: float, to_name: str,
        mode: str = 'drive',
    ) -> Optional[DetailedLeg]:
        profile = PROFILE.get(mode, 'driving')
        cache_key = f'{profile}:{from_lat},{from_lon}->{to_lat},{to_lon}'
        cached = self.cache.get('osrm', cache_key)
        if cached:
            return self._leg_from_cache(cached, from_name, to_name, mode)

        self._rate_limit()
        coords = f'{from_lon},{from_lat};{to_lon},{to_lat}'
        url = f'{OSRM_BASE}/route/v1/{profile}/{coords}'
        try:
            resp = requests.get(
                url,
                params={'steps': 'true', 'overview': 'full', 'geometries': 'geojson'},
                headers={'User-Agent': 'GLOBALNAV/1.0'},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get('code') != 'Ok' or not data.get('routes'):
                return None
            route = data['routes'][0]
            leg_data = {
                'distance_m': route['distance'],
                'duration_s': route['duration'],
                'steps': self._extract_steps(route),
                'geometry': route['geometry']['coordinates'],
            }
            self.cache.set('osrm', cache_key, leg_data)
            return self._leg_from_cache(leg_data, from_name, to_name, mode,
                                        from_lat, from_lon, to_lat, to_lon)
        except (requests.RequestException, KeyError, ValueError):
            return None

    @staticmethod
    def _sanitized_walk_duration(mode: str, distance_km: float, duration_min: float) -> float:
        if mode != 'walk' or distance_km <= 0.1:
            return duration_min
        implied_kmh = distance_km / (duration_min / 60) if duration_min > 0 else 99
        if implied_kmh > MAX_REALISTIC_WALK_KMH:
            return distance_km / WALK_SPEED_KMH * 60
        return duration_min

    def _extract_steps(self, route: dict) -> list[dict]:
        steps = []
        for leg in route.get('legs', []):
            for step in leg.get('steps', []):
                maneuver = step.get('maneuver', {})
                name = step.get('name') or maneuver.get('type', '')
                instr = f'{maneuver.get("type", "continue")}'
                if name:
                    instr += f' on {name}'
                steps.append({
                    'instruction': instr,
                    'distance_m': step.get('distance', 0),
                    'duration_s': step.get('duration', 0),
                })
        return steps

    def _leg_from_cache(
        self, data: dict, from_name: str, to_name: str, mode: str,
        from_lat: float = 0, from_lon: float = 0,
        to_lat: float = 0, to_lon: float = 0,
    ) -> DetailedLeg:
        geom = data.get('geometry', [])
        geometry = [(c[1], c[0]) for c in geom] if geom else []
        if geometry:
            from_lat, from_lon = geometry[0]
            to_lat, to_lon = geometry[-1]
        distance_km = data['distance_m'] / 1000
        duration_min = self._sanitized_walk_duration(mode, distance_km, data['duration_s'] / 60)

        return DetailedLeg(
            mode=mode,
            from_name=from_name,
            to_name=to_name,
            from_lat=from_lat,
            from_lon=from_lon,
            to_lat=to_lat,
            to_lon=to_lon,
            distance_km=distance_km,
            duration_min=duration_min,
            steps=[RouteStep(s['instruction'], s['distance_m'], s['duration_s'])
                   for s in data.get('steps', [])],
            geometry=geometry,
            verified=True,
            note='OSRM/OpenStreetMap',
        )
