"""Transit stop discovery via OpenStreetMap Overpass API."""

import time
from dataclasses import dataclass
from typing import Optional

import requests

from globe_nav.maps.cache import DiskCache
from globe_nav.maps.geocoder import haversine_km

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'


@dataclass
class TransitStop:
    name: str
    lat: float
    lon: float
    stop_type: str  # train, tram, bus


class TransitFinder:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache = DiskCache(cache_dir)
        self._last_request = 0.0

    def nearest_stops(self, lat: float, lon: float, radius_m: int = 1200,
                      limit: int = 5) -> list[TransitStop]:
        cache_key = f'{lat:.4f},{lon:.4f}:{radius_m}'
        cached = self.cache.get('transit', cache_key)
        if cached:
            return [TransitStop(**s) for s in cached]

        query = f'''
        [out:json][timeout:25];
        (
          node(around:{radius_m},{lat},{lon})["railway"="station"];
          node(around:{radius_m},{lat},{lon})["railway"="halt"];
          node(around:{radius_m},{lat},{lon})["public_transport"="stop_position"]["railway"];
          node(around:{radius_m},{lat},{lon})["railway"="tram_stop"];
        );
        out body {limit};
        '''
        self._rate_limit()
        try:
            resp = requests.post(
                OVERPASS_URL, data={'data': query},
                headers={'User-Agent': 'GLOBALNAV/1.0'}, timeout=30,
            )
            resp.raise_for_status()
            elements = resp.json().get('elements', [])
            stops = []
            for el in elements:
                tags = el.get('tags', {})
                name = tags.get('name') or tags.get('ref') or 'Transit stop'
                stype = 'train'
                if tags.get('railway') == 'tram_stop':
                    stype = 'tram'
                stops.append(TransitStop(name=name, lat=el['lat'], lon=el['lon'], stop_type=stype))
            stops.sort(key=lambda s: haversine_km(lat, lon, s.lat, s.lon))
            stops = stops[:limit]
            self.cache.set('transit', cache_key, [s.__dict__ for s in stops])
            return stops
        except (requests.RequestException, KeyError, ValueError):
            return []

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_request = time.time()

    def estimate_transit_leg(self, stop_a: TransitStop, stop_b: TransitStop) -> Optional[dict]:
        """Estimate rail/tram leg between two stops (no GTFS schedule — distance-based)."""
        dist = haversine_km(stop_a.lat, stop_a.lon, stop_b.lat, stop_b.lon)
        if dist < 0.1:
            return None
        speed = 35 if stop_a.stop_type == 'tram' else 60
        return {
            'distance_km': dist * 1.3,
            'duration_min': dist * 1.3 / speed * 60,
            'mode': 'tram' if stop_a.stop_type == 'tram' else 'train',
        }
