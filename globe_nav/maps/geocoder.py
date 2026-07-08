"""Global geocoding via Nominatim (OpenStreetMap) with offline fallback."""

import math
import time
from typing import Optional

import requests

from globe_nav.env.transport import Waypoint
from globe_nav.maps.cache import DiskCache

# Built-in landmarks for offline/demo use (global coverage)
OFFLINE_PLACES: dict[str, dict] = {
    'beijing': {'lat': 39.9042, 'lon': 116.4074, 'type': 'city'},
    'shanghai': {'lat': 31.2304, 'lon': 121.4737, 'type': 'city'},
    'tokyo': {'lat': 35.6762, 'lon': 139.6503, 'type': 'city'},
    'new york': {'lat': 40.7128, 'lon': -74.0060, 'type': 'city'},
    'london': {'lat': 51.5074, 'lon': -0.1278, 'type': 'city'},
    'paris': {'lat': 48.8566, 'lon': 2.3522, 'type': 'city'},
    'singapore': {'lat': 1.3521, 'lon': 103.8198, 'type': 'city'},
    'sydney': {'lat': -33.8688, 'lon': 151.2093, 'type': 'city'},
    'dubai': {'lat': 25.2048, 'lon': 55.2708, 'type': 'city'},
    'san francisco': {'lat': 37.7749, 'lon': -122.4194, 'type': 'city'},
    'los angeles': {'lat': 34.0522, 'lon': -118.2437, 'type': 'city'},
    'hong kong': {'lat': 22.3193, 'lon': 114.1694, 'type': 'city'},
    'peking airport': {'lat': 40.0799, 'lon': 116.6031, 'type': 'airport', 'iata': 'PEK'},
    'beijing capital airport': {'lat': 40.0799, 'lon': 116.6031, 'type': 'airport', 'iata': 'PEK'},
    'jfk': {'lat': 40.6413, 'lon': -73.7781, 'type': 'airport', 'iata': 'JFK'},
    'john f kennedy airport': {'lat': 40.6413, 'lon': -73.7781, 'type': 'airport', 'iata': 'JFK'},
    'heathrow': {'lat': 51.4700, 'lon': -0.4543, 'type': 'airport', 'iata': 'LHR'},
    'london heathrow': {'lat': 51.4700, 'lon': -0.4543, 'type': 'airport', 'iata': 'LHR'},
    'narita': {'lat': 35.7720, 'lon': 140.3929, 'type': 'airport', 'iata': 'NRT'},
    'shanghai pudong': {'lat': 31.1443, 'lon': 121.8083, 'type': 'airport', 'iata': 'PVG'},
    'port of shanghai': {'lat': 31.2304, 'lon': 121.4900, 'type': 'port', 'port_code': 'CNSHA'},
    'port of singapore': {'lat': 1.2644, 'lon': 103.8220, 'type': 'port', 'port_code': 'SGSIN'},
    'port of rotterdam': {'lat': 51.9496, 'lon': 4.1453, 'type': 'port', 'port_code': 'NLRTM'},
    'suez canal': {'lat': 30.0444, 'lon': 32.3482, 'type': 'channel'},
    'panama canal': {'lat': 9.0800, 'lon': -79.6800, 'type': 'channel'},
    'eiffel tower': {'lat': 48.8584, 'lon': 2.2945, 'type': 'landmark'},
    'statue of liberty': {'lat': 40.6892, 'lon': -74.0445, 'type': 'landmark'},
    'golden gate bridge': {'lat': 37.8199, 'lon': -122.4783, 'type': 'landmark'},
    # Australia / China corridor (UTS → Jiuzhaigou demo)
    'uts': {'lat': -33.8833, 'lon': 151.2006, 'type': 'university'},
    'sydney opera house': {'lat': -33.8568, 'lon': 151.2153, 'type': 'landmark'},
    'times square': {'lat': 40.7580, 'lon': -73.9855, 'type': 'landmark'},
    'times square new york': {'lat': 40.7580, 'lon': -73.9855, 'type': 'landmark'},
    '纽约时代广场': {'lat': 40.7580, 'lon': -73.9855, 'type': 'landmark'},
    'liverpool': {'lat': 53.4084, 'lon': -2.9916, 'type': 'city'},
    'university of liverpool': {'lat': 53.4066, 'lon': -2.9665, 'type': 'university'},
    '利物浦大学': {'lat': 53.4066, 'lon': -2.9665, 'type': 'university'},
    'opera house': {'lat': -33.8568, 'lon': 151.2153, 'type': 'landmark'},
    'university of technology sydney': {'lat': -33.8833, 'lon': 151.2006, 'type': 'university'},
    'manchester airport': {'lat': 53.3537, 'lon': -2.2750, 'type': 'airport', 'iata': 'MAN'},
    '曼彻斯特机场': {'lat': 53.3537, 'lon': -2.2750, 'type': 'airport', 'iata': 'MAN'},
    'manchester': {'lat': 53.4808, 'lon': -2.2426, 'type': 'city'},
    '曼彻斯特': {'lat': 53.4808, 'lon': -2.2426, 'type': 'city'},
    'sydney central station': {'lat': -33.8833, 'lon': 151.2065, 'type': 'station'},
    'sydney airport': {'lat': -33.9399, 'lon': 151.1753, 'type': 'airport', 'iata': 'SYD'},
    'sydney kingsford smith airport': {'lat': -33.9399, 'lon': 151.1753, 'type': 'airport', 'iata': 'SYD'},
    'chengdu': {'lat': 30.5728, 'lon': 104.0668, 'type': 'city'},
    'chengdu tianfu airport': {'lat': 30.3190, 'lon': 104.4410, 'type': 'airport', 'iata': 'TFU'},
    'chengdu tianfu international airport': {'lat': 30.3190, 'lon': 104.4410, 'type': 'airport', 'iata': 'TFU'},
    'chengdu airport': {'lat': 30.5784, 'lon': 103.9471, 'type': 'airport', 'iata': 'CTU'},
    'chengdu shuangliu international airport': {'lat': 30.5784, 'lon': 103.9471, 'type': 'airport', 'iata': 'CTU'},
    'jiuzhaigou': {'lat': 33.2600, 'lon': 103.9190, 'type': 'landmark'},
    'jiuzhaigou national park': {'lat': 33.2600, 'lon': 103.9190, 'type': 'landmark'},
    '九寨沟': {'lat': 33.2600, 'lon': 103.9190, 'type': 'landmark'},
    'jiuzhai huanglong airport': {'lat': 32.8533, 'lon': 103.6822, 'type': 'airport', 'iata': 'JZH'},
    '四川九寨沟': {'lat': 33.2600, 'lon': 103.9190, 'type': 'landmark'},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Geocoder:
  def __init__(self, cache_dir: str = 'data/cache', use_online: bool = True,
               nominatim_url: str = 'https://nominatim.openstreetmap.org/search'):
    self.cache = DiskCache(cache_dir)
    self.use_online = use_online
    self.nominatim_url = nominatim_url
    self._last_request = 0.0

  def _rate_limit(self) -> None:
    elapsed = time.time() - self._last_request
    if elapsed < 1.0:
      time.sleep(1.0 - elapsed)
    self._last_request = time.time()

  def _offline_lookup(self, query: str) -> Optional[Waypoint]:
    key = query.lower().strip()
    if key in OFFLINE_PLACES:
      p = OFFLINE_PLACES[key]
      return Waypoint(
        name=query,
        lat=p['lat'],
        lon=p['lon'],
        place_type=p.get('type', 'city'),
        iata=p.get('iata'),
        port_code=p.get('port_code'),
      )
    # fuzzy: substring match
    for place_key, p in OFFLINE_PLACES.items():
      if place_key in key or key in place_key:
        return Waypoint(
          name=query,
          lat=p['lat'],
          lon=p['lon'],
          place_type=p.get('type', 'city'),
          iata=p.get('iata'),
          port_code=p.get('port_code'),
        )
    return None

  def geocode(self, query: str) -> Optional[Waypoint]:
    cache_key = query.lower().strip()
    cached = self.cache.get('geocode', cache_key)
    if cached:
      return Waypoint(**cached)

    waypoint = self._offline_lookup(query)
    if waypoint:
      self.cache.set('geocode', cache_key, waypoint.__dict__)
      return waypoint

    if not self.use_online:
      return None

    self._rate_limit()
    try:
      resp = requests.get(
        self.nominatim_url,
        params={'q': query, 'format': 'json', 'limit': 1},
        headers={'User-Agent': 'GLOBALNAV/1.0 (research navigation agent)'},
        timeout=10,
      )
      resp.raise_for_status()
      results = resp.json()
      if not results:
        return None
      r = results[0]
      waypoint = Waypoint(
        name=r.get('display_name', query),
        lat=float(r['lat']),
        lon=float(r['lon']),
        place_type=r.get('type', 'place'),
      )
      self.cache.set('geocode', cache_key, waypoint.__dict__)
      return waypoint
    except (requests.RequestException, KeyError, ValueError):
      return self._offline_lookup(query)

  def search_candidates(self, query: str, limit: int = 5) -> list[Waypoint]:
    """Return multiple geocode candidates for disambiguation."""
    q = query.lower().strip()
    candidates = []
    for key, p in OFFLINE_PLACES.items():
      if q in key or key in q:
        candidates.append(Waypoint(
          name=key.title(),
          lat=p['lat'],
          lon=p['lon'],
          place_type=p.get('type', 'city'),
          iata=p.get('iata'),
          port_code=p.get('port_code'),
        ))
    if candidates:
      return candidates[:limit]

    if not self.use_online:
      return []

    self._rate_limit()
    try:
      resp = requests.get(
        self.nominatim_url,
        params={'q': query, 'format': 'json', 'limit': limit},
        headers={'User-Agent': 'GLOBALNAV/1.0 (research navigation agent)'},
        timeout=10,
      )
      resp.raise_for_status()
      return [
        Waypoint(
          name=r.get('display_name', query),
          lat=float(r['lat']),
          lon=float(r['lon']),
          place_type=r.get('type', 'place'),
        )
        for r in resp.json()
      ]
    except (requests.RequestException, KeyError, ValueError):
      return []
