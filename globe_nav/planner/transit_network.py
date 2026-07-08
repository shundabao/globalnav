"""Build a transit graph from OpenStreetMap stops and route relations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from globe_nav.maps.cache import DiskCache
from globe_nav.maps.geocoder import haversine_km
from globe_nav.planner.transit import TransitStop

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

ROUTE_MODES = {
    'bus': 'bus', 'trolleybus': 'bus', 'share_taxi': 'bus',
    'train': 'train', 'subway': 'train', 'light_rail': 'train',
    'monorail': 'train', 'tram': 'tram',
}


@dataclass
class TransitEdge:
    from_stop: TransitStop
    to_stop: TransitStop
    mode: str
    route_name: str
    distance_km: float
    duration_min: float


@dataclass
class TransitNetwork:
    stops: dict[str, TransitStop] = field(default_factory=dict)
    edges: list[TransitEdge] = field(default_factory=list)

    def add_stop(self, stop: TransitStop) -> str:
        sid = f'{stop.lat:.5f},{stop.lon:.5f}'
        if sid not in self.stops:
            self.stops[sid] = stop
        return sid

    def add_edge(self, edge: TransitEdge) -> None:
        self.add_stop(edge.from_stop)
        self.add_stop(edge.to_stop)
        self.edges.append(edge)


class TransitNetworkBuilder:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache = DiskCache(cache_dir)
        self._last_request = 0.0

    def build_corridor_network(self, lat1, lon1, lat2, lon2, padding_km=3.0) -> TransitNetwork:
        south, west, north, east = self._bbox(lat1, lon1, lat2, lon2, padding_km)
        cache_key = f'{south:.3f},{west:.3f},{north:.3f},{east:.3f}'
        cached = self.cache.get('transit_net', cache_key)
        if cached and (cached.get('stops') or cached.get('edges')):
            return self._network_from_cache(cached)
        stops = self._fetch_stops(south, west, north, east)
        route_edges = self._fetch_route_edges(south, west, north, east, stops)
        network = TransitNetwork()
        for stop in stops:
            network.add_stop(stop)
        for edge in route_edges:
            network.add_edge(edge)
        self.cache.set('transit_net', cache_key, {
            'stops': [s.__dict__ for s in network.stops.values()],
            'edges': [
                {'from': e.from_stop.__dict__, 'to': e.to_stop.__dict__, 'mode': e.mode,
                 'route_name': e.route_name, 'distance_km': e.distance_km, 'duration_min': e.duration_min}
                for e in network.edges
            ],
        })
        return network

    def nearby_stops(self, lat, lon, radius_m=2000, limit=12):
        s, w, n, e = self._bbox(lat, lon, lat, lon, radius_m / 1000.0)
        stops = self._fetch_stops(s, w, n, e)
        stops.sort(key=lambda x: haversine_km(lat, lon, x.lat, x.lon))
        return stops[:limit]

    @staticmethod
    def _bbox(lat1, lon1, lat2, lon2, pad_km):
        pad = pad_km / 111.0
        return min(lat1, lat2) - pad, min(lon1, lon2) - pad, max(lat1, lat2) + pad, max(lon1, lon2) + pad

    def _fetch_stops(self, south, west, north, east):
        query = f'''[out:json][timeout:40];(
          node({south},{west},{north},{east})["highway"="bus_stop"];
          node({south},{west},{north},{east})["public_transport"="platform"];
          node({south},{west},{north},{east})["public_transport"="stop_position"];
          node({south},{west},{north},{east})["railway"~"station|halt|tram_stop"];
        );out body 120;'''
        data = self._overpass(query)
        stops, seen = [], set()
        for el in data.get('elements', []):
            if el.get('type') != 'node':
                continue
            tags = el.get('tags', {})
            sid = f"{el['lat']:.5f},{el['lon']:.5f}"
            if sid in seen:
                continue
            seen.add(sid)
            stops.append(TransitStop(
                name=tags.get('name') or tags.get('ref') or 'Transit stop',
                lat=el['lat'], lon=el['lon'], stop_type=self._stop_type(tags),
            ))
        return stops

    def _fetch_route_edges(self, south, west, north, east, stops):
        stop_index = {f'{s.lat:.5f},{s.lon:.5f}': s for s in stops}
        query = f'''[out:json][timeout:45];
        relation({south},{west},{north},{east})["route"~"bus|trolleybus|train|tram|subway|light_rail|monorail"];
        out body;>;out skel qt;'''
        data = self._overpass(query)
        nodes = {}
        for el in data.get('elements', []):
            if el.get('type') == 'node':
                tags = el.get('tags') or {}
                nodes[el['id']] = (el['lat'], el['lon'], tags.get('name') or tags.get('ref') or 'Transit stop')
        edges, seen = [], set()
        for el in data.get('elements', []):
            if el.get('type') != 'relation':
                continue
            tags = el.get('tags', {})
            mode = ROUTE_MODES.get(tags.get('route', ''))
            if not mode:
                continue
            route_name = tags.get('name') or tags.get('ref') or tags.get('route', 'route')
            prev = None
            for member in el.get('members', []):
                if member.get('type') != 'node':
                    continue
                nid = member.get('ref')
                if nid not in nodes:
                    continue
                lat, lon, name = nodes[nid]
                sid = f'{lat:.5f},{lon:.5f}'
                stop = stop_index.get(sid) or TransitStop(name=name, lat=lat, lon=lon, stop_type=mode)
                stop_index[sid] = stop
                if prev is not None and prev is not stop:
                    dist = haversine_km(prev.lat, prev.lon, stop.lat, stop.lon) * 1.15
                    if dist >= 0.05:
                        key = f'{prev.lat:.5f},{prev.lon:.5f}->{stop.lat:.5f},{stop.lon:.5f}:{mode}'
                        if key not in seen:
                            seen.add(key)
                            edges.append(TransitEdge(prev, stop, mode, route_name, dist, max(dist / {'bus':22,'train':55,'tram':28}.get(mode,30)*60, 2.0)))
                prev = stop
        return edges

    @staticmethod
    def _stop_type(tags):
        if tags.get('railway') == 'tram_stop':
            return 'tram'
        if tags.get('railway') in ('station', 'halt'):
            return 'train'
        if tags.get('highway') == 'bus_stop':
            return 'bus'
        return 'train'

    def _overpass(self, query):
        elapsed = time.time() - self._last_request
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_request = time.time()
        try:
            resp = requests.post(OVERPASS_URL, data={'data': query}, headers={'User-Agent': 'GLOBALNAV/1.0'}, timeout=50)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return {'elements': []}

    @staticmethod
    def _network_from_cache(data):
        network = TransitNetwork()
        for raw in data.get('stops', []):
            network.add_stop(TransitStop(**raw))
        for raw in data.get('edges', []):
            network.add_edge(TransitEdge(
                TransitStop(**raw['from']), TransitStop(**raw['to']),
                raw['mode'], raw['route_name'], raw['distance_km'], raw['duration_min'],
            ))
        return network
