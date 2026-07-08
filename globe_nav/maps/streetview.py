"""Global street imagery: Google Street View + optional Mapillary fallback."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

from globe_nav.config import get_google_maps_api_key, load_env

# Probe near Sydney Opera House — known urban GSV coverage
_PROBE_LAT, _PROBE_LON = -33.8568, 151.2153


class GlobalStreetViewProvider:
    """Fetch street-level imagery by geographic coordinate + heading."""

    METADATA_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
    STATIC_URL = 'https://maps.googleapis.com/maps/api/streetview'
    MAPILLARY_GRAPH = 'https://graph.mapillary.com/images'

    def __init__(
        self,
        cache_dir: str = 'data/cache/streetview',
        enabled: bool = True,
        size: str = '640x480',
        fov: int = 90,
    ):
        load_env()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.size = size
        self.fov = fov
        self.google_key = get_google_maps_api_key()
        self.mapillary_token = os.environ.get('MAPILLARY_ACCESS_TOKEN', '')
        self._probe_cache: Optional[dict] = None

    @property
    def configured(self) -> bool:
        return bool(self.google_key or self.mapillary_token)

    @property
    def available(self) -> bool:
        if not self.enabled or not self.configured:
            return False
        return self.probe().get('ok', False)

    def probe(self) -> dict:
        """Check whether imagery APIs actually work (cached per instance)."""
        if self._probe_cache is not None:
            return self._probe_cache

        if not self.enabled:
            self._probe_cache = {
                'ok': False, 'reason': 'disabled', 'message': '街景已关闭',
            }
            return self._probe_cache

        if not self.configured:
            self._probe_cache = {
                'ok': False, 'reason': 'no_key',
                'message': '未配置 GOOGLE_MAPS_API_KEY 或 MAPILLARY_ACCESS_TOKEN',
            }
            return self._probe_cache

        if self.google_key:
            meta = self._google_metadata(_PROBE_LAT, _PROBE_LON)
            status = meta.get('status', '')
            if status == 'OK':
                self._probe_cache = {'ok': True, 'source': 'google', 'message': 'Google Street View 可用'}
                return self._probe_cache
            if status in ('REQUEST_DENIED', 'ERROR'):
                data = self._google_image(_PROBE_LAT, _PROBE_LON, 0)
                if data:
                    self._probe_cache = {
                        'ok': True,
                        'source': 'google-static',
                        'message': 'Google Street View Static 图像接口可用',
                        'detail': meta.get('error_message', ''),
                    }
                    return self._probe_cache
            if status == 'REQUEST_DENIED':
                self._probe_cache = {
                    'ok': False, 'reason': 'api_not_enabled',
                    'message': '请在 GCP 启用 Street View Static API 并使用 GOOGLE_MAPS_API_KEY',
                    'detail': meta.get('error_message', ''),
                }
                return self._probe_cache

        if self.mapillary_token:
            m = self._mapillary_nearest(_PROBE_LAT, _PROBE_LON)
            if m:
                self._probe_cache = {'ok': True, 'source': 'mapillary', 'message': 'Mapillary 可用'}
                return self._probe_cache

        self._probe_cache = {
            'ok': False, 'reason': 'no_coverage',
            'message': '当前区域无街景覆盖或 API 不可用',
        }
        return self._probe_cache

    def status_dict(self) -> dict:
        p = self.probe()
        return {
            'enabled': self.enabled,
            'configured': self.configured,
            'working': p.get('ok', False),
            'reason': p.get('reason', ''),
            'message': p.get('message', ''),
            'source': p.get('source', ''),
        }

    def lookup(self, lat: float, lon: float, heading: float = 0.0) -> dict:
        if not self.enabled:
            return {'available': False, 'reason': 'disabled', 'message': '街景已关闭'}

        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return {'available': False, 'reason': 'no_position', 'message': '无坐标'}

        if self.google_key:
            meta = self._google_metadata(lat, lon)
            if meta.get('status') == 'OK':
                return {
                    'available': True,
                    'source': 'google',
                    'pano_id': meta.get('pano_id', ''),
                    'date': meta.get('date', ''),
                    'lat': meta.get('location', {}).get('lat', lat),
                    'lon': meta.get('location', {}).get('lng', lon),
                    'heading': round(heading, 1),
                    'message': 'Google Street View',
                }
            probe = self.probe()
            if meta.get('status') in ('REQUEST_DENIED', 'ERROR') and probe.get('source') in ('google', 'google-static'):
                return {
                    'available': True,
                    'source': 'google-static',
                    'lat': lat,
                    'lon': lon,
                    'heading': round(heading, 1),
                    'message': 'Google Street View Static',
                }

        probe = self.probe()
        if not probe.get('ok'):
            return {
                'available': False,
                'reason': probe.get('reason', 'unavailable'),
                'message': probe.get('message', ''),
            }

        if self.mapillary_token:
            m = self._mapillary_nearest(lat, lon)
            if m:
                return {
                    'available': True,
                    'source': 'mapillary',
                    'image_id': m['id'],
                    'heading': round(m.get('compass_angle', heading), 1),
                    'lat': m.get('lat', lat),
                    'lon': m.get('lon', lon),
                    'message': 'Mapillary',
                }

        return {'available': False, 'reason': 'no_coverage', 'message': '此位置无街景'}

    def image_bytes(self, lat: float, lon: float, heading: float = 0.0) -> Optional[bytes]:
        if not self.enabled:
            return None

        cache_key = f'{lat:.5f}_{lon:.5f}_{int(heading) % 360}'
        cache_path = self.cache_dir / f'{hashlib.md5(cache_key.encode()).hexdigest()}.jpg'
        if cache_path.is_file():
            return cache_path.read_bytes()

        if self.google_key:
            data = self._google_image(lat, lon, heading)
            if data:
                cache_path.write_bytes(data)
                return data

        if self.mapillary_token:
            m = self._mapillary_nearest(lat, lon)
            if m:
                data = self._mapillary_thumb(m['id'])
                if data:
                    cache_path.write_bytes(data)
                    return data
        return None

    def api_image_path(self, lat: float, lon: float, heading: float = 0.0) -> str:
        q = urlencode({
            'lat': f'{lat:.6f}',
            'lon': f'{lon:.6f}',
            'heading': f'{int(heading) % 360}',
        })
        return f'/api/streetview/image?{q}'

    def enrich_observation(self, obs: dict) -> dict:
        if obs.get('done'):
            obs['streetview'] = {'available': False, 'reason': 'done'}
            return obs

        lat, lon = obs.get('lat'), obs.get('lon')
        if not lat or not lon:
            obs['streetview'] = {'available': False, 'reason': 'no_position'}
            return obs

        heading = obs.get('heading_deg', 0.0)
        meta = self.lookup(lat, lon, heading)
        if meta.get('available'):
            meta['image_url'] = self.api_image_path(lat, lon, heading)
        else:
            meta['image_url'] = ''
        obs['streetview'] = meta
        return obs

    def _google_metadata(self, lat: float, lon: float) -> dict:
        try:
            resp = requests.get(
                self.METADATA_URL,
                params={'location': f'{lat},{lon}', 'key': self.google_key},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {'status': 'ERROR'}

    def _google_image(self, lat: float, lon: float, heading: float) -> Optional[bytes]:
        try:
            resp = requests.get(
                self.STATIC_URL,
                params={
                    'size': self.size,
                    'location': f'{lat},{lon}',
                    'heading': int(heading) % 360,
                    'fov': self.fov,
                    'key': self.google_key,
                },
                timeout=20,
            )
            if resp.status_code != 200:
                return None
            if 'image' not in resp.headers.get('content-type', ''):
                return None
            return resp.content
        except requests.RequestException:
            return None

    def _mapillary_nearest(self, lat: float, lon: float) -> Optional[dict]:
        try:
            resp = requests.get(
                self.MAPILLARY_GRAPH,
                params={
                    'access_token': self.mapillary_token,
                    'fields': 'id,computed_geometry,compass_angle',
                    'closeto': f'{lon},{lat}',
                    'limit': 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get('data', [])
            if not data:
                return None
            item = data[0]
            coords = item.get('computed_geometry', {}).get('coordinates', [lon, lat])
            return {
                'id': item['id'],
                'lon': coords[0],
                'lat': coords[1],
                'compass_angle': item.get('compass_angle', 0),
            }
        except (requests.RequestException, KeyError, IndexError):
            return None

    def _mapillary_thumb(self, image_id: str) -> Optional[bytes]:
        try:
            url = f'https://graph.mapillary.com/{image_id}/thumb'
            resp = requests.get(
                url,
                params={'access_token': self.mapillary_token},
                timeout=20,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
        except requests.RequestException:
            pass
        return None
