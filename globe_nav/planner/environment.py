"""Mobility environment: graph-based multi-modal route enumeration."""

from __future__ import annotations

import itertools
from typing import Optional

from globe_nav.env.transport import Waypoint
from globe_nav.maps.geocoder import Geocoder, haversine_km
from globe_nav.maps.geo import line_interpolate
from globe_nav.planner.flights import FlightGraph
from globe_nav.planner.models import DetailedLeg, RouteOption, RouteStep
from globe_nav.planner.multimodal_router import MultimodalRouter
from globe_nav.planner.osrm import OSRMClient
from globe_nav.planner.transit import TransitFinder


class MobilityEnvironment:
    """Enumerate feasible routes: 1 drive + top-k walk/transit combos per local segment."""

    GROUND_MODES = ('walk', 'drive')
    MAX_OPTIONS = 12
    LOCAL_ACTIVE_TOP_K = 5

    def __init__(self, cache_dir: str = 'data/cache', use_online: bool = True):
        self.geocoder = Geocoder(cache_dir, use_online=use_online)
        self.osrm = OSRMClient(cache_dir)
        self.flights = FlightGraph()
        self.transit = TransitFinder(cache_dir)
        self.router = MultimodalRouter(self.osrm, cache_dir)

    def geocode(self, query: str) -> Optional[Waypoint]:
        return self.geocoder.geocode(query)

    def get_local_segment_options(self, from_wp: Waypoint, to_wp: Waypoint) -> list[list[DetailedLeg]]:
        """1 best drive + top LOCAL_ACTIVE_TOP_K non-drive paths from graph search."""
        return self.router.find_top_options(from_wp, to_wp, active_k=self.LOCAL_ACTIVE_TOP_K)

    def plan_all_options(self, origin_name: str, dest_name: str) -> list[RouteOption]:
        origin = self.geocode(origin_name)
        dest = self.geocode(dest_name)
        if not origin or not dest:
            raise ValueError(f'Geocode failed: origin={origin_name!r} dest={dest_name!r}')

        dist = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)
        if dist < 80:
            options = self._local_trip_options(origin, dest)
        else:
            options = self._long_trip_options(origin, dest)

        if not options:
            options.append(self._fallback_option(origin, dest))
        options.sort(key=lambda o: o.total_duration_min)
        for i, opt in enumerate(options[:self.MAX_OPTIONS]):
            opt.option_id = f'option_{i + 1}'
        return options[:self.MAX_OPTIONS]

    def _local_trip_options(self, origin: Waypoint, dest: Waypoint) -> list[RouteOption]:
        options = []
        for legs in self.router.find_top_options(origin, dest, active_k=self.LOCAL_ACTIVE_TOP_K):
            options.append(RouteOption(option_id='', legs=legs, description=' → '.join(l.mode for l in legs)))
        return options

    def _long_trip_options(self, origin: Waypoint, dest: Waypoint) -> list[RouteOption]:
        options = []
        self.flights.ensure_loaded()
        dep_airports = self.flights.nearest_airports(origin.lat, origin.lon, n=2, max_km=80)
        arr_airports = self.flights.nearest_airports(dest.lat, dest.lon, n=2, max_km=120)
        if not dep_airports or not arr_airports:
            return [self._fallback_option(origin, dest)]

        for dep_ap, arr_ap in itertools.product(dep_airports, arr_airports):
            paths = self.flights.find_paths(dep_ap.iata, arr_ap.iata, max_stops=1)
            if not paths:
                continue
            access_opts = self.get_local_segment_options(
                origin, Waypoint(f'{dep_ap.name} ({dep_ap.iata})', dep_ap.lat, dep_ap.lon, 'airport', dep_ap.iata)
            )
            egress_opts = self.get_local_segment_options(
                Waypoint(arr_ap.name, arr_ap.lat, arr_ap.lon, 'airport', arr_ap.iata),
                Waypoint(dest.name, dest.lat, dest.lon, dest.place_type, dest.short_name),
            )
            for path in paths[:3]:
                flight_legs = self.flights.build_flight_legs(path)
                stops = 'direct' if len(path) == 2 else '1-stop'
                for access in access_opts:
                    for egress in egress_opts:
                        legs = access + flight_legs + egress
                        options.append(RouteOption(
                            option_id='', legs=legs,
                            description=f'{" + ".join(l.mode for l in legs)} ({stops}: {dep_ap.iata}→{arr_ap.iata})',
                        ))
        if not options:
            fb = self._fallback_option(origin, dest)
            fb.description += ' (no OpenFlights route)'
            options.append(fb)
        return options

    def _fallback_option(self, origin: Waypoint, dest: Waypoint) -> RouteOption:
        leg = self.osrm.route(origin.lat, origin.lon, origin.name, dest.lat, dest.lon, dest.name, 'drive')
        if not leg:
            from globe_nav.env.transport import TransportMode
            dist = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon) * 1.3
            tm = TransportMode.from_str('drive')
            leg = DetailedLeg(
                mode='drive', from_name=origin.name, to_name=dest.name,
                from_lat=origin.lat, from_lon=origin.lon, to_lat=dest.lat, to_lon=dest.lon,
                distance_km=dist, duration_min=dist / tm.speed_kmh * 60,
                verified=False, note='Estimated fallback',
            )
        return RouteOption(option_id='option_fallback', legs=[leg], description='drive (fallback)')
