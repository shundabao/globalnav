"""Build modular trip segments with per-segment transport options."""

from __future__ import annotations

import hashlib
from typing import Optional

from globe_nav.env.transport import Waypoint
from globe_nav.maps.geocoder import Geocoder, haversine_km
from globe_nav.planner.environment import MobilityEnvironment
from globe_nav.planner.models import DetailedLeg
from globe_nav.planner.recommender import SegmentRecommender
from globe_nav.planner.segment_models import MicroLeg, ModularTripPlan, SegmentOption, TripSegment


class ModularTripBuilder:
    def __init__(self, env: Optional[MobilityEnvironment] = None):
        self.env = env or MobilityEnvironment()

    def build(self, origin_name: str, dest_name: str, instruction: str = '') -> ModularTripPlan:
        origin = self._geocode_resilient(origin_name)
        dest = self._geocode_resilient(dest_name)
        if not origin:
            raise ValueError(f'Could not geocode origin: {origin_name!r}')
        if not dest:
            raise ValueError(f'Could not geocode destination: {dest_name!r}')

        dist = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)
        if dist < 80:
            segments = [self._local_segment('seg_1', origin, dest, f'{origin.name} → {dest.name}')]
        else:
            segments = self._long_haul_segments(origin, dest, instruction=instruction)

        plan = ModularTripPlan(
            origin=origin.name,
            destination=dest.name,
            segments=segments,
            instruction=instruction,
        )
        SegmentRecommender().apply_defaults(plan)
        return plan

    def _geocode_resilient(self, query: str) -> Optional[Waypoint]:
        wp = self.env.geocode(query)
        if wp:
            return wp
        if not self.env.geocoder.use_online:
            self.env.geocoder.use_online = True
            wp = self.env.geocode(query)
            self.env.geocoder.use_online = False
        return wp

    def _long_haul_segments(
        self, origin: Waypoint, dest: Waypoint, instruction: str = '',
    ) -> list[TripSegment]:
        self.env.flights.ensure_loaded()
        dep_airports = self.env.flights.nearest_commercial_airports(
            origin.lat, origin.lon, n=3, max_km=80,
        )
        arr_airports = self.env.flights.nearest_commercial_airports(
            dest.lat, dest.lon, n=3, max_km=250,
        )
        dep_airports = self._prefer_departure_airports(origin, instruction, dep_airports)
        arr_airports = self._prefer_airports_from_instruction(instruction, arr_airports)
        if not dep_airports:
            return [self._local_segment('seg_1', origin, dest, 'Full route (no available departure airport)')]
        if not arr_airports:
            return [self._local_segment('seg_1', origin, dest, 'Full route (no available arrival airport)')]

        dep_ap, arr_ap = self._best_airport_pair(dep_airports, arr_airports)
        dep_label = f'{dep_ap.name} ({dep_ap.iata})'
        arr_label = f'{arr_ap.name} ({arr_ap.iata})'

        access = self._local_segment(
            'seg_access',
            origin,
            Waypoint(dep_label, dep_ap.lat, dep_ap.lon, 'airport', dep_ap.iata),
            f'{origin.name} → {dep_label}',
        )
        flight = self._flight_segment_multi(dep_ap.iata, arr_airports, dep_label)
        flight.title = f'{dep_label} → {arr_label}'
        flight.to_name = arr_label

        egress = self._local_segment(
            'seg_egress',
            Waypoint(arr_label, arr_ap.lat, arr_ap.lon, 'airport', arr_ap.iata),
            dest,
            f'{arr_label} → {dest.name}',
        )
        return [access, flight, egress]

    def _prefer_airports_from_instruction(self, instruction: str, airports: list):
        if not instruction:
            return airports
        text = instruction.lower()
        prefs = []
        for token, iata in (
            ('manchester', 'MAN'), ('曼彻斯特', 'MAN'),
            ('heathrow', 'LHR'), ('dubai', 'DXB'), ('迪拜', 'DXB'),
        ):
            if token in text:
                prefs.append(iata)
        if not prefs:
            return airports
        self.env.flights.ensure_loaded()
        ordered = []
        seen = set()
        for iata in prefs:
            ap = self.env.flights.airports.get(iata)
            if ap and ap.iata not in seen:
                ordered.append(ap)
                seen.add(ap.iata)
        for ap in airports:
            if ap.iata not in seen:
                ordered.append(ap)
                seen.add(ap.iata)
        return ordered or airports

    def _prefer_departure_airports(self, origin: Waypoint, instruction: str, airports: list):
        text = f'{origin.name} {instruction}'.lower()
        prefs = []
        if 'jfk' in text or 'john f kennedy' in text or 'new york' in text:
            prefs.append('JFK')
        if not prefs:
            return airports
        self.env.flights.ensure_loaded()
        ordered = []
        seen = set()
        for iata in prefs:
            ap = self.env.flights.airports.get(iata)
            if ap and ap.iata not in seen and any(candidate.iata == ap.iata for candidate in airports):
                ordered.append(ap)
                seen.add(ap.iata)
        for ap in airports:
            if ap.iata not in seen:
                ordered.append(ap)
                seen.add(ap.iata)
        return ordered or airports

    def _best_airport_pair(self, dep_list, arr_list):
        best = None
        for dep in dep_list[:2]:
            for arr in arr_list[:4]:
                paths = self.env.flights.find_paths(dep.iata, arr.iata, max_stops=2)
                if not paths:
                    paths = self.env.flights.estimate_via_hubs(dep.iata, arr.iata)
                if paths:
                    score = (len(paths[0]) - 1, self.env.flights._path_distance(paths[0]))
                    if best is None or score < best[0]:
                        best = (score, dep, arr)
        if best:
            return best[1], best[2]
        return dep_list[0], arr_list[0]

    def _flight_segment_multi(
        self, dep_iata: str, arr_airports: list, dep_label: str,
    ) -> TripSegment:
        options: list[SegmentOption] = []
        seen: set[str] = set()

        for arr_ap in arr_airports:
            arr_label = f'{arr_ap.name} ({arr_ap.iata})'
            paths = self.env.flights.find_paths(dep_iata, arr_ap.iata, max_stops=2)
            estimated = False
            if not paths:
                paths = self.env.flights.estimate_via_hubs(dep_iata, arr_ap.iata)
                estimated = True

            for i, path in enumerate(paths[:4]):
                chain_key = '→'.join(path)
                if chain_key in seen:
                    continue
                seen.add(chain_key)
                legs = self.env.flights.build_flight_legs(path, estimated=estimated)
                stops = 'direct' if len(path) == 2 else f'{len(path) - 2} stop(s)'
                dur = sum(l.duration_min for l in legs)
                dist = sum(l.distance_km for l in legs)
                tag = ' (estimated)' if estimated else ''
                options.append(SegmentOption(
                    option_id=f'seg_flight_{arr_ap.iata}_{i}',
                    label=f'Flight {" → ".join(path)} ({stops}){tag}',
                    mode_chain='fly',
                    micro_legs=[MicroLeg.from_detailed(l) for l in legs],
                    duration_min=dur,
                    distance_km=dist,
                    verified=not estimated,
                    tooltip=f'{stops} · {dur/60:.1f}h · arrives at {arr_ap.iata}',
                ))

        if not options:
            options.append(SegmentOption(
                option_id='seg_flight_estimated',
                label=f'Estimated flight {dep_iata} → airport near destination',
                mode_chain='fly',
                micro_legs=[MicroLeg(
                    mode='fly',
                    from_name=dep_label,
                    to_name='Destination region airport',
                    distance_km=0, duration_min=0,
                    steps=['No route in database — try different destination airport'],
                    note='estimated',
                )],
                duration_min=0,
                distance_km=0,
                verified=False,
                tooltip='No route found; check the destination',
            ))

        options.sort(key=lambda o: o.duration_min if o.duration_min > 0 else 1e9)
        return TripSegment(
            segment_id='seg_flight',
            title=f'{dep_label} → …',
            segment_type='flight',
            from_name=dep_label,
            to_name='…',
            options=options,
            description='Flight segment: OpenFlights connectivity plus hub-estimated fallbacks',
        )

    def _local_segment(self, seg_id: str, a: Waypoint, b: Waypoint, title: str) -> TripSegment:
        raw_options = self.env.get_local_segment_options(a, b)
        options = []
        seen_paths: set[str] = set()
        for legs in raw_options:
            chain = '+'.join(l.mode for l in legs)
            signature = '|'.join(
                f'{l.mode}:{l.from_name}->{l.to_name}:'
                f'{l.from_lat:.5f},{l.from_lon:.5f}->{l.to_lat:.5f},{l.to_lon:.5f}'
                for l in legs
            )
            if signature in seen_paths:
                continue
            seen_paths.add(signature)
            opt_id = f'{seg_id}_{hashlib.md5(signature.encode()).hexdigest()[:8]}'
            dur = max(sum(l.duration_min for l in legs), 1.0)
            dist = sum(l.distance_km for l in legs)
            options.append(SegmentOption(
                option_id=opt_id,
                label=self._option_label(legs),
                mode_chain=chain.replace('+', ' → '),
                micro_legs=[MicroLeg.from_detailed(l) for l in legs],
                duration_min=dur,
                distance_km=dist,
                verified=all(l.verified for l in legs),
                tooltip=self._tooltip(legs, dur, dist),
            ))
        options.sort(key=lambda o: o.duration_min)
        return TripSegment(
            segment_id=seg_id,
            title=title,
            segment_type='local',
            from_name=a.name,
            to_name=b.name,
            options=options,
            description='1 drive option plus top-5 transit/walk options (OSM routes + OSRM graph search)',
        )

    @staticmethod
    def _option_label(legs: list[DetailedLeg]) -> str:
        names = {
            'walk': 'Walk',
            'drive': 'Drive',
            'bus': 'Bus',
            'train': 'Train',
            'tram': 'Tram',
        }
        if len(legs) == 1:
            return names.get(legs[0].mode, legs[0].mode.title())
        return ' → '.join(names.get(l.mode, l.mode) for l in legs)

    @staticmethod
    def _tooltip(legs: list[DetailedLeg], dur: float, dist: float) -> str:
        from globe_nav.planner.segment_models import format_duration
        parts = [f'{format_duration(dur)} · {dist:.1f} km']
        for leg in legs:
            parts.append(f'{leg.mode}: {leg.from_name} → {leg.to_name}')
        return ' | '.join(parts)
