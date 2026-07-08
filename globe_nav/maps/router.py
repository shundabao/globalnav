"""Multi-modal global routing: walk, drive, bus, train, fly, maritime."""

import math
from typing import Optional

from globe_nav.env.transport import RouteSegment, TransportMode, Waypoint
from globe_nav.maps.cache import DiskCache
from globe_nav.maps.geocoder import Geocoder, haversine_km

# Major shipping lanes (simplified maritime corridors)
MARITIME_CHANNELS = [
    ('Singapore Strait', 1.2644, 103.8220),
    ('Malacca Strait', 2.5, 101.5),
    ('Suez Canal', 30.0444, 32.3482),
    ('Panama Canal', 9.0800, -79.6800),
    ('English Channel', 50.5, 1.0),
    ('Strait of Gibraltar', 35.9, -5.5),
]

# Hub airports for multi-leg flight planning
FLIGHT_HUBS = [
    Waypoint('Dubai', 25.2048, 55.2708, 'airport', 'DXB'),
    Waypoint('Singapore', 1.3644, 103.9915, 'airport', 'SIN'),
    Waypoint('London Heathrow', 51.4700, -0.4543, 'airport', 'LHR'),
    Waypoint('Tokyo Narita', 35.7720, 140.3929, 'airport', 'NRT'),
]


def _interpolate_great_circle(a: Waypoint, b: Waypoint, n: int = 20) -> list[tuple[float, float]]:
    """Sample points along a great-circle arc."""
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    ))
    if d < 1e-9:
        return [(a.lat, a.lon), (b.lat, b.lon)]

    points = []
    for i in range(n + 1):
        f = i / n
        a_interp = math.sin((1 - f) * d) / math.sin(d)
        b_interp = math.sin(f * d) / math.sin(d)
        x = a_interp * math.cos(lat1) * math.cos(lon1) + b_interp * math.cos(lat2) * math.cos(lon2)
        y = a_interp * math.cos(lat1) * math.sin(lon1) + b_interp * math.cos(lat2) * math.sin(lon2)
        z = a_interp * math.sin(lat1) + b_interp * math.sin(lat2)
        points.append((math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
                       math.degrees(math.atan2(y, x))))
    return points


def _nearest_hub(waypoint: Waypoint) -> Waypoint:
    return min(FLIGHT_HUBS, key=lambda h: haversine_km(waypoint.lat, waypoint.lon, h.lat, h.lon))


class MultiModalRouter:
    def __init__(self, cache_dir: str = 'data/cache', use_online: bool = False):
        self.cache = DiskCache(cache_dir)
        self.geocoder = Geocoder(cache_dir, use_online=use_online)
        self.use_online = use_online

    def route(self, origin: Waypoint, destination: Waypoint,
              mode: TransportMode) -> RouteSegment:
        cache_key = f'{origin.lat},{origin.lon}->{destination.lat},{destination.lon}:{mode.value}'
        cached = self.cache.get('route', cache_key)
        if cached:
            return RouteSegment(
                mode=TransportMode(cached['mode']),
                from_waypoint=Waypoint(**cached['from_waypoint']),
                to_waypoint=Waypoint(**cached['to_waypoint']),
                distance_km=cached['distance_km'],
                duration_hours=cached['duration_hours'],
                instructions=cached['instructions'],
                geometry=[tuple(p) for p in cached['geometry']],
            )

        if mode == TransportMode.FLY:
            segment = self._route_fly(origin, destination)
        elif mode == TransportMode.MARITIME:
            segment = self._route_maritime(origin, destination)
        elif mode == TransportMode.TRAIN:
            segment = self._route_rail(origin, destination, mode)
        elif mode in (TransportMode.DRIVE, TransportMode.WALK, TransportMode.BUS):
            segment = self._route_ground(origin, destination, mode)
        else:
            segment = self._route_ground(origin, destination, mode)

        self.cache.set('route', cache_key, {
            'mode': segment.mode.value,
            'from_waypoint': segment.from_waypoint.__dict__,
            'to_waypoint': segment.to_waypoint.__dict__,
            'distance_km': segment.distance_km,
            'duration_hours': segment.duration_hours,
            'instructions': segment.instructions,
            'geometry': segment.geometry,
        })
        return segment

    def _route_ground(self, origin: Waypoint, destination: Waypoint,
                      mode: TransportMode) -> RouteSegment:
        dist = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
        factors = {
            TransportMode.DRIVE: 1.35,
            TransportMode.BUS: 1.45,
            TransportMode.WALK: 1.15,
        }
        distance = dist * factors.get(mode, 1.2)
        duration = distance / mode.speed_kmh

        geometry = _interpolate_great_circle(origin, destination, n=30)
        verbs = {
            TransportMode.DRIVE: 'Drive',
            TransportMode.BUS: 'Take bus',
            TransportMode.WALK: 'Walk',
        }
        verb = verbs.get(mode, 'Travel')
        instructions = [
            f'{verb} from {origin.name} toward {destination.name}',
            f'Continue for approximately {distance:.0f} km',
            f'Arrive at {destination.name}',
        ]
        return RouteSegment(mode, origin, destination, distance, duration, instructions, geometry)

    def _route_rail(self, origin: Waypoint, destination: Waypoint,
                    mode: TransportMode) -> RouteSegment:
        dist = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
        distance = dist * 1.2  # rail network factor
        duration = distance / mode.speed_kmh

        geometry = _interpolate_great_circle(origin, destination, n=30)
        instructions = [
            f'Board train at {origin.name}',
            f'Ride toward {destination.name} (~{distance:.0f} km)',
            f'Arrive and alight at {destination.name}',
        ]
        return RouteSegment(mode, origin, destination, distance, duration, instructions, geometry)

    def _route_fly(self, origin: Waypoint, destination: Waypoint) -> RouteSegment:
        dist = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
        # Long-haul may route via hub
        via_hub = None
        if dist > 3000:
            hub = _nearest_hub(origin)
            hub2 = _nearest_hub(destination)
            if hub.name != hub2.name:
                via_hub = hub2

        instructions = [
            f'Proceed to departure airport near {origin.name}',
            f'Takeoff and climb to cruise altitude',
        ]
        geometry = _interpolate_great_circle(origin, destination, n=40)

        if via_hub:
            instructions.insert(2, f'Connect via {via_hub.name} ({via_hub.iata})')
            leg1 = haversine_km(origin.lat, origin.lon, via_hub.lat, via_hub.lon)
            leg2 = haversine_km(via_hub.lat, via_hub.lon, destination.lat, destination.lon)
            dist = leg1 + leg2
            geometry = (
                _interpolate_great_circle(origin, via_hub, n=15)
                + _interpolate_great_circle(via_hub, destination, n=15)[1:]
            )

        duration = dist / TransportMode.FLY.speed_kmh + 1.5  # taxi + layover buffer
        instructions.extend([
            f'Cruise for {dist:.0f} km',
            f'Descend and land at destination near {destination.name}',
        ])
        return RouteSegment(
            TransportMode.FLY, origin, destination, dist, duration, instructions, geometry
        )

    def _route_maritime(self, origin: Waypoint, destination: Waypoint) -> RouteSegment:
        dist = haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
        # Maritime routes are longer due to channels and coastal routing
        maritime_factor = 1.25
        distance = dist * maritime_factor

        channels_used = []
        mid_lat = (origin.lat + destination.lat) / 2
        mid_lon = (origin.lon + destination.lon) / 2
        for name, clat, clon in MARITIME_CHANNELS:
            if haversine_km(mid_lat, mid_lon, clat, clon) < 2000:
                channels_used.append(name)

        geometry = _interpolate_great_circle(origin, destination, n=35)
        instructions = [
            f'Depart port at {origin.name}',
            'Navigate to open sea following coastal traffic separation scheme',
        ]
        for ch in channels_used:
            instructions.append(f'Enter and transit {ch}')
        instructions.extend([
            f'Cruise {distance:.0f} nautical km equivalent',
            f'Dock at {destination.name}',
        ])

        duration = distance / TransportMode.MARITIME.speed_kmh
        return RouteSegment(
            TransportMode.MARITIME, origin, destination, distance, duration, instructions, geometry
        )

    def plan_multimodal(
        self,
        origin: Waypoint,
        destination: Waypoint,
        modes: list[TransportMode],
    ) -> list[RouteSegment]:
        """Chain segments when modes change (e.g., drive -> fly -> walk)."""
        if len(modes) == 1:
            return [self.route(origin, destination, modes[0])]

        segments = []
        # For demo: insert transfer hubs between mode changes
        transfer_points = self._get_transfer_points(origin, destination, modes)
        points = [origin] + transfer_points + [destination]

        for i, mode in enumerate(modes):
            seg = self.route(points[i], points[i + 1], mode)
            segments.append(seg)
        return segments

    def _get_transfer_points(
        self, origin: Waypoint, destination: Waypoint, modes: list[TransportMode]
    ) -> list[Waypoint]:
        """Generate transfer waypoints between transport mode changes."""
        transfers = []
        for i, mode in enumerate(modes[:-1]):
            next_mode = modes[i + 1]
            if mode == TransportMode.DRIVE and next_mode == TransportMode.FLY:
                hub = _nearest_hub(destination if i > 0 else origin)
                transfers.append(hub)
            elif mode == TransportMode.FLY and next_mode == TransportMode.DRIVE:
                hub = _nearest_hub(destination)
                transfers.append(hub)
            elif mode == TransportMode.MARITIME and next_mode in (TransportMode.DRIVE, TransportMode.WALK):
                port = Waypoint('Port of ' + destination.name.split(',')[0],
                                destination.lat, destination.lon, 'port')
                transfers.append(port)
            elif mode in (TransportMode.DRIVE, TransportMode.WALK) and next_mode == TransportMode.MARITIME:
                port = Waypoint('Port near ' + origin.name.split(',')[0],
                                origin.lat, origin.lon, 'port')
                transfers.append(port)
        return transfers

    def plan_from_legs(self, legs: list[dict]) -> list[RouteSegment]:
        """Build route from LLM-planned leg chain."""
        segments = []
        for leg in legs:
            from_q = leg.get('from', '')
            to_q = leg.get('to', '')
            mode = TransportMode.from_str(leg.get('mode', 'walk'))
            origin = self.resolve_place(from_q)
            dest = self.resolve_place(to_q)
            if not origin:
                raise ValueError(f'Could not geocode leg origin: {from_q}')
            if not dest:
                raise ValueError(f'Could not geocode leg destination: {to_q}')
            seg = self.route(origin, dest, mode)
            seg.description = leg.get('description', '')
            segments.append(seg)
        return segments

    def resolve_place(self, query: str) -> Optional[Waypoint]:
        return self.geocoder.geocode(query)
