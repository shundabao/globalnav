"""Graph-based multi-modal route enumeration with categorized output."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional

from globe_nav.env.transport import Waypoint
from globe_nav.maps.geocoder import haversine_km
from globe_nav.maps.geo import line_interpolate
from globe_nav.planner.models import DetailedLeg, RouteStep
from globe_nav.planner.osrm import OSRMClient
from globe_nav.planner.transit_network import TransitNetworkBuilder

ORIGIN_ID = '__origin__'
DEST_ID = '__dest__'


@dataclass
class GraphEdge:
    to_id: str
    leg: DetailedLeg
    edge_key: str


@dataclass
class GraphNode:
    node_id: str
    name: str
    lat: float
    lon: float


@dataclass
class MultimodalGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    adj: dict[str, list[GraphEdge]] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self.adj.setdefault(node.node_id, [])


class MultimodalRouter:
    """Return 1 best drive route + top-k non-drive (walk/bus/train/tram) paths."""

    ACTIVE_TOP_K = 5
    MAX_LEGS = 8
    MAX_ACCESS_KM = 6.0
    MAX_TRANSFER_WALK_KM = 0.45
    TRANSFER_PENALTY_MIN = 2.5
    MAX_EXPLORE = 5000
    MAX_GRAPH_STOPS = 80

    def __init__(self, osrm: OSRMClient, cache_dir: str = 'data/cache'):
        self.osrm = osrm
        self.network_builder = TransitNetworkBuilder(cache_dir)

    def find_top_options(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        active_k: int = ACTIVE_TOP_K,
    ) -> list[list[DetailedLeg]]:
        """
        Categorized local options:
          1. best drive (OSRM, single leg)
          2. top ``active_k`` paths without any drive leg (walk / bus / train / tram combos)
        """
        graph = self._build_graph(from_wp, to_wp)
        options: list[list[DetailedLeg]] = []
        seen_options: set[tuple[str, ...]] = set()

        drive = self.osrm.route(
            from_wp.lat, from_wp.lon, from_wp.name,
            to_wp.lat, to_wp.lon, to_wp.name, 'drive',
        )
        if drive:
            options.append([drive])
            seen_options.add(self._option_signature([drive]))

        walk = self.osrm.route(
            from_wp.lat, from_wp.lon, from_wp.name,
            to_wp.lat, to_wp.lon, to_wp.name, 'walk',
        )

        active_paths = self._k_shortest_paths(
            graph, k=max(active_k * 3, active_k), allow_drive=False,
        )
        for path in active_paths:
            signature = self._option_signature(path)
            if signature in seen_options:
                continue
            seen_options.add(signature)
            options.append(path)
            active_count = sum(1 for p in options if not self._uses_drive(p))
            if active_count >= active_k:
                break

        if walk:
            walk_sig = self._option_signature([walk])
            has_walk = any(
                len(p) == 1 and p[0].mode == 'walk'
                for p in options
            )
            active_count = sum(1 for p in options if not self._uses_drive(p))
            if not has_walk and walk_sig not in seen_options:
                if active_count < active_k:
                    options.append([walk])
                else:
                    self._replace_slowest_active_with_walk(options, [walk])
                seen_options.add(walk_sig)

        drive_options = [p for p in options if self._uses_drive(p)]
        active_options = [p for p in options if not self._uses_drive(p)]
        active_options.sort(key=lambda legs: sum(l.duration_min for l in legs))
        options = drive_options[:1] + active_options[:active_k]

        if not options:
            fb = drive or self._synthetic_leg(from_wp, to_wp, 'drive')
            options.append([fb])
        return options

    @staticmethod
    def _replace_slowest_active_with_walk(
        options: list[list[DetailedLeg]],
        walk: list[DetailedLeg],
    ) -> None:
        candidates = [
            (i, sum(l.duration_min for l in path))
            for i, path in enumerate(options)
            if not any(l.mode == 'drive' for l in path)
        ]
        if not candidates:
            options.append(walk)
            return
        idx, _ = max(candidates, key=lambda item: item[1])
        options[idx] = walk

    @staticmethod
    def _uses_drive(legs: list[DetailedLeg]) -> bool:
        return any(l.mode == 'drive' for l in legs)

    def _build_graph(self, origin: Waypoint, dest: Waypoint) -> MultimodalGraph:
        graph = MultimodalGraph()
        graph.add_node(GraphNode(ORIGIN_ID, origin.name, origin.lat, origin.lon))
        graph.add_node(GraphNode(DEST_ID, dest.name, dest.lat, dest.lon))

        for mode in ('walk', 'drive'):
            leg = self.osrm.route(
                origin.lat, origin.lon, origin.name,
                dest.lat, dest.lon, dest.name, mode,
            )
            if leg:
                self._add_edge(graph, ORIGIN_ID, DEST_ID, leg, f'direct:{mode}')

        corridor = self.network_builder.build_corridor_network(
            origin.lat, origin.lon, dest.lat, dest.lon, 4.0,
        )
        stop_nodes: dict[str, GraphNode] = {}
        relevant_stops = sorted(
            corridor.stops.values(),
            key=lambda s: min(
                haversine_km(origin.lat, origin.lon, s.lat, s.lon),
                haversine_km(dest.lat, dest.lon, s.lat, s.lon),
            ),
        )[:self.MAX_GRAPH_STOPS]
        for stop in relevant_stops:
            sid = f'{stop.lat:.5f},{stop.lon:.5f}'
            stop_nodes[sid] = GraphNode(sid, stop.name, stop.lat, stop.lon)
            graph.add_node(stop_nodes[sid])
        for stop in self.network_builder.nearby_stops(origin.lat, origin.lon, 2500, 10):
            sid = f'{stop.lat:.5f},{stop.lon:.5f}'
            if sid not in stop_nodes:
                stop_nodes[sid] = GraphNode(sid, stop.name, stop.lat, stop.lon)
                graph.add_node(stop_nodes[sid])
        for stop in self.network_builder.nearby_stops(dest.lat, dest.lon, 2500, 10):
            sid = f'{stop.lat:.5f},{stop.lon:.5f}'
            if sid not in stop_nodes:
                stop_nodes[sid] = GraphNode(sid, stop.name, stop.lat, stop.lon)
                graph.add_node(stop_nodes[sid])

        for te in corridor.edges:
            sf = f'{te.from_stop.lat:.5f},{te.from_stop.lon:.5f}'
            st = f'{te.to_stop.lat:.5f},{te.to_stop.lon:.5f}'
            if sf not in stop_nodes or st not in stop_nodes:
                continue
            for sid, stop, name in (
                (sf, te.from_stop, te.from_stop.name),
                (st, te.to_stop, te.to_stop.name),
            ):
                if sid not in stop_nodes:
                    continue
            leg = DetailedLeg(
                mode=te.mode,
                from_name=te.from_stop.name,
                to_name=te.to_stop.name,
                from_lat=te.from_stop.lat,
                from_lon=te.from_stop.lon,
                to_lat=te.to_stop.lat,
                to_lon=te.to_stop.lon,
                distance_km=te.distance_km,
                duration_min=te.duration_min,
                steps=[RouteStep(
                    f'{te.mode} {te.from_stop.name} → {te.to_stop.name} ({te.route_name})'
                )],
                geometry=line_interpolate(
                    te.from_stop.lat, te.from_stop.lon, te.to_stop.lat, te.to_stop.lon, 8,
                ),
                verified=True,
                note=f'OSM route: {te.route_name}',
            )
            self._add_edge(graph, sf, st, leg, f'transit:{te.mode}:{sf}->{st}')

        self._add_estimated_transit_edges(graph, stop_nodes, origin, dest)

        nodes = list(stop_nodes.values())
        direct_bus = self._estimated_public_leg(
            origin.name,
            dest.name,
            origin.lat,
            origin.lon,
            dest.lat,
            dest.lon,
            'bus',
            'Estimated public transit; OSM route relation unavailable',
        )
        self._add_edge(graph, ORIGIN_ID, DEST_ID, direct_bus, 'est:bus:origin->dest')

        for a, b in itertools.combinations(nodes, 2):
            dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
            if dist <= self.MAX_TRANSFER_WALK_KM:
                for na, nb in ((a, b), (b, a)):
                    leg = DetailedLeg(
                        mode='walk',
                        from_name=na.name,
                        to_name=nb.name,
                        from_lat=na.lat,
                        from_lon=na.lon,
                        to_lat=nb.lat,
                        to_lon=nb.lon,
                        distance_km=dist,
                        duration_min=max(dist / 5 * 60, 1),
                        steps=[RouteStep(f'Walk transfer {na.name} → {nb.name}')],
                        geometry=line_interpolate(na.lat, na.lon, nb.lat, nb.lon, 4),
                        verified=False,
                        note='Transfer walk between nearby stops',
                    )
                    self._add_edge(
                        graph, na.node_id, nb.node_id, leg, f'xfer:{na.node_id}->{nb.node_id}',
                    )

        ranked = sorted(
            nodes,
            key=lambda n: min(
                haversine_km(origin.lat, origin.lon, n.lat, n.lon),
                haversine_km(dest.lat, dest.lon, n.lat, n.lon),
            ),
        )
        for node in ranked[:20]:
            if 0.05 < haversine_km(origin.lat, origin.lon, node.lat, node.lon) <= self.MAX_ACCESS_KM:
                leg = self.osrm.route(
                    origin.lat, origin.lon, origin.name,
                    node.lat, node.lon, node.name, 'walk',
                )
                if leg:
                    self._add_edge(graph, ORIGIN_ID, node.node_id, leg, f'access:{node.node_id}')
            if 0.05 < haversine_km(dest.lat, dest.lon, node.lat, node.lon) <= self.MAX_ACCESS_KM:
                leg = self.osrm.route(
                    node.lat, node.lon, node.name,
                    dest.lat, dest.lon, dest.name, 'walk',
                )
                if leg:
                    self._add_edge(graph, node.node_id, DEST_ID, leg, f'egress:{node.node_id}')

        for node in ranked[:8]:
            to_stop = self._estimated_public_leg(
                origin.name,
                node.name,
                origin.lat,
                origin.lon,
                node.lat,
                node.lon,
                'bus',
                'Estimated access bus to OSM stop',
            )
            self._add_edge(graph, ORIGIN_ID, node.node_id, to_stop, f'est:bus:origin->{node.node_id}')
            from_stop = self._estimated_public_leg(
                node.name,
                dest.name,
                node.lat,
                node.lon,
                dest.lat,
                dest.lon,
                'bus',
                'Estimated egress bus from OSM stop',
            )
            self._add_edge(graph, node.node_id, DEST_ID, from_stop, f'est:bus:{node.node_id}->dest')
        return graph

    def _add_estimated_transit_edges(
        self,
        graph: MultimodalGraph,
        stop_nodes: dict[str, GraphNode],
        origin: Waypoint,
        dest: Waypoint,
    ) -> None:
        """Add OSM-stop-based transit edges when full OSM route relations are sparse."""
        nodes = list(stop_nodes.values())
        if not nodes:
            return
        origin_side = sorted(
            nodes, key=lambda n: haversine_km(origin.lat, origin.lon, n.lat, n.lon)
        )[:8]
        dest_side = sorted(
            nodes, key=lambda n: haversine_km(dest.lat, dest.lon, n.lat, n.lon)
        )[:8]
        seen: set[str] = set()
        for a in origin_side:
            for b in dest_side:
                if a.node_id == b.node_id:
                    continue
                dist = haversine_km(a.lat, a.lon, b.lat, b.lon) * 1.25
                if dist < 0.5:
                    continue
                mode = self._estimated_mode(a, b)
                speed = {'bus': 24, 'train': 55, 'tram': 30}.get(mode, 24)
                duration = max(dist / speed * 60, 3.0)
                key = f'est:{mode}:{a.node_id}->{b.node_id}'
                if key in seen:
                    continue
                seen.add(key)
                leg = DetailedLeg(
                    mode=mode,
                    from_name=a.name,
                    to_name=b.name,
                    from_lat=a.lat,
                    from_lon=a.lon,
                    to_lat=b.lat,
                    to_lon=b.lon,
                    distance_km=dist,
                    duration_min=duration,
                    steps=[RouteStep(f'{mode} {a.name} → {b.name} (OSM stops, estimated service)')],
                    geometry=line_interpolate(a.lat, a.lon, b.lat, b.lon, 8),
                    verified=False,
                    note='Estimated transit edge from nearby OSM stops',
                )
                self._add_edge(graph, a.node_id, b.node_id, leg, key)

    @staticmethod
    def _estimated_mode(a: GraphNode, b: GraphNode) -> str:
        text = f'{a.name} {b.name}'.lower()
        if any(token in text for token in ('station', 'railway', 'train', 'metro', 'subway')):
            return 'train'
        if 'tram' in text or 'light rail' in text:
            return 'tram'
        return 'bus'

    @staticmethod
    def _estimated_public_leg(
        from_name: str,
        to_name: str,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str,
        note: str,
    ) -> DetailedLeg:
        dist = haversine_km(from_lat, from_lon, to_lat, to_lon) * 1.25
        speed = {'bus': 24, 'train': 55, 'tram': 30}.get(mode, 24)
        return DetailedLeg(
            mode=mode,
            from_name=from_name,
            to_name=to_name,
            from_lat=from_lat,
            from_lon=from_lon,
            to_lat=to_lat,
            to_lon=to_lon,
            distance_km=dist,
            duration_min=max(dist / speed * 60, 3.0),
            steps=[RouteStep(f'{mode} {from_name} → {to_name} (estimated public transit)')],
            geometry=line_interpolate(from_lat, from_lon, to_lat, to_lon, 8),
            verified=False,
            note=note,
        )

    @staticmethod
    def _add_edge(graph: MultimodalGraph, from_id: str, to_id: str, leg: DetailedLeg, key: str) -> None:
        graph.adj.setdefault(from_id, []).append(GraphEdge(to_id, leg, key))

    def _k_shortest_paths(
        self, graph: MultimodalGraph, k: int, allow_drive: bool = True,
    ) -> list[list[DetailedLeg]]:
        heap: list[tuple[float, float, int, str, list[DetailedLeg], Optional[str], frozenset[str]]] = []
        results: list[list[DetailedLeg]] = []
        seen_paths: set[tuple[str, ...]] = set()
        counter = 0
        explored = 0

        heapq.heappush(heap, (0.0, 0.0, counter, ORIGIN_ID, [], None, frozenset({ORIGIN_ID})))

        while heap and explored < self.MAX_EXPLORE and len(results) < k:
            _, cost, _, node, legs, prev_mode, visited = heapq.heappop(heap)
            explored += 1
            if node == DEST_ID and legs:
                signature = self._path_signature(legs)
                if signature not in seen_paths:
                    seen_paths.add(signature)
                    results.append(legs)
                continue
            if len(legs) >= self.MAX_LEGS:
                continue

            for edge in graph.adj.get(node, []):
                if not allow_drive and edge.leg.mode == 'drive':
                    continue
                if edge.to_id in visited and edge.to_id != DEST_ID:
                    continue
                xfer = (
                    self.TRANSFER_PENALTY_MIN
                    if prev_mode and prev_mode != edge.leg.mode
                    else 0.0
                )
                new_legs = legs + [edge.leg]
                if edge.to_id == DEST_ID:
                    signature = self._path_signature(new_legs)
                    if signature not in seen_paths:
                        seen_paths.add(signature)
                        results.append(new_legs)
                        if len(results) >= k:
                            break
                    continue
                counter += 1
                priority = cost + edge.leg.duration_min + xfer + self._heuristic_to_dest(graph, edge.to_id)
                heapq.heappush(
                    heap,
                    (
                        priority,
                        cost + edge.leg.duration_min + xfer,
                        counter,
                        edge.to_id,
                        new_legs,
                        edge.leg.mode,
                        visited | {edge.to_id},
                    ),
                )
            if len(results) >= k:
                break

        results.sort(key=lambda legs: sum(l.duration_min for l in legs))
        return results[:k]

    @staticmethod
    def _heuristic_to_dest(graph: MultimodalGraph, node_id: str) -> float:
        if node_id == DEST_ID:
            return 0.0
        node = graph.nodes.get(node_id)
        dest = graph.nodes.get(DEST_ID)
        if not node or not dest:
            return 0.0
        # Optimistic lower bound for public transport / walking mixtures.
        return haversine_km(node.lat, node.lon, dest.lat, dest.lon) / 55.0 * 60

    @staticmethod
    def _path_signature(legs: list[DetailedLeg]) -> tuple[str, ...]:
        return tuple(
            f'{l.mode}:{l.from_name}:{l.to_name}:{l.from_lat:.5f},{l.from_lon:.5f}->{l.to_lat:.5f},{l.to_lon:.5f}'
            for l in legs
        )

    @staticmethod
    def _option_signature(legs: list[DetailedLeg]) -> tuple[str, ...]:
        """Signature for suppressing duplicate-looking GUI options."""
        return tuple(f'{l.mode}:{l.from_name}->{l.to_name}' for l in legs)

    def _dijkstra(
        self,
        graph: MultimodalGraph,
        excluded: set[str],
        cost_cap: float = 1e9,
        allow_drive: bool = True,
    ) -> tuple[Optional[list[DetailedLeg]], list[str]]:
        heap: list[tuple[float, int, str, list[DetailedLeg], Optional[str], list[str]]] = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, ORIGIN_ID, [], None, []))

        best: dict[tuple[str, int, Optional[str]], float] = {}
        explored = 0

        while heap and explored < self.MAX_EXPLORE:
            cost, _, node, legs, prev_mode, path_keys = heapq.heappop(heap)
            explored += 1
            if cost > cost_cap:
                continue
            if node == DEST_ID and legs:
                return legs, path_keys
            if len(legs) >= self.MAX_LEGS:
                continue

            state = (node, len(legs), prev_mode)
            if state in best and best[state] <= cost:
                continue
            best[state] = cost

            for edge in graph.adj.get(node, []):
                if edge.edge_key in excluded:
                    continue
                if not allow_drive and edge.leg.mode == 'drive':
                    continue
                xfer = (
                    self.TRANSFER_PENALTY_MIN
                    if prev_mode and prev_mode != edge.leg.mode
                    else 0.0
                )
                new_cost = cost + edge.leg.duration_min + xfer
                if new_cost > cost_cap:
                    continue
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        counter,
                        edge.to_id,
                        legs + [edge.leg],
                        edge.leg.mode,
                        path_keys + [edge.edge_key],
                    ),
                )
        return None, []

    @staticmethod
    def _synthetic_leg(a: Waypoint, b: Waypoint, mode: str) -> DetailedLeg:
        from globe_nav.env.transport import TransportMode
        dist = haversine_km(a.lat, a.lon, b.lat, b.lon) * 1.3
        tm = TransportMode.from_str(mode)
        return DetailedLeg(
            mode=mode,
            from_name=a.name,
            to_name=b.name,
            from_lat=a.lat,
            from_lon=a.lon,
            to_lat=b.lat,
            to_lon=b.lon,
            distance_km=dist,
            duration_min=dist / tm.speed_kmh * 60,
            verified=False,
            note='Estimated fallback',
        )
