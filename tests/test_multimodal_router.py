"""Tests for graph-based multimodal router."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.env.transport import Waypoint
from globe_nav.planner.models import DetailedLeg, RouteStep
from globe_nav.planner.multimodal_router import (
    DEST_ID,
    ORIGIN_ID,
    GraphEdge,
    GraphNode,
    MultimodalGraph,
    MultimodalRouter,
)


def _leg(mode, frm, to, dur, lat=0.0, lon=0.0):
    return DetailedLeg(
        mode=mode,
        from_name=frm,
        to_name=to,
        from_lat=lat,
        from_lon=lon,
        to_lat=lat,
        to_lon=lon,
        distance_km=1,
        duration_min=dur,
        steps=[RouteStep(f'{mode} {frm}→{to}')],
    )


class TestGraphSearch(unittest.TestCase):
    def test_finds_bus_walk_train_chain(self):
        graph = MultimodalGraph()
        graph.add_node(GraphNode(ORIGIN_ID, 'A', -33.88, 151.20))
        graph.add_node(GraphNode('s1', 'Stop1', -33.881, 151.201))
        graph.add_node(GraphNode('s2', 'Stop2', -33.882, 151.202))
        graph.add_node(GraphNode(DEST_ID, 'B', -33.89, 151.21))
        graph.adj[ORIGIN_ID] = [GraphEdge('s1', _leg('bus', 'A', 'Stop1', 10), 'b1')]
        graph.adj['s1'] = [
            GraphEdge('s2', _leg('walk', 'Stop1', 'Stop2', 5), 'w1'),
            GraphEdge(DEST_ID, _leg('walk', 'Stop1', 'B', 40), 'w1d'),
        ]
        graph.adj['s2'] = [GraphEdge(DEST_ID, _leg('train', 'Stop2', 'B', 12), 't1')]
        router = MultimodalRouter(MagicMock())
        paths = router._k_shortest_paths(graph, k=5, allow_drive=False)
        self.assertTrue(paths)
        chains = [tuple(l.mode for l in p) for p in paths]
        self.assertIn(('bus', 'walk', 'train'), chains)

    def test_k_shortest_finds_parallel_modes(self):
        graph = MultimodalGraph()
        graph.add_node(GraphNode(ORIGIN_ID, 'A', 0, 0))
        graph.add_node(GraphNode(DEST_ID, 'B', 0, 0))
        graph.adj[ORIGIN_ID] = [
            GraphEdge(DEST_ID, _leg('walk', 'A', 'B', 30), 'direct:walk'),
            GraphEdge(DEST_ID, _leg('drive', 'A', 'B', 10), 'direct:drive'),
            GraphEdge(DEST_ID, _leg('bus', 'A', 'B', 15), 'direct:bus'),
        ]
        router = MultimodalRouter(MagicMock())
        active = router._k_shortest_paths(graph, k=5, allow_drive=False)
        self.assertEqual(len(active), 2)
        chains = {tuple(l.mode for l in p) for p in active}
        self.assertEqual(chains, {('walk',), ('bus',)})

    def test_categorized_drive_plus_active(self):
        osrm = MagicMock()
        walk = _leg('walk', 'A', 'B', 30)
        drive = _leg('drive', 'A', 'B', 10)
        bus = _leg('bus', 'A', 'B', 15)
        def route_side_effect(flat, flon, fn, tlat, tlon, tn, mode='drive'):
            return drive if mode == 'drive' else walk

        osrm.route.side_effect = route_side_effect

        router = MultimodalRouter(osrm)
        graph = MultimodalGraph()
        graph.add_node(GraphNode(ORIGIN_ID, 'A', -33.88, 151.20))
        graph.add_node(GraphNode(DEST_ID, 'B', -33.89, 151.21))
        graph.adj[ORIGIN_ID] = [
            GraphEdge(DEST_ID, walk, 'direct:walk'),
            GraphEdge(DEST_ID, bus, 'direct:bus'),
        ]

        with patch.object(router, '_build_graph', return_value=graph):
            options = router.find_top_options(
                Waypoint('A', -33.88, 151.20, 'poi'),
                Waypoint('B', -33.89, 151.21, 'poi'),
                active_k=5,
            )

        self.assertGreaterEqual(len(options), 2)
        self.assertEqual(options[0][0].mode, 'drive')
        active = [p for p in options if not router._uses_drive(p)]
        self.assertGreaterEqual(len(active), 1)
        self.assertLessEqual(len(active), 5)
        self.assertTrue(all(not router._uses_drive(p) for p in active))


class TestEnvironmentIntegration(unittest.TestCase):
    @patch('globe_nav.planner.multimodal_router.TransitNetworkBuilder.build_corridor_network')
    @patch('globe_nav.planner.osrm.OSRMClient.route')
    def test_local_drive_plus_active(self, mock_route, mock_net):
        from globe_nav.planner.environment import MobilityEnvironment
        from globe_nav.planner.transit_network import TransitNetwork

        mock_net.return_value = TransitNetwork()
        walk = _leg('walk', 'UTS', 'Opera', 120, -33.883, 151.200)
        walk.to_lat, walk.to_lon = -33.857, 151.215
        walk.verified = True
        drive = _leg('drive', 'UTS', 'Opera', 12, -33.883, 151.200)
        drive.to_lat, drive.to_lon = -33.857, 151.215
        drive.verified = True

        def route_side_effect(flat, flon, fn, tlat, tlon, tn, mode='drive'):
            if mode == 'drive':
                return drive
            return walk

        mock_route.side_effect = route_side_effect

        env = MobilityEnvironment(use_online=False)
        opts = env.get_local_segment_options(
            Waypoint('UTS', -33.883, 151.200, 'poi'),
            Waypoint('Opera', -33.857, 151.215, 'poi'),
        )
        self.assertGreaterEqual(len(opts), 1)
        self.assertLessEqual(len(opts), 6)
        self.assertEqual(opts[0][0].mode, 'drive')
        active = [p for p in opts if not any(l.mode == 'drive' for l in p)]
        self.assertLessEqual(len(active), 5)


if __name__ == '__main__':
    unittest.main()
