"""Multi-modal route graph for global navigation simulation."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.env.transport import RouteSegment, TransportMode, Waypoint


@dataclass
class NavNode:
    node_id: str
    waypoint: Waypoint
    segment_idx: int
    step_idx: int
    instruction: str
    mode: TransportMode
    neighbors: dict[str, 'NavNode'] = field(default_factory=dict)


class RouteGraph:
    """Turn a multi-segment route into a navigable graph."""

    def __init__(self, segments: list[RouteSegment]):
        self.segments = segments
        self.nodes: dict[str, NavNode] = {}
        self.node_order: list[str] = []
        self._build()

    def _build(self) -> None:
        node_list: list[NavNode] = []
        global_step = 0

        for seg_idx, segment in enumerate(self.segments):
            n_steps = max(len(segment.instructions), 3)
            geom = segment.geometry
            for step in range(n_steps):
                if geom:
                    gi = min(int(step / n_steps * (len(geom) - 1)), len(geom) - 1)
                    lat, lon = geom[gi]
                else:
                    lat = segment.from_waypoint.lat
                    lon = segment.from_waypoint.lon

                instr_idx = min(step, len(segment.instructions) - 1)
                wp = Waypoint(
                    name=f'{segment.from_waypoint.name} → {segment.to_waypoint.name} (step {step})',
                    lat=lat,
                    lon=lon,
                    place_type=segment.mode.value,
                )
                node_id = f's{seg_idx}_t{step}'
                node = NavNode(
                    node_id=node_id,
                    waypoint=wp,
                    segment_idx=seg_idx,
                    step_idx=step,
                    instruction=segment.instructions[instr_idx],
                    mode=segment.mode,
                )
                node_list.append(node)
                self.nodes[node_id] = node
                self.node_order.append(node_id)
                global_step += 1

        # Link consecutive nodes
        for i in range(len(node_list) - 1):
            curr, nxt = node_list[i], node_list[i + 1]
            action = self._default_action(curr, nxt)
            curr.neighbors[action] = nxt

        # Terminal node can stop
        if node_list:
            node_list[-1].neighbors['stop'] = node_list[-1]

    def _default_action(self, curr: NavNode, nxt: NavNode) -> str:
        if curr.mode in (TransportMode.WALK, TransportMode.DRIVE):
            return 'forward'
        if curr.mode in (TransportMode.BUS, TransportMode.TRAIN):
            if curr.step_idx == 0:
                return 'board'
            if nxt.step_idx >= len(self.segments[curr.segment_idx].instructions) - 1:
                return 'arrive'
            return 'cruise'
        if curr.mode == TransportMode.FLY:
            if curr.step_idx == 0:
                return 'takeoff'
            if nxt.step_idx >= len(self.segments[curr.segment_idx].instructions) - 1:
                return 'land'
            return 'cruise'
        if curr.step_idx == 0:
            return 'depart'
        if nxt.step_idx >= len(self.segments[curr.segment_idx].instructions) - 1:
            return 'dock'
        return 'cruise'

    @property
    def start_node_id(self) -> str:
        return self.node_order[0] if self.node_order else ''

    @property
    def goal_node_id(self) -> str:
        return self.node_order[-1] if self.node_order else ''

    def get_observation(self, node_id: str) -> dict:
        node = self.nodes[node_id]
        segment = self.segments[node.segment_idx]
        return {
            'location': node.waypoint.name,
            'lat': node.waypoint.lat,
            'lon': node.waypoint.lon,
            'mode': node.mode.display_name,
            'instruction': node.instruction,
            'segment': f'{segment.from_waypoint.name} → {segment.to_waypoint.name}',
            'progress': f'step {node.step_idx + 1} of segment {node.segment_idx + 1}/{len(self.segments)}',
            'available_actions': list(node.neighbors.keys()),
        }
