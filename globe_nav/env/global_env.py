"""Global multi-modal navigation environment."""

from dataclasses import dataclass, field
from typing import Optional

from globe_nav.env.route_graph import RouteGraph
from globe_nav.env.transport import RouteSegment, TransportMode, Waypoint
from globe_nav.maps.router import MultiModalRouter


@dataclass
class NavState:
    node_id: str
    heading: float = 0.0
    actions: list[str] = field(default_factory=list)


class GlobalNavEnv:
    """Global navigation environment with LLM-planned multi-modal leg chains."""

    def __init__(self, use_online_maps: bool = True, cache_dir: str = 'data/cache'):
        self.router = MultiModalRouter(cache_dir=cache_dir, use_online=use_online_maps)
        self.graph: Optional[RouteGraph] = None
        self.state: Optional[NavState] = None
        self.task: Optional[dict] = None
        self.segments: list[RouteSegment] = []

    def reset(self, task: dict) -> NavState:
        self.task = task

        if task.get('legs'):
            self.segments = self.router.plan_from_legs(task['legs'])
        else:
            origin_query = task.get('origin') or task.get('resolved_origin')
            dest_query = task.get('destination') or task.get('resolved_destination')
            if not origin_query or not dest_query:
                raise ValueError('Task must have legs or resolved origin/destination')

            origin = self.router.resolve_place(origin_query)
            destination = self.router.resolve_place(dest_query)
            if not origin or not destination:
                raise ValueError(f'Could not geocode: origin={origin_query}, dest={dest_query}')

            mode_strs = task.get('modes', ['walk'])
            modes = [TransportMode.from_str(m) for m in mode_strs]
            if len(modes) == 1:
                self.segments = [self.router.route(origin, destination, modes[0])]
            else:
                self.segments = self.router.plan_multimodal(origin, destination, modes)

        self.graph = RouteGraph(self.segments)
        self.state = NavState(node_id=self.graph.start_node_id, actions=['init'])
        return self.state

    def get_observation(self) -> dict:
        assert self.graph and self.state
        obs = self.graph.get_observation(self.state.node_id)
        obs['task_instruction'] = self.task.get('instruction', '') if self.task else ''
        obs['total_distance_km'] = sum(s.distance_km for s in self.segments)
        obs['total_eta'] = self._format_total_eta()
        obs['leg_plan'] = [
            {
                'mode': s.mode.value,
                'from': s.from_waypoint.name,
                'to': s.to_waypoint.name,
                'description': s.description,
            }
            for s in self.segments
        ]
        return obs

    def _format_total_eta(self) -> str:
        total_h = sum(s.duration_hours for s in self.segments)
        if total_h < 1:
            return f'{int(total_h * 60)} min'
        if total_h < 24:
            return f'{total_h:.1f} hours'
        return f'{int(total_h // 24)}d {total_h % 24:.0f}h'

    def step(self, action: str) -> tuple[dict, bool, dict]:
        assert self.graph and self.state
        action = self._normalize_action(action)
        node = self.graph.nodes[self.state.node_id]

        if action == 'stop':
            self.state.actions.append('stop')
            done = self.state.node_id == self.graph.goal_node_id
            info = {'success': done, 'reason': 'stopped' if done else 'stopped_early'}
            return self.get_observation(), True, info

        if action not in node.neighbors:
            action = self._fallback_action(node)

        next_node = node.neighbors[action]
        self.state.node_id = next_node.node_id
        self.state.actions.append(action)

        at_goal = self.state.node_id == self.graph.goal_node_id
        if at_goal:
            self.state.actions.append('stop')
            return self.get_observation(), True, {'success': True, 'reason': 'reached_goal'}

        return self.get_observation(), False, {}

    def _normalize_action(self, action: str) -> str:
        mapping = {
            'head forward': 'forward', 'head_forward': 'forward',
            'turn left': 'left', 'turn_left': 'left',
            'turn right': 'right', 'turn_right': 'right',
            'turn around': 'turn_around', 'u-turn': 'u_turn', 'uturn': 'u_turn',
            'take off': 'takeoff', 'touch down': 'land',
        }
        return mapping.get(action.lower().strip(), action.lower().strip())

    def _fallback_action(self, node) -> str:
        actions = list(node.neighbors.keys())
        preferred = ['forward', 'cruise', 'takeoff', 'depart', 'board', 'land', 'dock', 'arrive']
        for p in preferred:
            if p in actions:
                return p
        return actions[0] if actions else 'stop'

    def get_action_space(self) -> list[str]:
        if not self.graph or not self.state:
            return []
        node = self.graph.nodes[self.state.node_id]
        return list(node.neighbors.keys()) + ['stop']
