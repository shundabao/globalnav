"""VELMA-style observation verbalization for global navigation."""

from globe_nav.follower.decomposer import DecomposedInstruction
from globe_nav.follower.simulator import GlobalInstructionSimulator
from globe_nav.planner.segment_models import ModularTripPlan

ACTION_SPACE_WALK = (
    'Action Space:\n'
    'forward (go straight), left (rotate left), right (rotate right), '
    'turn_around (u-turn), stop (end navigation)\n\n'
)
ACTION_SPACE_DRIVE = (
    'Action Space:\n'
    'forward, left, right, u_turn, stop\n\n'
)
ACTION_SPACE_TRANSIT = (
    'Action Space:\n'
    'board, forward, cruise, arrive, stop\n\n'
)
ACTION_SPACE_FLY = (
    'Action Space:\n'
    'takeoff, cruise, land, taxi, stop\n\n'
)


def action_space_for_mode(mode: str) -> str:
    if mode == 'drive':
        return ACTION_SPACE_DRIVE
    if mode in ('bus', 'train', 'tram'):
        return ACTION_SPACE_TRANSIT
    if mode == 'fly':
        return ACTION_SPACE_FLY
    return ACTION_SPACE_WALK


def build_init_prompt(instruction: str, decomposed: DecomposedInstruction,
                      plan: ModularTripPlan) -> str:
    lines = [
        'Navigate following the user instruction step by step!',
        '',
        action_space_for_mode('walk'),
        f'Navigation Instructions:\n"{instruction}"',
        '',
        f'Initial pose: at {decomposed.initial_location}, facing {decomposed.initial_facing}',
        f'Final goal: {decomposed.goal or plan.destination}',
        '',
    ]
    if decomposed.segment_goals:
        lines.append('Trip phases:')
        for g in decomposed.segment_goals:
            notes = f' — {g.procedural_notes}' if g.procedural_notes else ''
            lines.append(f'  - [{g.mode_hint}] {g.from_hint} → {g.to_hint}{notes}')
        lines.append('')
    lines.append('Action Sequence:')
    return '\n'.join(lines)


def get_navigation_lines(sim: GlobalInstructionSimulator, from_step: int = 0) -> tuple[list[str], list[bool]]:
    """Build VELMA-style action + observation lines for prompt."""
    lines: list[str] = []
    is_action: list[bool] = []
    actions = sim.state.actions

    step_id = 0
    while step_id < len(actions):
        action = actions[step_id]
        if action != 'init':
            lines.append(f'{len([a for a in actions[:step_id+1] if a != "init"])}. {action}')
            is_action.append(True)

        if step_id < len(actions) - 1 or (actions[-1] != 'stop' and not sim.state.done):
            pass  # observations appended after each step in replay

        step_id += 1

    # Current observation block
    if not sim.state.done:
        obs = sim.get_observation()
        obs_str = observations_to_str(obs)
        if obs_str:
            lines.append(obs_str)
            is_action.append(False)
        lines.append(f'{len([a for a in actions if a != "init"]) + 1}.')
        is_action.append(False)

    return lines, is_action


def observations_to_str(obs: dict) -> str:
    if obs.get('done'):
        return 'You have reached the destination.'

    parts = []
    mode = obs.get('mode', 'walk')
    parts.append(f'[Segment {obs.get("segment_index")}/{obs.get("segment_total")}, '
                 f'leg {obs.get("leg_index")}/{obs.get("leg_total")}, mode={mode}]')

    if obs.get('facing'):
        parts.append(f'You are facing {obs["facing"]} ({obs.get("heading_deg")}°).')

    if obs.get('osm_instruction'):
        instr = obs['osm_instruction']
        if obs.get('is_turn_ahead'):
            parts.append(f'Navigation cue: {instr} (turn ahead).')
        else:
            parts.append(f'Navigation cue: {instr}.')

    if obs.get('procedural_notes'):
        parts.append(f'Instruction reminder: {obs["procedural_notes"]}')

    parts.append(f'Progress: {obs.get("progress", "")} toward {obs.get("to", "")}.')

    if obs.get('is_turn_ahead'):
        parts.append('There is a turn ahead.')

    sv = obs.get('streetview') or {}
    if sv.get('available'):
        src = sv.get('source', 'imagery')
        parts.append(f'Street-level view available ({src}).')
    elif sv.get('reason') == 'no_coverage' and obs.get('mode') in ('walk', 'drive'):
        parts.append('No street-level imagery at this location.')

    return ' '.join(parts)


def build_step_prompt(init_prompt: str, sim: GlobalInstructionSimulator,
                      history_lines: list[str]) -> str:
    obs = sim.get_observation()
    mode_space = action_space_for_mode(obs.get('mode', 'walk'))
    body = init_prompt.replace(ACTION_SPACE_WALK, mode_space, 1)
    if history_lines:
        body += '\n' + '\n'.join(history_lines)
    else:
        body += '\n'
        obs_str = observations_to_str(obs)
        if obs_str:
            body += obs_str + '\n'
        body += '1.'
    return body
