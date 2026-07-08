"""Human-readable route option display."""

from globe_nav.planner.models import RouteOption


def format_options(options: list[RouteOption]) -> str:
    lines = [f'=== {len(options)} route option(s) from environment ===\n']
    for opt in options:
        lines.append(f'--- {opt.option_id}: {opt.description} ---')
        lines.append(f'    Total: {opt.total_duration_display} | {opt.total_distance_km:.0f} km')
        lines.append(f'    Chain: {opt.mode_chain}')
        for j, leg in enumerate(opt.legs, 1):
            if leg.mode == 'fly' and 'OpenFlights' in leg.note:
                verified = '✓ OpenFlights (connectivity)'
            elif leg.verified:
                verified = '✓ OSM/OSRM'
            else:
                verified = '~ estimated'
            lines.append(f'    [{j}] {leg.summary()} [{verified}]')
            if leg.note:
                lines.append(f'        Note: {leg.note}')
            for step in leg.steps[:8]:
                lines.append(f'          → {step.instruction}')
            if len(leg.steps) > 8:
                lines.append(f'          ... +{len(leg.steps) - 8} more steps')
        lines.append('')
    return '\n'.join(lines)
