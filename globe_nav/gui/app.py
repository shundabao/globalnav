"""Flask GUI for modular segment-by-segment trip planning."""

import json
import math
import os
import random
import sys
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory, Response

# Ensure package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from globe_nav.clarification import InstructionClarifier
from globe_nav.config import DEFAULT_MODEL, load_env
from globe_nav.follower.simulator import bearing_deg, bearing_label
from globe_nav.follower.agent import InstructionFollowerAgent
from globe_nav.maps.geocoder import OFFLINE_PLACES
from globe_nav.maps.streetview import GlobalStreetViewProvider
from globe_nav.planner.segment_models import ModularTripPlan
from globe_nav.planner.trip_builder import ModularTripBuilder

GUI_DIR = os.path.join(os.path.dirname(__file__), 'static')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO_ROOT, 'data', 'globnav_bench')
BENCH_PILOT_PATH = os.path.join(BENCH_DIR, 'pilot_500.jsonl')
BENCH_REVIEW_PATH = os.path.join(BENCH_DIR, 'annotations.review.jsonl')
INSTRUCTION_ANNOTATION_PATH = os.path.join(BENCH_DIR, 'instruction_annotations.jsonl')

ANNOTATION_RANDOM_POOLS = [
    {
        'city': 'Sydney',
        'places': ['UTS', 'Sydney Opera House', 'Sydney Central Station', 'Sydney Airport'],
    },
    {
        'city': 'New York',
        'places': ['Times Square', 'JFK', 'New York'],
    },
    {
        'city': 'London',
        'places': ['London', 'Heathrow'],
    },
    {
        'city': 'Paris',
        'places': ['Paris', 'Eiffel Tower'],
    },
    {
        'city': 'Global',
        'places': ['UTS', 'Times Square', 'Heathrow', 'Eiffel Tower', 'Sydney Opera House'],
    },
]


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _selected_option(seg, selections: dict):
    oid = selections.get(seg.segment_id, seg.default_option_id)
    return next((o for o in seg.options if o.option_id == oid), seg.options[0] if seg.options else None)


def _turn_delta(in_deg: float, out_deg: float) -> float:
    """Signed turn in degrees: positive is right, negative is left."""
    return (out_deg - in_deg + 540) % 360 - 180


def _instruction_action(instruction: str, mode: str) -> str | None:
    text = (instruction or '').lower()
    if any(tok in text for tok in ('u-turn', 'u turn', 'turn around', '掉头')):
        return 'u_turn' if mode == 'drive' else 'turn_around'
    if 'left' in text or '左' in text:
        return 'left'
    if 'right' in text or '右' in text:
        return 'right'
    return None


def _geometry_action(points: list[tuple[float, float]], idx: int, mode: str) -> tuple[str, float]:
    if idx <= 0 or idx >= len(points) - 1:
        return 'forward', 0.0
    prev_pt, cur_pt, next_pt = points[idx - 1], points[idx], points[idx + 1]
    incoming = bearing_deg(prev_pt[0], prev_pt[1], cur_pt[0], cur_pt[1])
    outgoing = bearing_deg(cur_pt[0], cur_pt[1], next_pt[0], next_pt[1])
    delta = _turn_delta(incoming, outgoing)
    if abs(delta) >= 150:
        return ('u_turn' if mode == 'drive' else 'turn_around'), round(delta, 1)
    if delta >= 50:
        return 'right', round(delta, 1)
    if delta <= -50:
        return 'left', round(delta, 1)
    return 'forward', round(delta, 1)


def _route_nodes_for_selection(
    plan: ModularTripPlan,
    selections: dict,
    *,
    sample_every: int = 1,
    ground_only: bool = True,
    max_nodes: int = 1200,
) -> list[dict]:
    nodes: list[dict] = []
    streetview_provider = GlobalStreetViewProvider()
    selected_micro_legs = []
    for seg_idx, seg in enumerate(plan.segments):
        opt = _selected_option(seg, selections)
        if not opt:
            continue
        for micro_idx, micro in enumerate(opt.micro_legs):
            selected_micro_legs.append((seg_idx, seg, opt, micro_idx, micro))

    ground_modes = {'walk', 'drive'}
    global_ground_points = [
        (micro.mode, pt)
        for _, _, _, _, micro in selected_micro_legs
        if micro.mode in ground_modes
        for pt in (micro.geometry or [])
    ]
    final_ground = global_ground_points[-1][1] if global_ground_points else None

    for leg_idx, (seg_idx, seg, opt, micro_idx, micro) in enumerate(selected_micro_legs):
        mode = micro.mode
        if ground_only and mode not in ground_modes:
            continue

        geom = [(float(lat), float(lon)) for lat, lon in (micro.geometry or []) if lat or lon]
        if not geom:
            if ground_only:
                continue
            geom = [(0.0, 0.0)]

        step_texts = list(micro.steps or [])
        keep_indices = set(range(0, len(geom), max(1, sample_every)))
        keep_indices.add(len(geom) - 1)

        for point_idx, (lat, lon) in enumerate(geom):
            if point_idx not in keep_indices:
                continue
            if len(nodes) >= max_nodes:
                break

            next_idx = min(point_idx + 1, len(geom) - 1)
            heading = (
                bearing_deg(lat, lon, geom[next_idx][0], geom[next_idx][1])
                if point_idx < len(geom) - 1 else
                (bearing_deg(geom[point_idx - 1][0], geom[point_idx - 1][1], lat, lon) if point_idx > 0 else 0.0)
            )
            step_idx = min(int(point_idx * max(len(step_texts), 1) / max(len(geom), 1)), max(len(step_texts) - 1, 0))
            route_cue = step_texts[step_idx] if step_texts else ''

            is_final_ground = bool(final_ground and abs(lat - final_ground[0]) < 1e-9 and abs(lon - final_ground[1]) < 1e-9)
            if mode in ground_modes:
                action = 'stop' if is_final_ground else (_instruction_action(route_cue, mode) or _geometry_action(geom, point_idx, mode)[0])
                turn_degrees = 0.0 if action == 'stop' else _geometry_action(geom, point_idx, mode)[1]
                streetview = {
                    'available': True,
                    'source': 'google',
                    'image_url': streetview_provider.api_image_path(lat, lon, heading),
                    'lat': round(lat, 7),
                    'lon': round(lon, 7),
                    'heading': round(heading, 1),
                }
            else:
                phase_actions = {
                    'bus': ['board', 'cruise', 'arrive', 'stop'],
                    'train': ['board', 'cruise', 'arrive', 'stop'],
                    'tram': ['board', 'cruise', 'arrive', 'stop'],
                    'fly': ['takeoff', 'cruise', 'land', 'stop'],
                    'maritime': ['depart', 'cruise', 'dock', 'stop'],
                }.get(mode, ['forward', 'stop'])
                phase_idx = min(point_idx, len(phase_actions) - 1)
                action = phase_actions[phase_idx]
                turn_degrees = 0.0
                streetview = {
                    'available': False,
                    'reason': f'{mode}_phase',
                    'message': f'{mode} phase has no street-level imagery',
                    'image_url': '',
                }

            nodes.append({
                'node_id': f'n{len(nodes):05d}',
                'index': len(nodes),
                'segment_id': seg.segment_id,
                'segment_title': seg.title,
                'segment_index': seg_idx,
                'option_id': opt.option_id,
                'option_label': opt.label,
                'leg_index': leg_idx,
                'micro_leg_index': micro_idx,
                'point_index': point_idx,
                'mode': mode,
                'from': micro.from_name,
                'to': micro.to_name,
                'lat': round(lat, 7),
                'lon': round(lon, 7),
                'heading_deg': round(heading, 1),
                'facing': bearing_label(heading),
                'route_cue': route_cue,
                'oracle_action': action,
                'turn_degrees': turn_degrees,
                'is_keypoint': action not in ('forward', 'cruise'),
                'streetview': streetview,
                'annotation_instruction': '',
                'action_override': '',
            })

        if len(nodes) >= max_nodes:
            break

    return nodes


def _random_annotation_pair(scope: str = 'local') -> dict:
    rng = random.Random(time.time_ns())
    if scope == 'global':
        places = sorted(OFFLINE_PLACES.keys())
        origin, destination = rng.sample(places, 2)
        return {'scope': 'global', 'origin': origin.title(), 'destination': destination.title()}

    pool = rng.choice(ANNOTATION_RANDOM_POOLS[:-1])
    origin, destination = rng.sample(pool['places'], 2)
    return {
        'scope': 'local',
        'city': pool['city'],
        'origin': origin,
        'destination': destination,
    }


def create_app(model: str = DEFAULT_MODEL, offline: bool = False) -> Flask:
    load_env()
    app = Flask(__name__, static_folder=GUI_DIR)
    clarifier = InstructionClarifier(model=model)
    builder = ModularTripBuilder()
    builder.env.geocoder.use_online = not offline
    streetview = GlobalStreetViewProvider()
    follower_sessions: dict[str, InstructionFollowerAgent] = {}
    annotation_sessions: dict[str, dict] = {}

    app.config['last_plan'] = None

    @app.route('/')
    def index():
        return send_from_directory(GUI_DIR, 'index.html')

    @app.route('/follower')
    def follower_page():
        return send_from_directory(GUI_DIR, 'follower.html')

    @app.route('/bench')
    def bench_page():
        return send_from_directory(GUI_DIR, 'bench.html')

    @app.route('/annotate')
    def annotate_page():
        return send_from_directory(GUI_DIR, 'annotate.html')

    @app.route('/<path:path>')
    def static_files(path):
        return send_from_directory(GUI_DIR, path)

    @app.post('/api/plan')
    def api_plan():
        data = request.get_json(force=True) or {}
        instruction = (data.get('instruction') or '').strip()
        if not instruction:
            return jsonify({'error': 'instruction required'}), 400

        try:
            result = clarifier.analyze(instruction)
            if not result.is_clear:
                return jsonify({
                    'status': 'clarifying',
                    'questions': [
                        {'category': a.category, 'question': a.question, 'options': a.options}
                        for a in result.ambiguities
                    ],
                })

            plan = builder.build(
                result.resolved_origin,
                result.resolved_destination,
                instruction=instruction,
            )
            app.config['last_plan'] = plan
            payload = plan.to_dict()
            payload['status'] = 'ok'
            payload['route_geometry'] = plan.path_for_selections()
            return jsonify(payload)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/clarify')
    def api_clarify():
        data = request.get_json(force=True) or {}
        instruction = data.get('instruction', '')
        category = data.get('category', '')
        answer = data.get('answer', '')
        clarifier._last_instruction = instruction
        result = clarifier.answer_and_reparse(category, data.get('question', category), answer)
        if not result.is_clear:
            return jsonify({
                'status': 'clarifying',
                'questions': [
                    {'category': a.category, 'question': a.question, 'options': a.options}
                    for a in result.ambiguities
                ],
            })
        plan = builder.build(result.resolved_origin, result.resolved_destination, instruction)
        app.config['last_plan'] = plan
        payload = plan.to_dict()
        payload['status'] = 'ok'
        payload['route_geometry'] = plan.path_for_selections()
        return jsonify(payload)

    @app.post('/api/total')
    def api_total():
        data = request.get_json(force=True) or {}
        plan: ModularTripPlan = app.config.get('last_plan')
        selections = data.get('selections', {})
        if plan:
            out = plan.compute_total(selections)
            out['route_geometry'] = plan.path_for_selections(selections)
            return jsonify(out)
        return jsonify({'error': 'no active plan'}), 400

    @app.get('/api/streetview/status')
    def api_streetview_status():
        enabled = request.args.get('enabled', '1') != '0'
        provider = GlobalStreetViewProvider(enabled=enabled)
        return jsonify(provider.status_dict())

    @app.get('/api/annotate/random')
    def api_annotate_random():
        scope = request.args.get('scope', 'local').strip() or 'local'
        return jsonify({'status': 'ok', **_random_annotation_pair(scope)})

    @app.post('/api/annotate/plan')
    def api_annotate_plan():
        data = request.get_json(force=True) or {}
        origin = (data.get('origin') or '').strip()
        destination = (data.get('destination') or '').strip()
        if not origin or not destination:
            return jsonify({'error': 'origin and destination required'}), 400
        instruction = (data.get('instruction') or f'Annotate route from {origin} to {destination}').strip()
        try:
            plan = builder.build(origin, destination, instruction=instruction)
            sid = str(uuid.uuid4())
            annotation_sessions[sid] = {
                'plan': plan,
                'origin_input': origin,
                'destination_input': destination,
                'instruction': instruction,
                'created_at': int(time.time()),
            }
            payload = plan.to_dict()
            payload.update({
                'status': 'ok',
                'session_id': sid,
                'route_geometry': plan.path_for_selections(),
                'streetview_status': streetview.status_dict(),
            })
            return jsonify(payload)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/annotate/start')
    def api_annotate_start():
        data = request.get_json(force=True) or {}
        sid = data.get('session_id')
        session = annotation_sessions.get(sid)
        if not session:
            return jsonify({'error': 'annotation session not found'}), 404

        plan: ModularTripPlan = session['plan']
        selections = dict(data.get('selections') or {})
        for seg in plan.segments:
            selections.setdefault(seg.segment_id, seg.default_option_id)
        try:
            sample_every = max(1, int(data.get('sample_every', 1)))
            max_nodes = min(3000, max(10, int(data.get('max_nodes', 1200))))
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid sample_every or max_nodes'}), 400

        nodes = _route_nodes_for_selection(
            plan,
            selections,
            sample_every=sample_every,
            ground_only=bool(data.get('ground_only', True)),
            max_nodes=max_nodes,
        )
        session.update({
            'selections': selections,
            'nodes': nodes,
            'sample_every': sample_every,
            'ground_only': bool(data.get('ground_only', True)),
        })
        return jsonify({
            'status': 'ok',
            'session_id': sid,
            'origin': plan.origin,
            'destination': plan.destination,
            'selections': selections,
            'node_count': len(nodes),
            'nodes': nodes,
            'route_geometry': plan.path_for_selections(selections),
            'total': plan.compute_total(selections),
            'streetview_status': streetview.status_dict(),
        })

    @app.post('/api/annotate/save')
    def api_annotate_save():
        data = request.get_json(force=True) or {}
        sid = data.get('session_id')
        session = annotation_sessions.get(sid, {})
        plan: ModularTripPlan | None = session.get('plan')
        nodes = data.get('nodes') or []
        if not sid:
            return jsonify({'error': 'session_id required'}), 400
        if not nodes:
            return jsonify({'error': 'nodes required'}), 400
        selections = data.get('selections') or session.get('selections') or {}
        row = {
            'id': f'instr_ann_{int(time.time())}_{sid[:8]}',
            'session_id': sid,
            'annotator': data.get('annotator', ''),
            'comments': data.get('comments', ''),
            'created_at': int(time.time()),
            'schema_version': 'instruction_nodes_v1',
            'origin': plan.origin if plan else data.get('origin', ''),
            'destination': plan.destination if plan else data.get('destination', ''),
            'origin_input': session.get('origin_input', ''),
            'destination_input': session.get('destination_input', ''),
            'route_instruction': session.get('instruction', ''),
            'selections': selections,
            'sample_every': session.get('sample_every'),
            'ground_only': session.get('ground_only'),
            'node_count': len(nodes),
            'annotated_count': sum(1 for n in nodes if (n.get('annotation_instruction') or '').strip()),
            'keypoint_count': sum(1 for n in nodes if n.get('is_keypoint')),
            'total': plan.compute_total(selections) if plan else {},
            'plan': plan.to_dict() if plan else {},
            'route_geometry': plan.path_for_selections(selections) if plan else {},
            'nodes': nodes,
        }
        _append_jsonl(INSTRUCTION_ANNOTATION_PATH, row)
        return jsonify({
            'status': 'ok',
            'saved_to': INSTRUCTION_ANNOTATION_PATH,
            'record_id': row['id'],
            'node_count': len(nodes),
        })

    @app.get('/api/bench/examples')
    def api_bench_examples():
        examples = _read_jsonl(BENCH_PILOT_PATH)
        split = request.args.get('split', '').strip()
        query = request.args.get('q', '').strip().lower()
        try:
            offset = int(request.args.get('offset', 0))
            limit = min(100, max(1, int(request.args.get('limit', 25))))
        except ValueError:
            return jsonify({'error': 'invalid offset/limit'}), 400
        if split:
            examples = [ex for ex in examples if ex.get('split') == split]
        if query:
            examples = [
                ex for ex in examples
                if query in ex.get('id', '').lower()
                or query in ex.get('instruction', '').lower()
                or query in str(ex.get('categories', [])).lower()
            ]
        page = examples[offset:offset + limit]
        return jsonify({
            'status': 'ok',
            'total': len(examples),
            'offset': offset,
            'limit': limit,
            'examples': page,
        })

    @app.post('/api/bench/review')
    def api_bench_review():
        data = request.get_json(force=True) or {}
        example_id = (data.get('example_id') or '').strip()
        if not example_id:
            return jsonify({'error': 'example_id required'}), 400
        row = {
            'example_id': example_id,
            'review_status': data.get('review_status', 'needs_revision'),
            'endpoint_ok': bool(data.get('endpoint_ok', False)),
            'route_ok': bool(data.get('route_ok', False)),
            'clarification_ok': bool(data.get('clarification_ok', False)),
            'follower_oracle_ok': bool(data.get('follower_oracle_ok', False)),
            'rewritten_instruction': data.get('rewritten_instruction', ''),
            'procedural_notes': data.get('procedural_notes', ''),
            'comments': data.get('comments', ''),
            'annotator': data.get('annotator', ''),
            'created_at': int(time.time()),
        }
        _append_jsonl(BENCH_REVIEW_PATH, row)
        return jsonify({'status': 'ok', 'saved_to': BENCH_REVIEW_PATH, 'review': row})

    @app.get('/api/streetview/image')
    def api_streetview_image():
        try:
            lat = float(request.args.get('lat', 0))
            lon = float(request.args.get('lon', 0))
            heading = float(request.args.get('heading', 0))
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid params'}), 400
        enabled = request.args.get('enabled', '1') != '0'
        provider = GlobalStreetViewProvider(enabled=enabled)
        data = provider.image_bytes(lat, lon, heading)
        if not data:
            return Response(status=204)
        return Response(data, mimetype='image/jpeg')

    @app.post('/api/follower/clarify')
    def api_follower_clarify():
        data = request.get_json(force=True) or {}
        instruction = (data.get('instruction') or '').strip()
        if not instruction:
            return jsonify({'error': 'instruction required'}), 400
        try:
            clarifier._last_instruction = instruction
            result = clarifier.answer_and_reparse(
                data.get('category', ''),
                data.get('question', data.get('category', '')),
                data.get('answer', ''),
            )
            if not result.is_clear:
                return jsonify({
                    'status': 'clarifying',
                    'questions': [
                        {'category': a.category, 'question': a.question, 'options': a.options}
                        for a in result.ambiguities
                    ],
                })
            agent = InstructionFollowerAgent(
                model=model, use_online_maps=not offline,
                rule_based=bool(data.get('rule_based', True)),
                streetview=bool(data.get('streetview', True)),
            )
            prep = agent.prepare(
                instruction, data.get('selections'),
                origin=result.resolved_origin,
                destination=result.resolved_destination,
                use_llm=not data.get('no_llm', False),
            )
            if prep.get('status') == 'clarifying':
                return jsonify(prep)
            sid = str(uuid.uuid4())
            follower_sessions[sid] = agent
            prep['session_id'] = sid
            return jsonify(prep)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/follower/prepare')
    def api_follower_prepare():
        data = request.get_json(force=True) or {}
        instruction = (data.get('instruction') or '').strip()
        if not instruction:
            return jsonify({'error': 'instruction required'}), 400
        try:
            agent = InstructionFollowerAgent(
                model=model, use_online_maps=not offline,
                rule_based=bool(data.get('rule_based', True)),
                streetview=bool(data.get('streetview', True)),
            )
            prep = agent.prepare(
                instruction, data.get('selections'),
                origin=data.get('origin'), destination=data.get('destination'),
                use_llm=not data.get('no_llm', False),
            )
            if prep.get('status') == 'clarifying':
                return jsonify(prep)
            sid = str(uuid.uuid4())
            follower_sessions[sid] = agent
            prep['session_id'] = sid
            return jsonify(prep)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/follower/select')
    def api_follower_select():
        data = request.get_json(force=True) or {}
        sid = data.get('session_id')
        agent = follower_sessions.get(sid)
        if not agent:
            return jsonify({'error': 'session not found'}), 404
        try:
            out = agent.update_selections(data.get('selections', {}))
            out['plan'] = agent.plan.to_dict() if agent.plan else {}
            out['selections'] = agent.selections
            return jsonify(out)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/follower/step')
    def api_follower_step():
        data = request.get_json(force=True) or {}
        sid = data.get('session_id')
        agent = follower_sessions.get(sid)
        if not agent:
            return jsonify({'error': 'session not found'}), 404
        try:
            return jsonify(agent.step_once())
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post('/api/follower/run')
    def api_follower_run():
        data = request.get_json(force=True) or {}
        instruction = (data.get('instruction') or '').strip()
        if not instruction:
            return jsonify({'error': 'instruction required'}), 400
        rule_based = bool(data.get('rule_based', True))
        max_steps = int(data.get('max_steps', 500))
        try:
            agent = InstructionFollowerAgent(
                model=model, use_online_maps=not offline, rule_based=rule_based,
                streetview=bool(data.get('streetview', True)),
            )
            prep = agent.prepare(
                instruction, data.get('selections'),
                origin=data.get('origin'), destination=data.get('destination'),
                use_llm=not data.get('no_llm', False),
            )
            if prep.get('status') == 'clarifying':
                return jsonify(prep)
            result = agent.run(max_steps=max_steps, verbatim=False)
            return jsonify({
                'status': 'done',
                'success': result.success,
                'steps': result.steps,
                'decomposed': result.decomposed,
                'trajectory': result.trajectory,
                'prepare': prep,
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8765)))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    app = create_app(model=args.model, offline=args.offline)
    print(f'GLOBALNAV GUI → http://{args.host}:{args.port}')
    print(f'  Planner:  http://{args.host}:{args.port}/')
    print(f'  Follower: http://{args.host}:{args.port}/follower')
    print(f'  Annotate: http://{args.host}:{args.port}/annotate')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
