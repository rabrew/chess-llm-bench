"""Offline-first, bounded OpenAI feasibility pilot; see specs/openai-pilot.md.

Default invocation only prints a plan. --execute requires spending approval.
No engines, Ollama, tools, response rescue, or automatic request retries.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib import error, request

import chess
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import score_all  # noqa: E402
from src.llm_client import build_prompt, parse_response  # noqa: E402

TIERS = ('easy', 'medium', 'hard', 'extreme')
MODEL_PRICES = {'gpt-6.1-sol': {'input': 2.0, 'output': 10.0},
                'gpt-6-luna': {'input': 0.10, 'output': 0.50}}
API_URL = 'https://api.openai.com/v1/responses'
TOKENS_PER_MILLION = 1_000_000
FRAMING_TOKEN_MARGIN = 4096
RESERVATION_MARGIN = 1.10
DEFAULT_CONFIG = 'config/config_openai_pilot.yaml'
DEFAULT_EVAL_RANGE = (-2000, 2000)
PILOT_OUTPUT_CAP = 4000


class PilotError(Exception):
    """Safe public message, never a transport/credential exception's contents."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    return math.isfinite(value) and value > 0 and (not integer or isinstance(value, int))


def validate_config(config):
    try:
        api, sample, paths = (config[k] for k in ('openai', 'benchmark', 'paths'))
        models = api['models']
        valid_models = (isinstance(models, list) and bool(models)
                        and all(isinstance(m, str) and m in MODEL_PRICES for m in models)
                        and len(set(models)) == len(models))
        if not valid_models or sample['prompt_format'] != 'board_text':
            raise PilotError('Unsupported or duplicate models/prompt format')
        for value, integer in ((api['budget_usd'], False), (api['timeout_seconds'], False),
                               (api['max_output_tokens'], True), (sample['positions_per_tier'], True)):
            if not positive(value, integer):
                raise PilotError('Configuration requires finite positive limits')
        if isinstance(sample['random_seed'], bool) or not isinstance(sample['random_seed'], int):
            raise PilotError('Sampling seed must be an integer')
        if api['max_output_tokens'] > PILOT_OUTPUT_CAP:
            raise PilotError('Pilot output cap exceeds registered 4000-token limit')
        if api['prices_per_mtok'] != MODEL_PRICES:
            raise PilotError('Pilot prices differ from the registered model prices')
        if Path(api['key_file']).expanduser().resolve().is_relative_to(ROOT):
            raise PilotError('Credential file must be outside the repository')
        if Path(paths['source_results']).resolve().is_relative_to(Path(paths['output_dir']).resolve()):
            raise PilotError('Pilot output directory must not contain its source input')
        for path in (api['key_file'], paths['source_results'], paths['output_dir']):
            if not isinstance(path, str) or not path:
                raise PilotError('Configured paths must be nonempty strings')
    except (KeyError, TypeError, ValueError):
        raise PilotError('Invalid pilot configuration') from None


def position_record(row):
    try:
        pos_id, fen, tier = (row[k] for k in ('position_id', 'fen', 'difficulty'))
        truth = row.get('stockfish_eval', row.get('t1_stockfish_eval'))
        if (isinstance(pos_id, bool) or not isinstance(pos_id, (int, str)) or str(pos_id) == ''
                or tier not in TIERS or not isinstance(fen, str)
                or isinstance(truth, bool) or not isinstance(truth, (int, float))
                or not math.isfinite(truth)):
            raise PilotError('Invalid source position metadata')
        chess.Board(fen)
        return {'position_id': pos_id, 'fen': fen, 'difficulty': tier,
                'stockfish_eval': truth, 'stockfish_best_move': row.get('stockfish_best_move'),
                'theme': row.get('theme', ''), 'phase': row.get('phase'), 'source': row.get('source')}
    except (KeyError, TypeError, ValueError):
        raise PilotError('Invalid source position metadata') from None


def sample_positions(raw, count, seed):
    if raw and not raw.endswith(b'\n'):
        raise PilotError('Source JSONL has an incomplete final line')
    positions = {}
    try:
        for line in raw.splitlines():
            if not line.strip():
                continue
            pos = position_record(json.loads(line))
            key = str(pos['position_id'])
            if key in positions and positions[key] != pos:
                raise PilotError('Inconsistent source FEN, truth or metadata for repeated position')
            positions[key] = pos
    except (ValueError, TypeError, UnicodeError):
        raise PilotError('Malformed source JSONL') from None
    rng, selected = random.Random(seed), []
    for tier in TIERS:
        candidates = sorted((p for p in positions.values() if p['difficulty'] == tier),
                            key=lambda p: str(p['position_id']))
        if len(candidates) < count:
            raise PilotError('Insufficient source positions for a balanced tier sample')
        selected.extend(rng.sample(candidates, count))
    return selected


def reservation_cost(body, prices):
    input_allowance = len(body['input'].encode('utf-8')) + FRAMING_TOKEN_MARGIN
    return RESERVATION_MARGIN * (input_allowance * prices['input']
                                + body['max_output_tokens'] * prices['output']) / TOKENS_PER_MILLION


def code_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def build_plan(config):
    validate_config(config)
    raw = Path(config['paths']['source_results']).read_bytes()
    api, sample = config['openai'], config['benchmark']
    selected = sample_positions(raw, sample['positions_per_tier'], sample['random_seed'])
    requests = []
    for model in api['models']:
        for position in selected:
            body = {'model': model, 'input': build_prompt(position['fen'], prompt_format='board_text'),
                    'tools': [], 'store': False, 'max_output_tokens': api['max_output_tokens']}
            requests.append({'id': digest({'model': model, 'position': position, 'body': body}),
                             'model': model, 'position': position, 'body': body,
                             'reserved_usd': reservation_cost(body, MODEL_PRICES[model])})
    plan = {'schema_version': 1, 'config': config,
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'code_revision': code_revision(),
            'settings': {'reasoning': 'API default', 'sampling': 'API default', 'rescue': False},
            'requests': requests, 'max_reserved_usd': sum(r['reserved_usd'] for r in requests)}
    plan['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan['plan_hash'] = digest({k: v for k, v in plan.items() if k != 'code_revision'})
    return plan


def append_event(path, event):
    with path.open('a', encoding='utf-8') as f:
        f.write(canonical(event) + '\n')
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


@contextmanager
def execution_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PilotError('Another pilot process is already running') from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def bind_plan(plan, output_dir):
    path = output_dir / 'plan.json'
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (ValueError, OSError):
            raise PilotError('Unreadable existing plan; manual review required') from None
        if ({k: v for k, v in existing.items() if k != 'code_revision'}
                != {k: v for k, v in plan.items() if k != 'code_revision'}):
            raise PilotError('Output plan mismatch; use a new output directory')
    else:
        if any(p.name != '.lock' for p in output_dir.iterdir()):
            raise PilotError('Output directory contains unbound files; refusing to overwrite')
        atomic_json(path, plan)


def usage_counts(response):
    try:
        usage = response['usage']
        counts = {k: usage[k] for k in ('input_tokens', 'output_tokens')}
        counts['reasoning_tokens'] = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()):
            raise PilotError('Invalid response usage; manual billing review required')
        if counts['reasoning_tokens'] > counts['output_tokens']:
            raise PilotError('Invalid reasoning-token usage; manual billing review required')
        return counts
    except (TypeError, KeyError, AttributeError):
        raise PilotError('Missing response usage; manual billing review required') from None


def actual_cost(response, model):
    usage, prices = usage_counts(response), MODEL_PRICES[model]
    return (usage['input_tokens'] * prices['input']
            + usage['output_tokens'] * prices['output']) / TOKENS_PER_MILLION


def read_events(path):
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise PilotError('Incomplete pilot ledger; manual review required')
        return [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError):
        raise PilotError('Malformed pilot ledger; manual review required') from None


def ledger_state(plan, path):
    requests = {r['id']: r for r in plan['requests']}
    states, received, complete = {}, {}, {}
    for event in read_events(path):
        try:
            identity, kind = event['id'], event['event']
            if identity not in requests or event['plan_hash'] != plan['plan_hash']:
                raise PilotError('Ledger request/plan mismatch')
            state = states.get(identity)
            if kind == 'reserved' and state is None:
                if event['reserved_usd'] != requests[identity]['reserved_usd']:
                    raise PilotError('Ledger reservation mismatch')
                states[identity] = kind
            elif kind == 'received' and state == 'reserved':
                received[identity], states[identity] = event, kind
            elif kind == 'completed' and state == 'received':
                actual_cost(received[identity]['response'], requests[identity]['model'])
                complete[identity], states[identity] = received[identity], kind
            else:
                raise PilotError('Duplicate or out-of-order pilot ledger event')
        except (KeyError, TypeError):
            raise PilotError('Malformed pilot ledger event') from None
    if set(states) != set(complete):
        raise PilotError('An outstanding request needs manual review; it will not be resent')
    return complete


def load_key(path):
    value = os.environ.get('OPENAI_API_KEY', '').strip()
    if not value:
        try:
            private = Path(path).expanduser()
            if private.stat().st_mode & 0o077:
                raise PilotError('Credential file must have owner-only permissions (600)')
            lines = private.read_text().splitlines()
            values = [line.split('=', 1)[1].strip() for line in lines
                      if line.startswith('OPENAI_API_KEY=')]
            value = values[0] if len(values) == 1 else ''
        except (OSError, UnicodeError):
            raise PilotError('OpenAI credential file is unavailable') from None
    if not value or '\n' in value or '\r' in value:
        raise PilotError('OpenAI credential is missing or malformed')
    return value


def http_transport(body, key, timeout):
    req = request.Request(API_URL, data=canonical(body).encode(), method='POST',
                          headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as reply:
            return json.load(reply)
    except error.HTTPError as exc:
        raise PilotError(f'API HTTP status {exc.code}; request left outstanding') from None
    except (error.URLError, OSError, ValueError):
        raise PilotError('API transport/response failed; request left outstanding') from None


def reply_content(response):
    texts, refused = [], False
    try:
        for item in response.get('output', []):
            if item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    texts.append(content['text'])
                refused |= content.get('type') == 'refusal'
    except (TypeError, KeyError, AttributeError):
        raise PilotError('Invalid response output; manual review required') from None
    if not all(isinstance(text, str) for text in texts):
        raise PilotError('Invalid response text; manual review required')
    return '\n'.join(texts), refused


def scored_record(job, event):
    response = event['response']
    text, refused = reply_content(response)
    parsed, position = parse_response(text), job['position']
    scores = score_all(parsed, position, engine=None, eval_range=DEFAULT_EVAL_RANGE)
    # No engine scoring in the pilot, even when the stored reference move matches.
    scores['t2_cpl'] = None
    return {'job_id': job['id'], 'job_type': 'openai_pilot', 'attempt': 1,
            **position, **scores, 'model': job['model'], 'model_family': 'openai',
            'model_size_b': None, 'prompt_format': 'board_text', 'raw_response': text,
            't2_first_attempt_legal': bool(scores['t2_legal']), 'move_source': 'first_answer',
            'parse_errors': parsed.get('parse_errors', []), 'api_model': response.get('model'),
            'response_id': response.get('id'), 'response_status': response.get('status'),
            'incomplete_details': response.get('incomplete_details'), 'refused': refused,
            'inference_ms': event['inference_ms'], 'timestamp': event['timestamp'],
            **usage_counts(response), 'cost_usd': actual_cost(response, job['model'])}


def pilot_summary(plan, records):
    count = len(records)

    def capped(record):
        return (record.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    summary = {'scope': 'setup/cost pilot only; no population or superiority verdict',
               'plan_hash': plan['plan_hash'], 'requested': len(plan['requests']), 'completed': count,
               'capped': sum(capped(r) for r in records), 'refused': sum(r['refused'] for r in records),
               'missing_move': sum(r['t2_move'] is None for r in records),
               'missing_eval': sum(not finite(r['t1_model_eval']) for r in records),
               'missing_explanation': sum(not r.get('t3_explanation') for r in records),
               'finite_evaluation_pairs': sum(finite(r['t1_model_eval']) and
                                              finite(r['t1_stockfish_eval']) for r in records),
               'first_answer_legal': sum(r['t2_first_attempt_legal'] for r in records) / count if count else None,
               'cpl_status': 'pending separate engine scoring',
               'best_move_superiority': 'not tested'}
    for key in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cost_usd'):
        summary[key] = sum(r[key] for r in records)
    summary['variants'] = {model: {'completed': sum(r['model'] == model for r in records),
                                  'cost_usd': sum(r['cost_usd'] for r in records if r['model'] == model)}
                           for model in plan['config']['openai']['models']}
    return summary


def write_exports(plan, output_dir, complete):
    jobs = [job for job in plan['requests'] if job['id'] in complete]
    records = [scored_record(job, complete[job['id']]) for job in jobs]
    for name, values in (('responses.jsonl', [complete[job['id']] for job in jobs]),
                         ('evaluations.jsonl', records)):
        path = output_dir / name
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as f:
            for value in values:
                f.write(canonical(value) + '\n')
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    summary = pilot_summary(plan, records)
    atomic_json(output_dir / 'summary.json', summary)
    return summary


def submit_one(job, plan, output_dir, key, transport):
    ledger = output_dir / 'ledger.jsonl'
    common = {'id': job['id'], 'plan_hash': plan['plan_hash']}
    append_event(ledger, {**common, 'event': 'reserved', 'reserved_usd': job['reserved_usd']})
    started = time.monotonic()
    try:
        reply = transport(job['body'], key, plan['config']['openai']['timeout_seconds'])
    except Exception:
        raise PilotError('API request failed; outstanding reservation requires manual review') from None
    # Preserve invalid numeric tokens as explicit markers in strict JSON.
    try:
        reply = json.loads(json.dumps(reply), parse_constant=lambda token: {'invalid_number': token})
    except (TypeError, ValueError):
        raise PilotError('Unserializable response usage/output; request remains outstanding') from None
    received = {**common, 'event': 'received', 'response': reply,
                'inference_ms': round((time.monotonic() - started) * 1000),
                'timestamp': datetime.now(timezone.utc).isoformat()}
    append_event(ledger, received)
    actual_cost(reply, job['model'])
    scored_record(job, received)
    append_event(ledger, {**common, 'event': 'completed'})
    return received


def prepare_execution(plan, output_dir):
    ledger = output_dir / 'ledger.jsonl'
    if (output_dir / 'plan.json').exists() and (not ledger.exists() or ledger.stat().st_size == 0):
        raise PilotError('Established pilot ledger is missing/empty; manual review required')
    bind_plan(plan, output_dir)
    if not ledger.exists():
        with ledger.open('x') as stream:
            stream.flush()
            os.fsync(stream.fileno())
    complete = ledger_state(plan, ledger)
    for name, key in (('responses.jsonl', 'id'), ('evaluations.jsonl', 'job_id')):
        for record in read_events(output_dir / name):
            if not isinstance(record, dict) or record.get(key) not in complete:
                raise PilotError('Exported response is missing from ledger; manual review required')
    return complete


def execute_plan(plan, output_dir, key, transport=http_transport):
    output_dir = Path(output_dir)
    with execution_lock(output_dir):
        complete = prepare_execution(plan, output_dir)
        for job in plan['requests']:
            if job['id'] in complete:
                continue
            spent = sum(actual_cost(event['response'], next(r['model'] for r in plan['requests']
                        if r['id'] == identity)) for identity, event in complete.items())
            if spent + job['reserved_usd'] > plan['config']['openai']['budget_usd']:
                write_exports(plan, output_dir, complete)
                raise PilotError('Pilot budget cannot cover another conservatively reserved request')
            complete[job['id']] = submit_one(job, plan, output_dir, key, transport)
            write_exports(plan, output_dir, complete)
        return write_exports(plan, output_dir, complete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--execute', action='store_true', help='Paid submission; requires prior spending approval')
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        plan = build_plan(config)
        if not args.execute:
            print(json.dumps({'mode': 'offline plan only', 'requests': len(plan['requests']),
                              'max_reserved_usd': plan['max_reserved_usd'],
                              'budget_usd': config['openai']['budget_usd'],
                              'plan_hash': plan['plan_hash']}, indent=2))
            return 0
        key = load_key(config['openai']['key_file'])
        summary = execute_plan(plan, Path(config['paths']['output_dir']), key)
        print(json.dumps(summary, indent=2))
        return 0
    except (PilotError, OSError, yaml.YAMLError):
        print('Pilot stopped: configuration, input, budget, transport or ledger needs review.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
