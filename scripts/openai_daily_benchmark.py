"""GPT-5.4 first-answer campaign with a conservative rolling token allowance.

Default invocation is offline. Submission requires --execute,
--confirm-free-project and --exclusive-quota-use. Local accounting does not
verify free-credit entitlement or invoice charges. No engines or tools run.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import openai_pilot as pilot  # noqa: E402

CampaignError = pilot.PilotError
TIERS = pilot.TIERS
FORMATS = ('fen_only', 'board_text', 'board_text+legal')
MODEL = 'gpt-5.4'
PRICES = {'input': 2.50, 'output': 15.0}
MAX_ROLLING_TOKENS = 200_000
WINDOW_SECONDS = 86_400
MAX_OUTPUT_TOKENS = 4000
MAX_POSITIONS_PER_TIER = 125
DEFAULT_CONFIG = 'config/config_openai_gpt54.yaml'
BILLING_STATUS = 'free-credit eligibility and invoice charges not verified by runner'
execution_lock = pilot.execution_lock
load_key = pilot.load_key
http_transport = pilot.http_transport


def validate_config(config):
    try:
        api, quota, sample, paths = (config[k] for k in ('openai', 'quota', 'benchmark', 'paths'))
        if api['model'] != MODEL or api['prices_per_mtok'] != PRICES:
            raise CampaignError('Unsupported model or list prices')
        if sample['prompt_formats'] != list(FORMATS):
            raise CampaignError('Campaign requires the exact three registered prompt formats')
        integers = (api['max_output_tokens'], quota['rolling_tokens'], quota['window_seconds'],
                    sample['positions_per_tier'])
        if not all(pilot.positive(value, integer=True) for value in integers):
            raise CampaignError('Campaign limits must be positive finite integers')
        if (api['max_output_tokens'] > MAX_OUTPUT_TOKENS or quota['rolling_tokens'] > MAX_ROLLING_TOKENS
                or quota['window_seconds'] != WINDOW_SECONDS
                or sample['positions_per_tier'] > MAX_POSITIONS_PER_TIER):
            raise CampaignError('Campaign limits exceed the specified bounds')
        if not pilot.positive(api['timeout_seconds']):
            raise CampaignError('Timeout must be finite and positive')
        if isinstance(sample['random_seed'], bool) or not isinstance(sample['random_seed'], int):
            raise CampaignError('Seed must be an integer')
        for path in (api['key_file'], paths['source_results'], paths['output_dir']):
            if not isinstance(path, str) or not path:
                raise CampaignError('Paths must be nonempty strings')
        if Path(api['key_file']).expanduser().resolve().is_relative_to(ROOT):
            raise CampaignError('Credential file must be outside the repository')
        if Path(paths['source_results']).resolve().is_relative_to(Path(paths['output_dir']).resolve()):
            raise CampaignError('Output directory cannot contain its source data')
    except (KeyError, TypeError, ValueError):
        raise CampaignError('Invalid campaign configuration') from None


def fixed_cohort(raw, count, seed):
    selected = pilot.sample_positions(raw, count, seed)
    counts = {tier: set() for tier in TIERS}
    try:
        for line in raw.splitlines():
            if line.strip():
                row = pilot.position_record(json.loads(line))
                counts[row['difficulty']].add(str(row['position_id']))
    except (ValueError, TypeError, AttributeError):
        raise CampaignError('Invalid source cohort') from None
    if any(len(ids) != count for ids in counts.values()):
        raise CampaignError('Source cohort must contain exactly the configured positions per tier')
    groups = {tier: [row for row in selected if row['difficulty'] == tier] for tier in TIERS}
    return [groups[tier][i] for i in range(count) for tier in TIERS]


def request_plan(position, fmt, output_cap):
    body = {'model': MODEL, 'input': pilot.build_prompt(position['fen'], prompt_format=fmt),
            'tools': [], 'store': False, 'max_output_tokens': output_cap}
    return {'id': pilot.digest({'position': position, 'format': fmt, 'body': body}),
            'model': MODEL, 'position': position, 'format': fmt, 'body': body,
            'reserved_tokens': len(body['input'].encode('utf-8')) + pilot.FRAMING_TOKEN_MARGIN + output_cap}


def build_plan(config):
    validate_config(config)
    raw = Path(config['paths']['source_results']).read_bytes()
    sample = config['benchmark']
    positions = fixed_cohort(raw, sample['positions_per_tier'], sample['random_seed'])
    requests = [request_plan(position, fmt, config['openai']['max_output_tokens'])
                for position in positions for fmt in FORMATS]
    dependency_files = [Path(__file__), Path(pilot.__file__), ROOT / 'src/evaluator.py',
                        ROOT / 'src/llm_client.py']
    plan = {'schema_version': 1, 'config': config, 'requests': requests,
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'code_revision': pilot.code_revision(),
            'implementation_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in dependency_files},
            'billing_status': BILLING_STATUS,
            'settings': {'reasoning': 'API default', 'sampling': 'API default', 'rescue': False},
            'scope': 'descriptive first answers only; existing registered criteria unchanged'}
    plan['plan_hash'] = pilot.digest({key: value for key, value in plan.items() if key != 'code_revision'})
    return plan


def utc_now():
    return datetime.now(timezone.utc)


def checked_time(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise CampaignError('Clock must provide an aware UTC timestamp')
    if not math.isfinite(value.timestamp()) or value.timestamp() < 0:
        raise CampaignError('Invalid clock timestamp')
    return value


def event_time(event):
    try:
        return checked_time(datetime.fromisoformat(event['timestamp']))
    except (KeyError, TypeError, ValueError, OverflowError):
        raise CampaignError('Malformed ledger timestamp') from None


def check_clock(current, previous):
    checked_time(current)
    if previous is not None and current < previous:
        raise CampaignError('Clock rollback or future ledger timestamp; manual review required')


def list_price(response):
    usage = pilot.usage_counts(response)
    return (usage['input_tokens'] * PRICES['input']
            + usage['output_tokens'] * PRICES['output']) / pilot.TOKENS_PER_MILLION


def replay_ledger(plan, path, current):
    jobs = {job['id']: job for job in plan['requests']}
    states, received, complete, previous = {}, {}, {}, None
    for event in pilot.read_events(path):
        try:
            stamp = event_time(event)
            check_clock(stamp, previous)
            check_clock(current, stamp)
            previous = stamp
            if event['plan_hash'] != plan['plan_hash']:
                raise CampaignError('Ledger plan mismatch')
            if event['event'] == 'checkpoint':
                continue
            identity, kind = event['id'], event['event']
            if identity not in jobs:
                raise CampaignError('Unknown ledger request')
            state = states.get(identity)
            if kind == 'reserved' and state is None:
                if event['reserved_tokens'] != jobs[identity]['reserved_tokens']:
                    raise CampaignError('Ledger reservation mismatch')
            elif kind == 'received' and state == 'reserved':
                received[identity] = event
            elif kind == 'completed' and state == 'received':
                list_price(received[identity]['response'])
                complete[identity] = {**received[identity], 'completed_at': event['timestamp']}
            else:
                raise CampaignError('Duplicate/out-of-order ledger event')
            states[identity] = kind
        except (KeyError, TypeError, AttributeError):
            raise CampaignError('Malformed campaign ledger') from None
    if set(states) != set(complete):
        raise CampaignError('Outstanding request reservation requires manual review; no automatic resend')
    return complete, previous


def prepare_execution(plan, output_dir, current):
    ledger = output_dir / 'ledger.jsonl'
    if (output_dir / 'plan.json').exists() and (not ledger.exists() or ledger.stat().st_size == 0):
        raise CampaignError('Established ledger is missing/empty; manual review required')
    pilot.bind_plan(plan, output_dir)
    if not ledger.exists():
        with ledger.open('x') as stream:
            stream.flush()
            pilot.os.fsync(stream.fileno())
    complete, previous = replay_ledger(plan, ledger, current)
    for name, key in (('responses.jsonl', 'id'), ('evaluations.jsonl', 'job_id')):
        for record in pilot.read_events(output_dir / name):
            if not isinstance(record, dict) or record.get(key) not in complete:
                raise CampaignError('Completed transaction is missing from ledger; refusing export loss')
    check_clock(current, previous)
    return complete


def rolling_usage(complete, current, window):
    active = []
    for event in complete.values():
        stamp = event_time({'timestamp': event['completed_at']})
        check_clock(current, stamp)
        if current - stamp < timedelta(seconds=window):
            usage = pilot.usage_counts(event['response'])
            active.append((stamp, usage['input_tokens'] + usage['output_tokens']))
    return active


def quota_decision(plan, complete, current, needed):
    quota = plan['config']['quota']
    active = rolling_usage(complete, current, quota['window_seconds'])
    used = sum(tokens for _, tokens in active)
    if used + needed <= quota['rolling_tokens']:
        return True, used, None
    if needed > quota['rolling_tokens']:
        return False, used, None
    remaining = used
    for stamp, tokens in sorted(active):
        remaining -= tokens
        if remaining + needed <= quota['rolling_tokens']:
            return False, used, (stamp + timedelta(seconds=quota['window_seconds'])).isoformat()
    raise CampaignError('Quota accounting cannot establish a safe resume time')


def scored_record(job, event):
    response, position = event['response'], job['position']
    text, refused = pilot.reply_content(response)
    parsed = pilot.parse_response(text)
    scores = pilot.score_all(parsed, position, engine=None, eval_range=pilot.DEFAULT_EVAL_RANGE)
    scores['t2_cpl'] = None
    return {'job_id': job['id'], 'job_type': 'openai_daily_benchmark', 'attempt': 1,
            **position, **scores, 'model': MODEL, 'model_family': 'openai', 'model_size_b': None,
            'prompt_format': job['format'], 'raw_response': text,
            't2_first_attempt_legal': bool(scores['t2_legal']), 'move_source': 'first_answer',
            'parse_errors': parsed.get('parse_errors', []), 'api_model': response.get('model'),
            'response_id': response.get('id'), 'response_status': response.get('status'),
            'incomplete_details': response.get('incomplete_details'), 'refused': refused,
            'inference_ms': event['inference_ms'], 'timestamp': event['timestamp'],
            **pilot.usage_counts(response), 'estimated_list_price_usd': list_price(response),
            'billing_status': BILLING_STATUS}


def describe_records(records):
    count = len(records)
    return {'completed': count, 'capped': sum((r.get('incomplete_details') or {}).get('reason')
                                            == 'max_output_tokens' for r in records),
            'refused': sum(r['refused'] for r in records),
            'missing_move': sum(r['t2_move'] is None for r in records),
            'missing_eval': sum(r['t1_model_eval'] is None for r in records),
            'missing_explanation': sum(not r.get('t3_explanation') for r in records),
            'first_answer_legal': sum(r['t2_first_attempt_legal'] for r in records) / count if count else None,
            **{key: sum(r[key] for r in records) for key in
               ('input_tokens', 'output_tokens', 'reasoning_tokens', 'estimated_list_price_usd')}}


def write_exports(plan, output_dir, complete, current, status, earliest):
    jobs = [job for job in plan['requests'] if job['id'] in complete]
    records = [scored_record(job, complete[job['id']]) for job in jobs]
    for name, values in (('responses.jsonl', [complete[job['id']] for job in jobs]),
                         ('evaluations.jsonl', records)):
        path = output_dir / name
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as stream:
            for value in values:
                stream.write(pilot.canonical(value) + '\n')
            stream.flush()
            pilot.os.fsync(stream.fileno())
        temporary.replace(path)
    summary = {'scope': plan['scope'], 'plan_hash': plan['plan_hash'], 'billing_status': BILLING_STATUS,
               'requested': len(plan['requests']), **describe_records(records),
               'by_format': {fmt: describe_records([r for r in records if r['prompt_format'] == fmt])
                             for fmt in FORMATS},
               'quota_status': status, 'earliest_resume_utc': earliest,
               'rolling_tokens_used': sum(tokens for _, tokens in rolling_usage(
                   complete, current, plan['config']['quota']['window_seconds'])),
               'rolling_token_limit': plan['config']['quota']['rolling_tokens'],
               'cpl_status': 'pending separate engine scoring', 'timestamp': current.isoformat()}
    pilot.atomic_json(output_dir / 'summary.json', summary)
    return summary


def submit_one(plan, job, output_dir, key, transport, now, reserved_at):
    ledger = output_dir / 'ledger.jsonl'
    common = {'id': job['id'], 'plan_hash': plan['plan_hash']}
    pilot.append_event(ledger, {**common, 'event': 'reserved', 'timestamp': reserved_at.isoformat(),
                               'reserved_tokens': job['reserved_tokens']})
    started = time.monotonic()
    try:
        response = transport(job['body'], key, plan['config']['openai']['timeout_seconds'])
        response = json.loads(json.dumps(response), parse_constant=lambda value: {'invalid_number': value})
    except Exception:
        raise CampaignError('API request failed; outstanding reservation requires manual review') from None
    completed_at = checked_time(now())
    clock_bad = completed_at < reserved_at
    received = {**common, 'event': 'received', 'response': response,
                'timestamp': max(completed_at, reserved_at).isoformat(),
                'inference_ms': round((time.monotonic() - started) * 1000)}
    pilot.append_event(ledger, received)
    if clock_bad:
        raise CampaignError('Clock rollback during request; outstanding reservation needs review')
    usage = pilot.usage_counts(response)
    if usage['input_tokens'] + usage['output_tokens'] > job['reserved_tokens']:
        raise CampaignError('Response usage exceeded reservation; manual quota review required')
    scored_record(job, received)
    pilot.append_event(ledger, {**common, 'event': 'completed', 'timestamp': completed_at.isoformat()})
    return {**received, 'completed_at': completed_at.isoformat()}


def execute_plan(plan, output_dir, key, transport=None, now=utc_now,
                 confirm_free_project=False, exclusive_quota_use=False, max_new_requests=None):
    if not confirm_free_project or not exclusive_quota_use:
        raise CampaignError('Confirm free-project eligibility and exclusive shared-pool use first')
    if max_new_requests is not None and not pilot.positive(max_new_requests, integer=True):
        raise CampaignError('Maximum new requests must be a positive integer')
    output_dir, transport = Path(output_dir), transport or http_transport
    with execution_lock(output_dir):
        current = checked_time(now())
        complete = prepare_execution(plan, output_dir, current)
        pilot.append_event(output_dir / 'ledger.jsonl', {'event': 'checkpoint',
                           'plan_hash': plan['plan_hash'], 'timestamp': current.isoformat()})
        submitted = 0
        for job in plan['requests']:
            if job['id'] in complete:
                continue
            new_time = checked_time(now())
            check_clock(new_time, current)
            current = new_time
            allowed, _, earliest = quota_decision(plan, complete, current, job['reserved_tokens'])
            if not allowed:
                return write_exports(plan, output_dir, complete, current, 'quota_exhausted', earliest)
            if max_new_requests is not None and submitted >= max_new_requests:
                return write_exports(plan, output_dir, complete, current, 'invocation_limit', None)
            complete[job['id']] = submit_one(plan, job, output_dir, key, transport, now, current)
            current = event_time({'timestamp': complete[job['id']]['completed_at']})
            submitted += 1
            write_exports(plan, output_dir, complete, current, 'running', None)
        return write_exports(plan, output_dir, complete, current, 'complete', None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-free-project', action='store_true')
    parser.add_argument('--exclusive-quota-use', action='store_true')
    parser.add_argument('--max-new-requests', type=int)
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        plan = build_plan(config)
        if not args.execute:
            print(json.dumps({'mode': 'offline only', 'requests': len(plan['requests']),
                              'model': MODEL, 'billing_status': BILLING_STATUS,
                              'rolling_token_limit': config['quota']['rolling_tokens'],
                              'plan_hash': plan['plan_hash']}, indent=2))
            return 0
        if not args.confirm_free_project or not args.exclusive_quota_use:
            raise CampaignError('Execution requires both project eligibility and exclusive-pool confirmations')
        if args.max_new_requests is not None and not pilot.positive(args.max_new_requests, integer=True):
            raise CampaignError('Maximum new requests must be positive')
        key = load_key(config['openai']['key_file'])
        summary = execute_plan(plan, Path(config['paths']['output_dir']), key,
                               confirm_free_project=True, exclusive_quota_use=True,
                               max_new_requests=args.max_new_requests)
        print(json.dumps(summary, indent=2))
        return 0
    except (CampaignError, OSError, yaml.YAMLError):
        print('Campaign stopped: configuration, eligibility, quota, clock, transport or ledger needs review.',
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
