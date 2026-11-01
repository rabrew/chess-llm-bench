"""Local-only report refresh and scored copies; specs/local-report-refresh.md.

Each command defaults to a read-only plan. --execute performs the selected
CPU action; score-snapshot --execute explicitly starts local Lc0 after checks.
No API credentials are loaded and no inference requests are submitted.
"""
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import chess
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import enrich_cpl  # noqa: E402
from scripts import openai_daily_benchmark as daily  # noqa: E402
from scripts import openai_pilot as pilot  # noqa: E402
from src import metrics, provenance  # noqa: E402
from src.utils import load_config  # noqa: E402

CHECKPOINT_FILES = ('plan.json', 'ledger.jsonl', 'responses.jsonl', 'evaluations.jsonl', 'summary.json')
ENRICHMENT_FIELDS = frozenset(('t2_cpl', 't2_best_move', 't2_eval_before', 't2_eval_nodes'))
BASE_COVERAGE = (526_662, 22, 4000)
LC0_NODES = 800
GPU_POWER_CEILING_WATTS = 250
NUMERIC_REL_TOL = 1e-10
NUMERIC_ABS_TOL = 1e-12
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
GPU_QUERY_TIMEOUT_SECONDS = 10
CODE_FILES = ('scripts/local_postprocess.py', 'scripts/enrich_cpl.py', 'src/metrics.py', 'src/utils.py')
PRIMARY_SCALARS = ('t1_mean_error', 't1_mean_error_excl_mate', 't1_relative_error',
                   't1_direction_accuracy_t50', 't2_legal_rate', 't2_mean_cpl_raw',
                   't2_mean_cpl_clamped', 't2_mean_wp_loss', 't3_mean_score', 't3_mean_score_v2')
PRIMARY_GROUPS = {'t1_headline': ('value', 'n'),
                  't1_zero_baseline': ('n', 'mae', 'zero_mae', 'mae_ratio'),
                  't2_blunder_rate': ('n', 'rate'),
                  't2_best_move_vs_random': ('n', 'best_move_rate', 'random_best_move_rate', 'ratio_vs_random'),
                  't3_side_mark': ('n', 'accuracy', 'majority_share', 'cohen_kappa')}


class PostprocessError(Exception):
    """Input/state safety checks failed before canonical research outputs change."""


def strict_read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeError):
        raise PostprocessError('Missing or malformed JSON input') from None


def normalized_json(value):
    if isinstance(value, dict):
        return {str(key): normalized_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    pilot.atomic_json(Path(path), normalized_json(value))


def new_output(path, protected):
    path = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise PostprocessError('Output path overlaps a protected input')
    if path.exists():
        raise PostprocessError('Output already exists; use a new isolated staging directory')
    return path


def file_hashes(directory):
    return {name: provenance.file_sha256(Path(directory) / name) for name in CHECKPOINT_FILES}


def code_hashes():
    return {name: provenance.file_sha256(ROOT / name) for name in CODE_FILES}


def coverage(frame):
    return {'records': int(len(frame)), 'models': int(frame['model'].nunique()),
            'positions': int(frame['position_id'].nunique())}


def checked_point(old, new, label):
    if old is None or new is None:
        if old is new:
            return
        raise PostprocessError(f'Primary point availability changed: {label}')
    if (not isinstance(old, (int, float)) or not isinstance(new, (int, float))
            or not math.isfinite(old) or not math.isfinite(new)
            or not math.isclose(old, new, rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)):
        raise PostprocessError(f'Primary point changed or is nonfinite: {label}')


def validate_primary_points(legacy, fresh):
    comparisons = {}
    try:
        old, new = legacy['overall_metrics'], fresh['overall_metrics']
        for key in PRIMARY_SCALARS:
            checked_point(old[key], new[key], key)
            comparisons[key] = {'legacy': old[key], 'fresh': new[key]}
        for group, names in PRIMARY_GROUPS.items():
            for name in names:
                checked_point(old[group][name], new[group][name], f'{group}.{name}')
                comparisons[f'{group}.{name}'] = {'legacy': old[group][name], 'fresh': new[group][name]}
    except (KeyError, TypeError):
        raise PostprocessError('Primary report fields are missing') from None
    return {'all_primary_points_unchanged': True, 'compared': comparisons,
            'excluded_legacy_failure_measures': ['t2_legal_rate_buggy_includes_no_move_prompts',
                                                'hallucination_rate_all_formats'],
            'legacy_blunder_uncertainty': legacy['overall_metrics']['t2_blunder_rate'],
            'fresh_blunder_uncertainty': fresh['overall_metrics']['t2_blunder_rate']}


def validate_model_points(legacy_dir, fresh_dir):
    name = 't2_blunder_rate.csv'
    old_path, new_path = Path(legacy_dir) / name, Path(fresh_dir) / name
    if not old_path.exists():
        return {'status': 'legacy model blunder CSV absent'}
    old, new = pd.read_csv(old_path).set_index('model'), pd.read_csv(new_path).set_index('model')
    if old.index.has_duplicates or new.index.has_duplicates or set(old.index) != set(new.index):
        raise PostprocessError('Model coverage changed in blunder table')
    for model in old.index:
        for key in ('n', 'blunder_rate', 'mean_cpl_clamped', 'median_cpl_clamped'):
            checked_point(float(old.loc[model, key]), float(new.loc[model, key]), f'{model}.{key}')
    return {'status': 'unchanged', 'models': len(old)}


def refresh_base_report(config_path, legacy_dir, output, expected=BASE_COVERAGE):
    config_path, legacy_dir = Path(config_path), Path(legacy_dir)
    config = load_config(str(config_path))
    source = Path(config['paths']['results_file'])
    output = new_output(output, [source, config_path, legacy_dir, config['paths']['data_dir']])
    before_hash = provenance.file_sha256(source)
    legacy_hash = provenance.file_sha256(legacy_dir / 'summary.json')
    legacy = strict_read(legacy_dir / 'summary.json')
    frame = metrics.load_results_df(str(source), data_dir=config['paths']['data_dir'])
    observed = coverage(frame)
    if tuple(observed.values()) != tuple(expected):
        raise PostprocessError('Base response/model/position coverage differs from expected cohort')
    output.mkdir()
    details = provenance.build_provenance(config_path, source, ROOT)
    details['config_sha256'] = provenance.file_sha256(config_path)
    details['analysis_code_sha256'] = code_hashes()
    metrics.save_metrics(frame, str(output / 'metrics'), provenance=details)
    fresh = strict_read(output / 'metrics/summary.json')
    invariants = validate_primary_points(legacy, fresh)
    model_points = validate_model_points(legacy_dir, output / 'metrics')
    if before_hash != provenance.file_sha256(source) or legacy_hash != provenance.file_sha256(legacy_dir / 'summary.json'):
        raise PostprocessError('An input changed during CPU reporting')
    report = {'status': 'validated', 'coverage': observed, 'invariants': invariants,
              'model_point_estimates': model_points, 'source_sha256': before_hash,
              'legacy_summary_sha256': legacy_hash, 'provenance': details,
              'promotion_status': 'not promoted; canonical reports remain unchanged'}
    write_json(output / 'metrics/summary.json', fresh)
    write_json(output / 'validation.json', report)
    return report


@contextmanager
def checkpoint_lock(source):
    try:
        stream = (Path(source) / '.lock').open('rb')
    except OSError:
        raise PostprocessError('Existing campaign lock file is required') from None
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PostprocessError('Campaign writer is active; snapshot refused') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def decoded_rows(raw):
    try:
        if raw and not raw.endswith(b'\n'):
            raise PostprocessError('Incomplete JSONL checkpoint')
        rows = [json.loads(line) for line in raw.splitlines()]
        if not all(isinstance(row, dict) for row in rows):
            raise PostprocessError('Checkpoint rows must be objects')
        return rows
    except (ValueError, UnicodeError):
        raise PostprocessError('Malformed JSONL checkpoint') from None


def validated_checkpoint(source):
    try:
        raw = {name: (source / name).read_bytes() for name in CHECKPOINT_FILES}
        plan, summary = json.loads(raw['plan.json']), json.loads(raw['summary.json'])
        if not raw['ledger.jsonl']:
            raise PostprocessError('Checkpoint ledger is empty')
        content = {key: value for key, value in plan.items() if key not in ('code_revision', 'plan_hash')}
        if pilot.digest(content) != plan['plan_hash']:
            raise PostprocessError('Checkpoint plan hash does not match its contents')
        if 'quota' in plan['config']:
            complete, _ = daily.replay_ledger(plan, source / 'ledger.jsonl', datetime.now(timezone.utc))
            constructor = daily.scored_record
            if summary.get('quota_status') not in ('complete', 'quota_exhausted', 'invocation_limit'):
                raise PostprocessError('Campaign is not at a stopped checkpoint')
        else:
            complete = pilot.ledger_state(plan, source / 'ledger.jsonl')
            constructor = pilot.scored_record
            if len(complete) != len(plan['requests']):
                raise PostprocessError('Pilot is not complete')
        jobs = [job for job in plan['requests'] if job['id'] in complete]
        expected_raw = [complete[job['id']] for job in jobs]
        expected_records = [constructor(job, complete[job['id']]) for job in jobs]
        if decoded_rows(raw['responses.jsonl']) != expected_raw or decoded_rows(raw['evaluations.jsonl']) != expected_records:
            raise PostprocessError('Ledger and response/evaluation exports disagree')
        if summary['completed'] != len(jobs) or summary['requested'] != len(plan['requests']):
            raise PostprocessError('Summary and ledger coverage disagree')
        if summary.get('plan_hash') != plan['plan_hash']:
            raise PostprocessError('Summary plan identity mismatch')
        return raw, plan
    except (OSError, ValueError, KeyError, TypeError, pilot.PilotError):
        raise PostprocessError('Invalid or inconsistent campaign checkpoint') from None


def snapshot_checkpoint(source, output):
    source = Path(source).resolve()
    output = new_output(output, [source])
    with checkpoint_lock(source):
        raw, plan = validated_checkpoint(source)
        hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
        (output / 'original').mkdir(parents=True)
        for name, value in raw.items():
            (output / 'original' / name).write_bytes(value)
        (output / 'evaluations_scored.jsonl').write_bytes(raw['evaluations.jsonl'])
        if file_hashes(output / 'original') != hashes or file_hashes(source) != hashes:
            raise PostprocessError('Snapshot copy/input hash validation failed')
    report = {'status': 'snapshot_created', 'source': str(source), 'source_hashes': hashes,
              'source_plan_hash': plan['plan_hash'], 'scope': 'partial first-answer checkpoint; not a completed experiment',
              'records': len(decoded_rows(raw['evaluations.jsonl'])),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'code_revision': pilot.code_revision(), 'analysis_code_sha256': code_hashes(),
              'billing_status': 'not verified; no API work performed'}
    write_json(output / 'snapshot.json', report)
    return report


def validate_enrichment(original, enriched):
    if len(original) != len(enriched):
        raise PostprocessError('Enrichment changed record count')
    ids = [row.get('job_id') for row in original]
    if None in ids or len(set(ids)) != len(ids):
        raise PostprocessError('Snapshot has missing/duplicate job identities')
    for before, after in zip(original, enriched):
        payload_before = {key: value for key, value in before.items() if key not in ENRICHMENT_FIELDS}
        payload_after = {key: value for key, value in after.items() if key not in ENRICHMENT_FIELDS}
        if payload_before != payload_after:
            raise PostprocessError('Enrichment changed original payload/identity/order')
    return True


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clamped_cpl(row):
    if row.get('t2_legal') is not True or not finite(row.get('t2_cpl')) or row['t2_cpl'] < 0:
        return None
    before = row.get('t2_eval_before')
    if not finite(before) or row.get('t2_eval_nodes') != LC0_NODES:
        return None
    try:
        white = chess.Board(row['fen']).turn == chess.WHITE
        after = before - row['t2_cpl'] if white else before + row['t2_cpl']
        return metrics.compute_clamped_cpl(before, after, white)
    except (KeyError, ValueError):
        return None


def descriptive_group(rows):
    cpls = [value for row in rows if (value := clamped_cpl(row)) is not None]
    known = [row['t2_first_attempt_legal'] for row in rows
             if row.get('t2_first_attempt_legal') in (True, False, 0, 1)]
    successes = sum(value < metrics.BLUNDER_CPL_CP for value in cpls)
    return {'requests': len(rows), 'positions': len({row.get('position_id') for row in rows}),
            'legal_moves': sum(row.get('t2_legal') is True for row in rows),
            'first_answer_known': len(known), 'first_answer_legal_rate': sum(known) / len(known) if known else None,
            'legal_scored': len(cpls), 'blunder_rate_legal_scored':
                sum(value >= metrics.BLUNDER_CPL_CP for value in cpls) / len(cpls) if cpls else None,
            'legal_nonblunder_successes': successes,
            'legal_nonblunder_per_request': successes / len(rows) if rows else None}


def snapshot_descriptive(rows):
    models = sorted({row['model'] for row in rows})
    formats = sorted({row['prompt_format'] for row in rows})
    return {'all_requests': descriptive_group(rows),
            'by_model': {model: descriptive_group([r for r in rows if r['model'] == model]) for model in models},
            'by_format': {fmt: descriptive_group([r for r in rows if r['prompt_format'] == fmt]) for fmt in formats},
            'by_model_format': {model: {fmt: descriptive_group([r for r in rows
                if r['model'] == model and r['prompt_format'] == fmt]) for fmt in formats} for model in models},
            'interpretation': 'descriptive partial first answers; no superiority or reasoning verdict'}


def ensure_gpu_idle(config):
    try:
        enrich_cpl.ensure_ollama_unloaded(config.get('ollama', {}).get('base_url', DEFAULT_OLLAMA_URL))
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=GPU_QUERY_TIMEOUT_SECONDS)
        power = subprocess.run(['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True, timeout=GPU_QUERY_TIMEOUT_SECONDS)
        if query.stdout.strip() or not power.stdout.strip():
            raise PostprocessError('GPU has compute processes or unknown power state')
        if any(not finite(float(line)) or not 0 < float(line) <= GPU_POWER_CEILING_WATTS
               for line in power.stdout.splitlines()):
            raise PostprocessError('GPU power limit is above the retained 250 W ceiling')
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        raise PostprocessError('Cannot verify GPU idle/power state; scoring refused') from None


def scoring_config(path):
    config = load_config(str(path))
    engine = config.get('lc0', {})
    if (not isinstance(engine.get('binary'), str) or not engine['binary']
            or not isinstance(engine.get('weights'), str) or not engine['weights']
            or engine.get('nodes') != LC0_NODES
            or not pilot.positive(engine.get('timeout', enrich_cpl.DEFAULT_PROTOCOL_TIMEOUT))):
        raise PostprocessError('Scoring requires configured existing Lc0 binary/weights and exactly 800 nodes')
    return config


def snapshot_inputs(snapshot):
    manifest = strict_read(snapshot / 'snapshot.json')
    if file_hashes(snapshot / 'original') != manifest['source_hashes']:
        raise PostprocessError('Immutable snapshot source hashes changed')
    original = decoded_rows((snapshot / 'original/evaluations.jsonl').read_bytes())
    current = decoded_rows((snapshot / 'evaluations_scored.jsonl').read_bytes())
    validate_enrichment(original, current)
    return original, current, manifest


def score_snapshot(snapshot, config_path, guard=None, engine_factory=None):
    snapshot, config_path = Path(snapshot), Path(config_path)
    config = scoring_config(config_path)
    original, current, manifest = snapshot_inputs(snapshot)
    needed = [row for row in current if row.get('t2_legal') is True and row.get('t2_cpl') is None]
    if needed:
        (guard or ensure_gpu_idle)(config)
        lc0 = config['lc0']
        factory = engine_factory or enrich_cpl.Lc0Engine
        with factory(path=lc0['binary'], weights=lc0['weights'], nodes=LC0_NODES,
                     timeout=lc0.get('timeout', enrich_cpl.DEFAULT_PROTOCOL_TIMEOUT)) as engine:
            enriched = enrich_cpl.enrich_records(copy.deepcopy(needed), engine)
        changed = {row['job_id']: row for row in enriched}
        current = [changed.get(row['job_id'], row) for row in current]
    validate_enrichment(original, current)
    descriptive = snapshot_descriptive(current)
    total = descriptive['all_requests']
    missing = total['legal_moves'] - total['legal_scored']
    output = snapshot / 'evaluations_scored.jsonl'
    temporary = output.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(pilot.canonical(row) + '\n' for row in current))
    temporary.replace(output)
    report = {'status': 'complete' if missing == 0 else 'pending', 'legal_unscored': missing,
              'payload_invariants_unchanged': True, 'descriptive': descriptive,
              'source_plan_hash': manifest['source_plan_hash'], 'source_hashes': manifest['source_hashes'],
              'scored_sha256': provenance.file_sha256(output),
              'provenance': {**provenance.build_provenance(config_path, output, ROOT),
                             'config_sha256': provenance.file_sha256(config_path),
                             'analysis_code_sha256': code_hashes()}}
    write_json(snapshot / 'scoring_report.json', report)
    return report


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    base = commands.add_parser('base-report')
    base.add_argument('--config', default='config/config.yaml')
    base.add_argument('--legacy-metrics', default='results/metrics')
    base.add_argument('--output', required=True)
    snap = commands.add_parser('snapshot')
    snap.add_argument('--source', required=True)
    snap.add_argument('--output', required=True)
    score = commands.add_parser('score-snapshot')
    score.add_argument('--snapshot', required=True)
    score.add_argument('--config', default='config/config.yaml')
    for command in (base, snap, score):
        command.add_argument('--execute', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({'mode': 'read-only plan; no work executed', **vars(args)}, indent=2))
        return 0
    try:
        if args.action == 'base-report':
            report = refresh_base_report(args.config, args.legacy_metrics, args.output)
        elif args.action == 'snapshot':
            report = snapshot_checkpoint(args.source, args.output)
        else:
            report = score_snapshot(args.snapshot, args.config)
        print(json.dumps(normalized_json(report), indent=2, allow_nan=False))
        return 0
    except (PostprocessError, OSError, KeyError, ValueError) as error:
        print(f'Local processing stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
