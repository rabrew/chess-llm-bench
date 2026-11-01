"""Temporary-fixture regression tests for local-only report refresh/scoring."""
from datetime import datetime, timezone
import json
from unittest.mock import Mock

import chess
import pytest
import yaml

from scripts import local_postprocess as local
from scripts import openai_daily_benchmark as daily
from scripts import openai_pilot as pilot
from src import metrics
from src.evaluator import score_all
from src.result_writer import build_result_record
from src.utils import parse_model_info

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def base_record(identity=1):
    position = {'position_id': identity, 'fen': chess.STARTING_FEN, 'stockfish_eval': identity * 50,
                'stockfish_best_move': 'e4', 'difficulty': daily.TIERS[(identity - 1) % 4],
                'phase': 'opening', 'theme': 'advantage', 'source': 'fixture'}
    parsed = {'eval': identity * 25, 'move': 'e4', 'explanation': 'Equal.', 'side_claimed': 'Equal'}
    job = dict(position, job_id=f'j{identity}', model='llama3.2:3b', prompt_format='fen_only')
    result = build_result_record(job, parsed, score_all(parsed, position), 1)
    result.update(t2_cpl=100, t2_eval_before=200, t2_eval_nodes=800)
    return result


@pytest.mark.parametrize('tag', ['gpt-5.4', 'gpt-5.4-2026-03-05', 'gpt-6.1-sol', 'gpt-6-luna'])
def test_exact_openai_metadata_no_guessed_parameter_count(tag):
    info = parse_model_info(tag)
    assert info['family'] == 'openai' and info['size_b'] is None and info['active_b'] is None


def test_unknown_still_rejected_and_claude_preserved():
    with pytest.raises(ValueError):
        parse_model_info('gpt-invented')
    assert parse_model_info('claude-haiku:5.5')['family'] == 'claude'


def test_fen_complete_skips_tier_loader_but_preserves_metadata_join(tmp_path, monkeypatch):
    source = tmp_path / 'results.jsonl'
    write_rows(source, [base_record()])
    (tmp_path / 'puzzle_meta.json').write_text(json.dumps({'positions': {
        chess.STARTING_FEN: {'themes_all': ['fork'], 'setup_move_san': 'e4'}}}))
    lookup = Mock(side_effect=AssertionError('massive dataset lookup'))
    monkeypatch.setattr(metrics, '_build_fen_lookup', lookup)
    result = metrics.load_results_df(str(source), data_dir=str(tmp_path))
    assert not lookup.called and result.iloc[0]['setup_move_san'] == 'e4'
    assert result.iloc[0]['t2_cpl_clamped'] == 100


def test_partial_fen_only_missing_rows_fall_back(tmp_path, monkeypatch):
    rows = [base_record(1), base_record(2), base_record(3)]
    rows[1]['fen'] = None
    rows[2]['fen'] = ''
    source = tmp_path / 'results.jsonl'
    write_rows(source, rows)
    black = chess.STARTING_FEN.replace(' w ', ' b ')
    lookup = Mock(return_value={1: black, 2: black})
    monkeypatch.setattr(metrics, '_build_fen_lookup', lookup)
    result = metrics.load_results_df(str(source), data_dir=str(tmp_path))
    assert lookup.call_count == 1
    assert result.iloc[0]['fen'] == chess.STARTING_FEN  # Existing self-contained FEN wins.
    assert result.iloc[1]['fen'] == black
    assert result.iloc[2]['t2_cpl_clamped'] != result.iloc[2]['t2_cpl_clamped']


def test_absent_fen_column_uses_lookup(tmp_path, monkeypatch):
    row = base_record()
    row.pop('fen')
    source = tmp_path / 'source.jsonl'
    write_rows(source, [row])
    monkeypatch.setattr(metrics, '_build_fen_lookup', Mock(return_value={1: chess.STARTING_FEN}))
    result = metrics.load_results_df(str(source), data_dir=str(tmp_path))
    assert result.iloc[0]['t2_cpl_clamped'] == 100


@pytest.fixture
def base_inputs(tmp_path):
    source, cfg, legacy = tmp_path / 'source.jsonl', tmp_path / 'config.yaml', tmp_path / 'legacy'
    legacy.mkdir()
    write_rows(source, [base_record(i) for i in range(1, 5)])
    cfg.write_text(yaml.safe_dump({'paths': {'results_file': str(source), 'data_dir': str(tmp_path / 'data'),
                                           'metrics_dir': str(legacy)},
                                   'lc0': {'binary': '/unused/lc0', 'weights': '/unused/weights', 'nodes': 800}}))
    frame = metrics.load_results_df(str(source), str(tmp_path))
    summary = metrics.generate_summary(frame)
    summary['overall_metrics']['t2_blunder_rate']['ci_method'] = 'legacy_iid'
    (legacy / 'summary.json').write_text(json.dumps(summary, default=str))
    return source, cfg, legacy


def test_base_refresh_preserves_inputs_and_checks_primary_points(base_inputs, tmp_path):
    source, cfg, legacy = base_inputs
    before = source.read_bytes(), (legacy / 'summary.json').read_bytes()
    output = tmp_path / 'staging'
    report = local.refresh_base_report(cfg, legacy, output, expected=(4, 1, 4))
    assert report['status'] == 'validated' and report['coverage']['records'] == 4
    assert report['invariants']['all_primary_points_unchanged']
    assert (output / 'metrics/summary.json').is_file()
    assert before == (source.read_bytes(), (legacy / 'summary.json').read_bytes())
    fresh = json.loads((output / 'metrics/summary.json').read_text())
    assert fresh['overall_metrics']['t2_blunder_rate']['ci_method'] == metrics.POSITION_CLUSTER_CI


@pytest.mark.parametrize('fault', ['coverage', 'point', 'nonfinite', 'overlap', 'existing'])
def test_base_refresh_rejects_unsafe_or_inconsistent_work(base_inputs, tmp_path, fault):
    _, cfg, legacy = base_inputs
    output, expected = tmp_path / 'out', (4, 1, 4)
    if fault == 'coverage':
        expected = (5, 1, 4)
    elif fault in ('point', 'nonfinite'):
        summary = json.loads((legacy / 'summary.json').read_text())
        summary['overall_metrics']['t2_blunder_rate']['rate'] = .9 if fault == 'point' else float('nan')
        (legacy / 'summary.json').write_text(json.dumps(summary))
    elif fault == 'overlap':
        output = legacy
    else:
        output.mkdir()
        (output / 'user.txt').write_text('preserve')
    with pytest.raises(local.PostprocessError):
        local.refresh_base_report(cfg, legacy, output, expected=expected)


def api_reply():
    return {'id': 'response_fixture', 'model': 'gpt-5.4', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text',
                        'text': 'Eval: 0\nMove: e4\nExplanation: Equal.'}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 20}}


@pytest.fixture(params=['daily', 'pilot'])
def checkpoint(tmp_path, request):
    source = tmp_path / 'source.jsonl'
    write_rows(source, [base_record(i) for i in range(1, 5)])
    output = tmp_path / 'campaign'
    cfg = {'benchmark': {'positions_per_tier': 1, 'random_seed': 42},
           'paths': {'source_results': str(source), 'output_dir': str(output)}}
    if request.param == 'daily':
        cfg['openai'] = {'model': 'gpt-5.4', 'max_output_tokens': 4000, 'timeout_seconds': 120,
                         'prices_per_mtok': dict(daily.PRICES), 'key_file': str(tmp_path / 'private.env')}
        cfg['benchmark']['prompt_formats'] = list(daily.FORMATS)
        cfg['quota'] = {'rolling_tokens': 200000, 'window_seconds': 86400}
        daily.execute_plan(daily.build_plan(cfg), output, 'fake-test-only', transport=Mock(return_value=api_reply()),
                           now=lambda: NOW, confirm_free_project=True, exclusive_quota_use=True,
                           max_new_requests=3)
    else:
        cfg['openai'] = {'models': ['gpt-6-luna'], 'max_output_tokens': 4000, 'timeout_seconds': 120,
                         'prices_per_mtok': dict(pilot.MODEL_PRICES), 'budget_usd': 2,
                         'key_file': str(tmp_path / 'private.env')}
        cfg['benchmark']['prompt_format'] = 'board_text'
        pilot.execute_plan(pilot.build_plan(cfg), output, 'fake-test-only', Mock(return_value=api_reply()))
    return output


def test_snapshot_consistent_copy_does_not_touch_campaign(checkpoint, tmp_path):
    before = {name: (checkpoint / name).read_bytes() for name in local.CHECKPOINT_FILES}
    target = tmp_path / 'snapshot'
    report = local.snapshot_checkpoint(checkpoint, target)
    assert report['status'] == 'snapshot_created'
    assert before == {name: (checkpoint / name).read_bytes() for name in before}
    assert before == {name: (target / 'original' / name).read_bytes() for name in before}
    assert (target / 'evaluations_scored.jsonl').read_bytes() == before['evaluations.jsonl']


@pytest.mark.parametrize('fault', ['live', 'outstanding', 'export', 'summary', 'missing'])
def test_snapshot_rejects_unsafe_checkpoint(checkpoint, tmp_path, fault):
    target = tmp_path / 'snapshot'
    if fault == 'live':
        with pilot.execution_lock(checkpoint):
            with pytest.raises(local.PostprocessError):
                local.snapshot_checkpoint(checkpoint, target)
        assert not target.exists()
        return
    if fault == 'outstanding':
        events = (checkpoint / 'ledger.jsonl').read_text().splitlines()
        (checkpoint / 'ledger.jsonl').write_text('\n'.join(events[:-1]) + '\n')
    elif fault == 'export':
        rows = [json.loads(line) for line in (checkpoint / 'evaluations.jsonl').read_text().splitlines()]
        rows[0]['raw_response'] = 'changed'
        write_rows(checkpoint / 'evaluations.jsonl', rows)
    elif fault == 'summary':
        summary = json.loads((checkpoint / 'summary.json').read_text())
        summary['completed'] += 1
        (checkpoint / 'summary.json').write_text(json.dumps(summary))
    else:
        (checkpoint / 'ledger.jsonl').unlink()
    with pytest.raises(local.PostprocessError):
        local.snapshot_checkpoint(checkpoint, target)
    assert not target.exists()


class FakeEngine:
    nodes = 800

    def evaluate(self, fen):
        return {'eval': 100, 'best_move': 'e4'}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_snapshot_scoring_fake_engine_and_payload_invariants(checkpoint, tmp_path):
    target = tmp_path / 'snapshot'
    local.snapshot_checkpoint(checkpoint, target)
    original = (target / 'original/evaluations.jsonl').read_bytes()
    config = tmp_path / 'engine.yaml'
    config.write_text(yaml.safe_dump({'lc0': {'binary': '/fake/lc0', 'weights': '/fake/net', 'nodes': 800},
                                     'paths': {'data_dir': str(tmp_path)}}))
    guard, factory = Mock(), Mock(return_value=FakeEngine())
    report = local.score_snapshot(target, config, guard=guard, engine_factory=factory)
    assert guard.call_count == 1 and factory.call_count == 1
    assert report['status'] == 'complete' and report['legal_unscored'] == 0
    assert report['payload_invariants_unchanged']
    assert original == (target / 'original/evaluations.jsonl').read_bytes()
    assert report['descriptive']['all_requests']['legal_nonblunder_per_request'] == 1


def test_payload_invariants_detect_any_unapproved_change():
    original = [dict(base_record(), raw_response='original', input_tokens=10)]
    changed = [dict(original[0], raw_response='changed')]
    with pytest.raises(local.PostprocessError):
        local.validate_enrichment(original, changed)


def test_missing_scores_remain_pending_and_failures_stay_denominator(tmp_path):
    rows = [dict(base_record(), model='gpt-5.4', t2_cpl=None),
            dict(base_record(2), model='gpt-5.4', t2_legal=False, t2_move=None, t2_cpl=None)]
    result = local.snapshot_descriptive(rows)
    assert result['all_requests']['requests'] == 2
    assert result['all_requests']['legal_scored'] == 0
    assert result['all_requests']['blunder_rate_legal_scored'] is None
    assert result['all_requests']['legal_nonblunder_per_request'] == 0
    rows[0]['t2_legal'] = False
    assert local.snapshot_descriptive(rows)['all_requests']['legal_moves'] == 0


def test_cli_default_actions_do_not_mutate_or_contact_services(tmp_path, base_inputs, monkeypatch):
    _, cfg, legacy = base_inputs
    effect = Mock(side_effect=AssertionError('execution attempted'))
    monkeypatch.setattr(local, 'refresh_base_report', effect)
    assert local.main(['base-report', '--config', str(cfg), '--legacy-metrics', str(legacy),
                       '--output', str(tmp_path / 'dry')]) == 0
    assert not effect.called and not (tmp_path / 'dry').exists()
