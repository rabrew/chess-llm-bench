"""Synthetic offline contracts for the rolling-quota GPT-5.4 campaign."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import Mock

import chess
import pytest
import yaml

from scripts import openai_daily_benchmark as daily

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def config(tmp_path):
    rows = [{'position_id': i, 'difficulty': tier, 'fen': chess.STARTING_FEN,
             'stockfish_eval': 0, 'stockfish_best_move': 'e4', 'theme': 'advantage'}
            for i, tier in enumerate(daily.TIERS)]
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows * 2))
    return {'openai': {'model': 'gpt-5.4', 'key_file': str(tmp_path / 'private.env'),
                       'max_output_tokens': 4000, 'timeout_seconds': 120,
                       'prices_per_mtok': {'input': 2.50, 'output': 15.0}},
            'quota': {'rolling_tokens': 200000, 'window_seconds': 86400},
            'benchmark': {'positions_per_tier': 1, 'random_seed': 42,
                          'prompt_formats': list(daily.FORMATS)},
            'paths': {'source_results': str(source), 'output_dir': str(tmp_path / 'out')}}


def reply(text='Eval: 0\nMove: e4\nExplanation: Equal.', usage=None, **changes):
    value = {'id': 'resp_fixture', 'model': 'gpt-5.4', 'status': 'completed',
             'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
             'usage': usage if usage is not None else {'input_tokens': 300, 'output_tokens': 100,
                     'output_tokens_details': {'reasoning_tokens': 50}}}
    value.update(changes)
    return value


def run(plan, config, transport=None, when=NOW, **kw):
    return daily.execute_plan(plan, Path(config['paths']['output_dir']), 'dummy-secret',
                              transport=transport or Mock(return_value=reply()), now=lambda: when,
                              confirm_free_project=True, exclusive_quota_use=True, **kw)


def test_fixed_balanced_interleaved_three_format_plan(config):
    original = Path(config['paths']['source_results']).read_bytes()
    plan = daily.build_plan(config)
    assert plan == daily.build_plan(config)
    assert len(plan['requests']) == 12
    assert [r['position']['difficulty'] for r in plan['requests'][::3]] == list(daily.TIERS)
    assert [r['format'] for r in plan['requests'][:3]] == list(daily.FORMATS)
    assert all(r['body']['tools'] == [] and r['body']['store'] is False for r in plan['requests'])
    assert all('temperature' not in r['body'] and 'reasoning' not in r['body'] for r in plan['requests'])
    assert Path(config['paths']['source_results']).read_bytes() == original
    assert not Path(config['paths']['output_dir']).exists()


@pytest.mark.parametrize('section,key,value', [('openai', 'model', 'gpt-6-luna'),
    ('openai', 'timeout_seconds', float('nan')), ('openai', 'max_output_tokens', True),
    ('openai', 'max_output_tokens', 4001), ('quota', 'rolling_tokens', 200001),
    ('quota', 'rolling_tokens', 0), ('quota', 'window_seconds', 3600),
    ('benchmark', 'positions_per_tier', 0), ('benchmark', 'random_seed', True),
    ('benchmark', 'prompt_formats', ['fen_only'])])
def test_invalid_config_rejected_before_effects(config, section, key, value):
    config[section][key] = value
    with pytest.raises(daily.CampaignError):
        daily.build_plan(config)
    assert not Path(config['paths']['output_dir']).exists()


def test_unknown_price_rejected(config):
    config['openai']['prices_per_mtok']['input'] = 0.01
    with pytest.raises(daily.CampaignError):
        daily.build_plan(config)


def test_extra_or_missing_position_rejected(config):
    source = Path(config['paths']['source_results'])
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows.append(dict(rows[0], position_id=100))
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(daily.CampaignError, match='cohort'):
        daily.build_plan(config)


def test_default_dryrun_no_key_network_writes(config, tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    key = Mock(side_effect=AssertionError('key read'))
    transport = Mock(side_effect=AssertionError('network'))
    monkeypatch.setattr(daily, 'load_key', key)
    monkeypatch.setattr(daily, 'http_transport', transport)
    assert daily.main(['--config', str(path)]) == 0
    assert '12' in capsys.readouterr().out
    assert not key.called and not transport.called
    assert not Path(config['paths']['output_dir']).exists()


@pytest.mark.parametrize('flags', [[], ['--confirm-free-project'], ['--exclusive-quota-use']])
def test_execute_requires_both_confirmation_flags(config, tmp_path, monkeypatch, flags):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    key = Mock(side_effect=AssertionError('key read'))
    monkeypatch.setattr(daily, 'load_key', key)
    assert daily.main(['--config', str(path), '--execute', *flags]) != 0
    assert not key.called and not Path(config['paths']['output_dir']).exists()


def test_limited_then_resumed_run_does_not_resend(config):
    plan, transport = daily.build_plan(config), Mock(return_value=reply())
    first = run(plan, config, transport, max_new_requests=1)
    assert first['completed'] == 1 and first['quota_status'] == 'invocation_limit'
    second = run(plan, config, transport)
    assert second['completed'] == 12 and second['quota_status'] == 'complete'
    assert transport.call_count == 12
    assert second['input_tokens'] == 3600 and second['output_tokens'] == 1200
    assert second['rolling_tokens_used'] == 4800  # Reasoning already included in output.
    assert second['reasoning_tokens'] == 600
    assert second['billing_status'] != 'free'
    cached = Mock(side_effect=AssertionError('duplicate request'))
    assert run(plan, config, cached)['completed'] == 12
    assert not cached.called


def test_quota_stops_before_oversize_request(config):
    config['quota']['rolling_tokens'] = 1
    plan, transport = daily.build_plan(config), Mock()
    summary = run(plan, config, transport)
    assert summary['quota_status'] == 'quota_exhausted' and summary['completed'] == 0
    assert not transport.called
    assert summary['earliest_resume_utc'] is None  # Request can never fit this quota.


def test_rolling_quota_release_at_exact_completion_boundary(config):
    config['quota']['rolling_tokens'] = 10000
    plan = daily.build_plan(config)
    # Large actual usage plus full next reservation cannot fit the rolling pool.
    transport = Mock(return_value=reply(usage={'input_tokens': 1000, 'output_tokens': 3000}))
    first = run(plan, config, transport)
    assert first['completed'] == 1 and first['quota_status'] == 'quota_exhausted'
    assert first['earliest_resume_utc'] == (NOW + timedelta(days=1)).isoformat()
    assert run(plan, config, Mock(), when=NOW + timedelta(days=1, seconds=-1))['completed'] == 1
    after = run(plan, config, transport, when=NOW + timedelta(days=1), max_new_requests=1)
    assert after['completed'] == 2 and after['rolling_tokens_used'] == 4000


def test_clock_rollback_or_future_events_rejected(config):
    plan = daily.build_plan(config)
    run(plan, config, max_new_requests=1)
    transport = Mock(side_effect=AssertionError('must not submit'))
    with pytest.raises(daily.CampaignError, match='clock|future'):
        run(plan, config, transport, when=NOW - timedelta(seconds=1))
    assert not transport.called


def test_first_answer_refused_capped_missing_retained(config):
    plan = daily.build_plan(config)
    transport = Mock(side_effect=[reply('Move: e5\nExplanation: e4 might work.'),
        reply('', status='incomplete', incomplete_details={'reason': 'max_output_tokens'}),
        reply('', output=[{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]}])])
    summary = run(plan, config, transport, max_new_requests=3)
    assert summary['completed'] == 3 and summary['capped'] == 1 and summary['refused'] == 1
    assert summary['first_answer_legal'] == 0
    rows = [json.loads(line) for line in (Path(config['paths']['output_dir']) / 'evaluations.jsonl').read_text().splitlines()]
    assert all(row['model_family'] == 'openai' and row['model_size_b'] is None for row in rows)
    assert all(row['t2_cpl'] is None for row in rows)
    assert summary['by_format']['fen_only']['completed'] == 1


@pytest.mark.parametrize('loss', ['missing', 'empty', 'valid_prefix'])
def test_lost_ledger_blocks_resubmission_without_export_loss(config, loss):
    plan = daily.build_plan(config)
    run(plan, config, max_new_requests=2)
    out = Path(config['paths']['output_dir'])
    before = {name: (out / name).read_bytes() for name in ('plan.json', 'responses.jsonl', 'evaluations.jsonl')}
    ledger = out / 'ledger.jsonl'
    if loss == 'missing':
        ledger.unlink()
    elif loss == 'empty':
        ledger.write_text('')
    else:
        ledger.write_text('\n'.join(ledger.read_text().splitlines()[:4]) + '\n')
    transport = Mock(side_effect=AssertionError('duplicate paid request'))
    with pytest.raises(daily.CampaignError):
        run(plan, config, transport)
    assert not transport.called
    assert before == {name: (out / name).read_bytes() for name in before}


@pytest.mark.parametrize('fault', ['timeout', 'usage', 'duplicate', 'malformed'])
def test_ambiguous_corrupt_and_unknown_usage_fail_closed(config, fault):
    plan, out = daily.build_plan(config), Path(config['paths']['output_dir'])
    if fault in ('timeout', 'usage'):
        transport = (Mock(side_effect=TimeoutError('dummy-secret')) if fault == 'timeout'
                     else Mock(return_value=reply(usage={})))
        with pytest.raises(daily.CampaignError) as error:
            run(plan, config, transport)
        assert 'dummy-secret' not in str(error.value)
    else:
        run(plan, config, max_new_requests=1)
        ledger = out / 'ledger.jsonl'
        extra = ledger.read_text().splitlines()[1] + '\n' if fault == 'duplicate' else '{'
        ledger.write_text(ledger.read_text() + extra)
    transport = Mock(side_effect=AssertionError('unsafe resubmit'))
    with pytest.raises(daily.CampaignError):
        run(plan, config, transport)
    assert not transport.called


def test_process_lock(config):
    plan = daily.build_plan(config)
    with daily.execution_lock(Path(config['paths']['output_dir'])):
        with pytest.raises(daily.CampaignError, match='running'):
            run(plan, config)
