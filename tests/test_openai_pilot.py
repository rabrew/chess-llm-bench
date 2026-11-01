"""Offline contracts for specs/openai-pilot.md; no real credentials or services."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import chess
import pytest
import yaml

from scripts import openai_pilot as pilot


@pytest.fixture
def setup(tmp_path):
    rows = []
    for tier_no, tier in enumerate(pilot.TIERS):
        for i in range(6):
            rows.append({'position_id': tier_no * 10 + i, 'fen': chess.STARTING_FEN,
                         'difficulty': tier, 'stockfish_eval': 0,
                         'stockfish_best_move': 'e4', 'theme': 'advantage'})
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows * 2))
    cfg = {'openai': {'models': ['gpt-6.1-sol', 'gpt-6-luna'], 'budget_usd': 2,
                      'key_file': str(tmp_path / 'private.env'), 'max_output_tokens': 4000,
                      'timeout_seconds': 120, 'prices_per_mtok': copy.deepcopy(pilot.MODEL_PRICES)},
           'benchmark': {'positions_per_tier': 5, 'random_seed': 42, 'prompt_format': 'board_text'},
           'paths': {'source_results': str(source), 'output_dir': str(tmp_path / 'out')}}
    return cfg, rows


def response(text='Eval: 0\nMove: e4\nExplanation: Equal.', **changes):
    body = {'id': 'resp_test', 'model': 'gpt-6.1-sol', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 400, 'output_tokens': 100,
                      'output_tokens_details': {'reasoning_tokens': 30}}}
    body.update(changes)
    return body


def tiny_plan(cfg):
    cfg['openai']['models'] = ['gpt-6.1-sol']
    cfg['benchmark']['positions_per_tier'] = 1
    return pilot.build_plan(cfg)


def test_balanced_deterministic_sample_and_no_source_writes(setup):
    cfg, _ = setup
    before = Path(cfg['paths']['source_results']).read_bytes()
    a = pilot.build_plan(cfg)
    b = pilot.build_plan(cfg)
    assert a == b
    assert len(a['requests']) == 40
    assert {r['position']['position_id'] for r in a['requests'] if r['model'] == 'gpt-6.1-sol'} == {
        r['position']['position_id'] for r in a['requests'] if r['model'] == 'gpt-6-luna'}
    assert Path(cfg['paths']['source_results']).read_bytes() == before
    assert not Path(cfg['paths']['output_dir']).exists()


def test_request_has_no_tools_sampling_or_hidden_rescue(setup):
    cfg, _ = setup
    p = pilot.build_plan(cfg)
    body = p['requests'][0]['body']
    assert body['tools'] == [] and body['store'] is False
    assert body['max_output_tokens'] == 4000
    assert 'temperature' not in body and 'reasoning' not in body
    assert 'FEN' not in body['input']
    assert 'Legal moves:' not in body['input']


@pytest.mark.parametrize('section,key,value', [
    ('openai', 'budget_usd', float('nan')), ('openai', 'timeout_seconds', float('inf')),
    ('openai', 'max_output_tokens', 0), ('openai', 'max_output_tokens', 1.5),
    ('benchmark', 'positions_per_tier', 0), ('benchmark', 'positions_per_tier', True),
    ('openai', 'models', ['unpriced-model']), ('openai', 'models', []),
    ('openai', 'models', ['gpt-6-luna', 'gpt-6-luna']),
    ('benchmark', 'prompt_format', 'board_text+legal')])
def test_bad_config_fails_without_writes(setup, section, key, value):
    cfg, _ = setup
    cfg[section][key] = value
    with pytest.raises(pilot.PilotError):
        pilot.build_plan(cfg)
    assert not Path(cfg['paths']['output_dir']).exists()


def test_prices_cannot_understate_model_cost(setup):
    cfg, _ = setup
    cfg['openai']['prices_per_mtok']['gpt-6.1-sol']['output'] = 0
    with pytest.raises(pilot.PilotError):
        pilot.build_plan(cfg)


@pytest.mark.parametrize('change', ['missing_tier', 'different_fen', 'different_truth', 'truncated'])
def test_bad_source_is_rejected(setup, change):
    cfg, rows = setup
    if change == 'missing_tier':
        rows = [r for r in rows if r['difficulty'] != 'extreme']
    elif change == 'different_fen':
        rows.append(dict(rows[0], fen=chess.STARTING_FEN.replace(' w ', ' b ')))
    elif change == 'different_truth':
        rows.append(dict(rows[0], stockfish_eval=1))
    raw = ''.join(json.dumps(r) + '\n' for r in rows)
    if change == 'truncated':
        raw += '{"position_id":'
    Path(cfg['paths']['source_results']).write_text(raw)
    with pytest.raises(pilot.PilotError):
        pilot.build_plan(cfg)


def test_dry_run_never_reads_key_transport_or_writes(setup, tmp_path, monkeypatch, capsys):
    cfg, _ = setup
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    key = Mock(side_effect=AssertionError('credential access'))
    network = Mock(side_effect=AssertionError('network'))
    monkeypatch.setattr(pilot, 'load_key', key)
    monkeypatch.setattr(pilot, 'http_transport', network)
    assert pilot.main(['--config', str(cfg_path)]) == 0
    assert '40' in capsys.readouterr().out
    assert not key.called and not network.called
    assert not Path(cfg['paths']['output_dir']).exists()


def test_success_resume_skips_requests_and_preserves_raw(setup):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    transport = Mock(return_value=response())
    summary = pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', transport)
    assert transport.call_count == 4
    assert summary['completed'] == 4 and summary['first_answer_legal'] == 1
    assert summary['output_tokens'] == 400 and summary['reasoning_tokens'] == 120
    again = Mock(side_effect=AssertionError('paid duplicate'))
    assert pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', again) == summary
    raw = (Path(cfg['paths']['output_dir']) / 'responses.jsonl').read_text()
    assert 'resp_test' in raw and 'dummy-secret' not in raw
    records = [json.loads(line) for line in (Path(cfg['paths']['output_dir']) / 'evaluations.jsonl').read_text().splitlines()]
    assert all(r['model_family'] == 'openai' and r['model_size_b'] is None for r in records)
    assert all(r['t2_cpl'] is None and r['t2_first_attempt_legal'] for r in records)


def test_first_answer_no_recovery_and_empty_capped_refused_retained(setup):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    replies = [response('Eval: 0\nMove: e5\nExplanation: e4 would be legal.'),
               response('', status='incomplete', incomplete_details={'reason': 'max_output_tokens'}),
               response('', output=[{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]}]),
               response('')]
    s = pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', Mock(side_effect=replies))
    assert s['completed'] == 4 and s['capped'] == 1 and s['refused'] == 1
    assert s['first_answer_legal'] == 0
    assert s['missing_move'] == 3


def test_budget_checked_before_any_submission(setup):
    cfg, _ = setup
    cfg['openai']['budget_usd'] = .000001
    plan = tiny_plan(cfg)
    transport = Mock(side_effect=AssertionError('must not submit'))
    with pytest.raises(pilot.PilotError, match='budget'):
        pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', transport)
    assert not transport.called


def test_ambiguous_failure_blocks_resume_and_redacts_secret(setup):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    transport = Mock(side_effect=TimeoutError('dummy-secret Authorization Bearer'))
    with pytest.raises(pilot.PilotError) as error:
        pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', transport)
    assert 'dummy-secret' not in str(error.value) and 'Bearer' not in str(error.value)
    with pytest.raises(pilot.PilotError, match='outstanding'):
        pilot.execute_plan(plan, Path(cfg['paths']['output_dir']), 'dummy-secret', Mock())


@pytest.mark.parametrize('usage', [None, {}, {'input_tokens': 1, 'output_tokens': float('nan')},
                                  {'input_tokens': 1, 'output_tokens': -1}])
def test_unknown_usage_is_saved_but_blocks_further_requests(setup, usage):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    transport = Mock(return_value=response(usage=usage))
    out = Path(cfg['paths']['output_dir'])
    with pytest.raises(pilot.PilotError, match='usage'):
        pilot.execute_plan(plan, out, 'dummy-secret', transport)
    assert transport.call_count == 1
    assert 'resp_test' in (out / 'ledger.jsonl').read_text()
    with pytest.raises(pilot.PilotError, match='outstanding'):
        pilot.execute_plan(plan, out, 'dummy-secret', Mock())


def test_plan_mismatch_rejected(setup):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    out = Path(cfg['paths']['output_dir'])
    pilot.execute_plan(plan, out, 'dummy-secret', Mock(return_value=response()))
    cfg['benchmark']['random_seed'] = 7
    with pytest.raises(pilot.PilotError, match='plan'):
        pilot.execute_plan(tiny_plan(cfg), out, 'dummy-secret', Mock())


@pytest.mark.parametrize('corruption', ['duplicate', 'truncated', 'unknown'])
def test_corrupt_ledger_rejected(setup, corruption):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    out = Path(cfg['paths']['output_dir'])
    pilot.execute_plan(plan, out, 'dummy-secret', Mock(return_value=response()))
    ledger = out / 'ledger.jsonl'
    if corruption == 'duplicate':
        ledger.write_text(ledger.read_text() + ledger.read_text().splitlines()[0] + '\n')
    elif corruption == 'truncated':
        ledger.write_text(ledger.read_text() + '{')
    else:
        ledger.write_text(ledger.read_text() + '{"event":"unknown"}\n')
    with pytest.raises(pilot.PilotError):
        pilot.execute_plan(plan, out, 'dummy-secret', Mock())


def test_exclusive_execution_lock(setup):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    out = Path(cfg['paths']['output_dir'])
    with pilot.execution_lock(out):
        with pytest.raises(pilot.PilotError, match='running'):
            pilot.execute_plan(plan, out, 'dummy-secret', Mock())


@pytest.mark.parametrize('loss', ['missing', 'empty', 'valid_prefix'])
def test_missing_established_ledger_never_resubmits_or_overwrites(setup, loss):
    cfg, _ = setup
    plan = tiny_plan(cfg)
    out = Path(cfg['paths']['output_dir'])
    pilot.execute_plan(plan, out, 'dummy-secret', Mock(return_value=response()))
    before = {name: (out / name).read_bytes() for name in ('plan.json', 'responses.jsonl', 'evaluations.jsonl')}
    ledger = out / 'ledger.jsonl'
    if loss == 'missing':
        ledger.unlink()
    elif loss == 'empty':
        ledger.write_text('')
    else:
        # Keep a complete, valid request triple while losing later paid requests.
        ledger.write_text('\n'.join(ledger.read_text().splitlines()[:3]) + '\n')
    transport = Mock(side_effect=AssertionError('must not resubmit'))
    with pytest.raises(pilot.PilotError, match='ledger'):
        pilot.execute_plan(plan, out, 'dummy-secret', transport)
    assert not transport.called
    assert before == {name: (out / name).read_bytes() for name in before}
