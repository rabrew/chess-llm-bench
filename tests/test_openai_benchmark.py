"""Budget safety and first-attempt collection regressions, no real API calls."""
import copy
import json
from pathlib import Path

import chess
import pytest
import requests

from src.openai_benchmark import build_manifest, load_settings, run_benchmark
from src.openai_budget import BudgetError


@pytest.fixture
def settings():
    configured = load_settings('config/openai_benchmark.json')
    configured['positions_per_tier'] = 1
    return configured


@pytest.fixture
def source(tmp_path):
    records = []
    for index, tier in enumerate(('easy', 'medium', 'hard', 'extreme')):
        board = chess.Board()
        board.fullmove_number = index + 1
        record = dict(position_id=index, fen=board.fen(), difficulty=tier,
                      t1_stockfish_eval=0, theme='advantage', phase='opening')
        records.extend([record, record])
    path = tmp_path / 'historical.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    return path


def completed(payload, key):
    return dict(id='resp_test', model=payload['model'], status='completed', service_tier='default',
                usage=dict(input_tokens=100, output_tokens=30),
                output=[dict(type='message', content=[dict(type='output_text',
                    text='Eval: 0\nMove: e4\nExplanation: Equal — balanced material.')])])


def test_plan_is_free_and_deterministic(settings, source, tmp_path):
    first = build_manifest(source, settings)
    assert first == build_manifest(source, settings)
    assert len(first['positions']) == 4
    assert len(first['jobs']) == 8
    destination, ledger = tmp_path/'run', tmp_path/'budget.json'
    report = run_benchmark(first, destination, ledger, execute=False)
    assert report['request_count'] == 8
    assert not destination.exists() and not ledger.exists()


def test_execute_resume_and_shared_session(settings, source, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-secret')
    manifest = build_manifest(source, settings)
    calls = []
    def transport(payload, key):
        calls.append(payload)
        return completed(payload, key)
    ledger = tmp_path/'budget.json'
    run_benchmark(manifest, tmp_path/'run', ledger, execute=True, transport=transport)
    assert len(calls) == 8
    run_benchmark(manifest, tmp_path/'run', ledger, execute=True, transport=transport)
    run_benchmark(manifest, tmp_path/'other-run', ledger, execute=True, transport=transport)
    assert len(calls) == 8
    stored = json.loads(ledger.read_text())
    assert all(entry['status'] == 'complete' for entry in stored['entries'].values())
    assert 'test-only-secret' not in ledger.read_text()
    record = next((tmp_path/'run'/'records').glob('*.json'))
    assert json.loads(record.read_text())['model_size_b'] is None


def test_timeout_reserved_no_repeat(settings, source, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-real-key')
    manifest = build_manifest(source, settings)
    calls = []
    def timeout(payload, key):
        calls.append(1)
        raise requests.Timeout('secret-sensitive-error')
    with pytest.raises(BudgetError, match='stopped'):
        run_benchmark(manifest, tmp_path/'run', tmp_path/'budget.json', execute=True, transport=timeout)
    state = json.loads((tmp_path/'budget.json').read_text())
    assert sum(row['charge_nano'] for row in state['entries'].values()) > 0
    with pytest.raises(BudgetError):
        run_benchmark(manifest, tmp_path/'run', tmp_path/'budget.json', execute=True, transport=timeout)
    assert calls == [1]
    assert 'secret-sensitive-error' not in (tmp_path/'budget.json').read_text()


@pytest.mark.parametrize('kind', ['missing_usage','incomplete','refusal','overflow'])
def test_unreliable_outcomes_stop_keep_reserve(kind, settings, source, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    def transport(payload,key):
        response = completed(payload,key)
        if kind == 'missing_usage': response.pop('usage')
        if kind == 'incomplete': response['status'] = 'incomplete'
        if kind == 'refusal': response['output'][0]['content'] = [dict(type='refusal', refusal='No')]
        if kind == 'overflow': response['usage']['input_tokens'] = 1000000
        return response
    with pytest.raises(BudgetError):
        run_benchmark(build_manifest(source,settings),tmp_path/'run',tmp_path/'budget.json',execute=True,transport=transport)
    entry = next(iter(json.loads((tmp_path/'budget.json').read_text())['entries'].values()))
    assert entry['charge_nano'] >= entry['reserved_nano']
    assert entry['status'] == 'stopped'


def test_entire_cohort_budget_preflight(settings, source, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    settings['pilot_max_usd']='0.000001'
    calls=[]
    with pytest.raises(BudgetError,match='budget'):
        run_benchmark(build_manifest(source,settings),tmp_path/'run',tmp_path/'budget.json',execute=True,
                      transport=lambda *args:calls.append(1))
    assert not calls


def test_changed_caps_and_manifest_fail_closed(settings, source, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    ledger=tmp_path/'budget.json'
    manifest=build_manifest(source,settings)
    run_benchmark(manifest,tmp_path/'run',ledger,execute=True,transport=completed)
    changed=copy.deepcopy(settings); changed['max_usd']='19'
    with pytest.raises(BudgetError):
        run_benchmark(build_manifest(source,changed),tmp_path/'next',ledger,execute=True,transport=completed)
    changed=copy.deepcopy(manifest); changed['positions'][0]['theme']='changed'
    with pytest.raises((BudgetError,ValueError)):
        run_benchmark(changed,tmp_path/'run',ledger,execute=True,transport=completed)


def test_corrupt_ledger_or_missing_with_output_fails(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    manifest=build_manifest(source,settings); out=tmp_path/'run'; ledger=tmp_path/'budget.json'
    run_benchmark(manifest,out,ledger,execute=True,transport=completed)
    ledger.write_text('{')
    with pytest.raises(BudgetError): run_benchmark(manifest,out,ledger,execute=True,transport=completed)
    ledger.unlink()
    with pytest.raises(BudgetError): run_benchmark(manifest,out,ledger,execute=True,transport=completed)


def test_source_conflict_and_pgn_missing(settings,source):
    records=[json.loads(line) for line in source.read_text().splitlines()]
    records[1]['t1_stockfish_eval']=999
    source.write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError,match='conflict'): build_manifest(source,settings)
    records[1]['t1_stockfish_eval']=0
    source.write_text(''.join(json.dumps(row)+'\n' for row in records))
    settings['formats']=['pgn+fen']
    with pytest.raises(ValueError,match='PGN'): build_manifest(source,settings)


def test_protected_output_and_main_without_pilot(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    manifest=build_manifest(source,settings)
    with pytest.raises(ValueError): run_benchmark(manifest,Path('results/new'),tmp_path/'budget.json',execute=True,transport=completed)
    with pytest.raises(BudgetError,match='pilot'):
        run_benchmark(manifest,tmp_path/'run',tmp_path/'budget.json',phase='main',execute=True,transport=completed)


def test_price_understatement_and_manifest_tampering(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    cheap=copy.deepcopy(settings); cheap['prices']['gpt-6.1-sol']['input']='0.000001'
    with pytest.raises(ValueError): build_manifest(source,cheap)
    manifest=build_manifest(source,settings); manifest['jobs'][0]['reserved_nano']=0
    with pytest.raises((ValueError,BudgetError)):
        run_benchmark(manifest,tmp_path/'run',tmp_path/'budget.json',execute=True,transport=completed)


def test_unknown_resolved_model_stops(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    def wrong(payload,key):
        response=completed(payload,key); response['model']='expensive-unknown-model'; return response
    with pytest.raises(BudgetError):
        run_benchmark(build_manifest(source,settings),tmp_path/'run',tmp_path/'budget.json',execute=True,transport=wrong)


def test_corrupt_reservation_cannot_release_money(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    manifest=build_manifest(source,settings); ledger=tmp_path/'budget.json'
    def timeout(*args): raise requests.Timeout()
    with pytest.raises(BudgetError): run_benchmark(manifest,tmp_path/'run',ledger,execute=True,transport=timeout)
    value=json.loads(ledger.read_text()); next(iter(value['entries'].values()))['charge_nano']=0
    ledger.write_text(json.dumps(value))
    with pytest.raises(BudgetError): run_benchmark(manifest,tmp_path/'run',ledger,execute=True,transport=completed)


def test_main_keeps_metered_parse_failure_without_retry(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    settings['positions_per_tier']=2
    rows=[json.loads(line) for line in source.read_text().splitlines()][::2]
    extra=[]
    for row in rows:
        second=copy.deepcopy(row); board=chess.Board(row['fen']); board.halfmove_clock=1
        second['fen']=board.fen(); second['position_id']+=100; extra.append(second)
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows+extra))
    manifest=build_manifest(source,settings); ledger=tmp_path/'budget.json'
    run_benchmark(manifest,tmp_path/'run',ledger,execute=True,transport=completed)
    calls=[]
    def missing(payload,key):
        calls.append(1); response=completed(payload,key)
        response['output'][0]['content'][0]['text']='unparseable'
        return response
    run_benchmark(manifest,tmp_path/'run',ledger,phase='main',execute=True,transport=missing)
    assert len(calls)==8
    entries=json.loads(ledger.read_text())['entries'].values()
    assert sum(entry['result'].get('parse_quality')=='failure' for entry in entries)==8
    assert all(entry['status']=='complete' for entry in entries)


@pytest.mark.parametrize('usage', [dict(input_tokens=100,output_tokens=30,total_tokens=1),
    dict(input_tokens=100,output_tokens=30,output_tokens_details=dict(reasoning_tokens=31)),
    dict(input_tokens=100,output_tokens=30,input_tokens_details=dict(cached_tokens=101))])
def test_malformed_usage_never_releases_reserve(usage,settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    def bad(payload,key):
        body=completed(payload,key); body['usage']=usage; return body
    with pytest.raises(BudgetError):
        run_benchmark(build_manifest(source,settings),tmp_path/'run',tmp_path/'budget.json',execute=True,transport=bad)


def test_symlink_records_rejected_before_transport(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    out=tmp_path/'run'; out.mkdir(); victim=tmp_path/'protected'; victim.mkdir()
    (out/'records').symlink_to(victim,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):
        run_benchmark(build_manifest(source,settings),out,tmp_path/'budget.json',execute=True,transport=completed)
    assert list(victim.iterdir())==[]


def test_durable_result_recovers_output_failure_without_paid_repeat(settings,source,tmp_path,monkeypatch):
    import src.openai_benchmark as runner
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    manifest=build_manifest(source,settings); out=tmp_path/'run'; ledger=tmp_path/'budget.json'
    original=runner.atomic_json; calls=[]
    def disk_error(path,value):
        if Path(path).parent.name=='records': raise OSError('disk full')
        return original(path,value)
    def transport(payload,key): calls.append(1); return completed(payload,key)
    monkeypatch.setattr(runner,'atomic_json',disk_error)
    with pytest.raises(OSError): run_benchmark(manifest,out,ledger,execute=True,transport=transport)
    monkeypatch.setattr(runner,'atomic_json',original)
    run_benchmark(manifest,out,ledger,execute=True,transport=transport)
    assert len(calls)==8


def test_atomic_write_ignores_predictable_pending_symlink(tmp_path):
    from src.openai_budget import atomic_json
    victim=tmp_path/'victim'; victim.write_text('preserve')
    target=tmp_path/'output.json'; (tmp_path/'output.json.pending').symlink_to(victim)
    atomic_json(target,dict(safe=True))
    assert victim.read_text()=='preserve'


def test_reservation_persisted_before_transport(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy'); ledger=tmp_path/'budget.json'
    def inspect(payload,key):
        entries=json.loads(ledger.read_text())['entries'].values()
        assert any(entry['status']=='reserved' and entry['charge_nano']>0 for entry in entries)
        return completed(payload,key)
    run_benchmark(build_manifest(source,settings),tmp_path/'run',ledger,execute=True,transport=inspect)


@pytest.mark.parametrize('ledger_name',['summary.json','manifest.json','records/paid.json'])
def test_ledger_output_collision_rejected_before_transport(ledger_name,settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy'); out=tmp_path/'run'
    with pytest.raises(ValueError,match='outside'):
        run_benchmark(build_manifest(source,settings),out,out/ledger_name,execute=True,transport=completed)
    assert not out.exists()


def test_manifest_cannot_omit_version_fingerprints(settings,source,tmp_path):
    manifest=build_manifest(source,settings); manifest['code_sha256']={}
    with pytest.raises(ValueError,match='schema'):
        run_benchmark(manifest,tmp_path/'run',tmp_path/'budget.json')



def test_secret_config_fields_rejected(settings,source):
    settings['api_key']='never-persist-this'
    with pytest.raises(ValueError,match='non-secret'):
        build_manifest(source,settings)


def truncated(payload,key):
    response=completed(payload,key)
    response.update(status='incomplete',incomplete_details={'reason':'max_output_tokens'})
    response['usage']={'input_tokens':100,'output_tokens':payload['max_output_tokens']}
    return response


def expand_source_for_main(source,settings):
    settings['positions_per_tier']=2
    rows=[json.loads(line) for line in source.read_text().splitlines()][::2]
    extra=[]
    for row in rows:
        second=copy.deepcopy(row); board=chess.Board(row['fen']); board.halfmove_clock=1
        second['fen']=board.fen(); second['position_id']+=100; extra.append(second)
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows+extra))


def test_main_terminal_truncation_continues_without_repay(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy'); expand_source_for_main(source,settings)
    manifest=build_manifest(source,settings); ledger=tmp_path/'budget.json'; out=tmp_path/'run'
    run_benchmark(manifest,out,ledger,execute=True,transport=completed)
    calls=[]
    def one_truncation(payload,key):
        calls.append(1)
        return truncated(payload,key) if len(calls)==1 else completed(payload,key)
    run_benchmark(manifest,out,ledger,phase='main',execute=True,transport=one_truncation)
    assert len(calls)==8
    run_benchmark(manifest,out,ledger,phase='main',execute=True,transport=one_truncation)
    assert len(calls)==8
    entries=json.loads(ledger.read_text())['entries']
    assert len(entries)==16
    failed=[entry for entry in entries.values() if entry['status']=='terminal_failure']
    assert len(failed)==1 and failed[0]['charge_nano']==failed[0]['reserved_nano']
    assert failed[0]['result']['parse_quality']=='failure'
    assert failed[0]['result']['parsed_first_attempt']['move']=='e4'
    assert failed[0]['result']['t2_legal_attempt_1'] is False
    assert sum(bucket['terminal_failure'] for bucket in json.loads((out/'summary.json').read_text())['models'].values())==1


def test_old_code_migration_recovers_partial_manifest_write(settings,source,tmp_path,monkeypatch):
    import src.openai_benchmark as runner
    from src.openai_budget import BudgetLedger
    monkeypatch.setenv('OPENAI_API_KEY','dummy'); expand_source_for_main(source,settings)
    manifest=build_manifest(source,settings); ledger=tmp_path/'budget.json'; out=tmp_path/'run'
    run_benchmark(manifest,out,ledger,execute=True,transport=completed)
    job=next(job for job in manifest['jobs'] if job['position']['fen'] not in manifest['pilot_fens'])
    account=BudgetLedger(ledger,settings); account.reserve(job,'main')
    record,charge,reliable=runner.interpret_response(truncated(job['payload'],'dummy'),job,settings)
    account.finish(job,record,charge,reliable)
    old=copy.deepcopy(manifest); old['code_sha256'].update(runner.KNOWN_ORIGINAL_RUNNER)
    (out/'manifest.json').write_text(json.dumps(old)); original_charge=sum(entry['charge_nano'] for entry in account.state['entries'].values())
    original=runner.atomic_json
    def crash(path,value):
        if Path(path).name=='manifest.json': raise OSError('simulated output failure')
        return original(path,value)
    monkeypatch.setattr(runner,'atomic_json',crash)
    with pytest.raises(OSError): runner.accept_known_truncations(source,settings,out,ledger)
    monkeypatch.setattr(runner,'atomic_json',original)
    runner.accept_known_truncations(source,settings,out,ledger)
    upgraded=runner.frozen_manifest_for_run(source,settings,out)
    state=json.loads(ledger.read_text())
    assert state['entries'][job['id']]['status']=='terminal_failure'
    assert sum(entry['charge_nano'] for entry in state['entries'].values())==original_charge
    assert len(state['terminal_policy_migrations'])==1
    assert upgraded['runtime_migrations'][0]['promoted_ids']==[job['id']]
    assert list(out.glob('manifest.before-terminal-*.json'))
    calls=[]
    def remaining(payload,key): calls.append(1); return completed(payload,key)
    run_benchmark(upgraded,out,ledger,phase='main',execute=True,transport=remaining)
    assert len(calls)==7
    assert runner.accept_known_truncations(source,settings,out,ledger)['promoted']==0



@pytest.mark.parametrize('mutation',[{'service_tier':'priority'},{'model':'other'},
    {'incomplete_details':{'reason':'other'}},{'usage':None},
    {'usage':{'input_tokens':100,'output_tokens':2049}}])
def test_terminal_recovery_rejects_unproven_body(mutation,settings,source):
    from src.openai_budget import verified_truncation
    job=build_manifest(source,settings)['jobs'][0]
    body=truncated(job['payload'],'dummy'); body.update(mutation)
    assert not verified_truncation(job,body,settings)


def test_future_truncation_has_distinct_full_reserved_failure(settings,source,tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','dummy')
    manifest=build_manifest(source,settings); ledger=tmp_path/'budget.json'; out=tmp_path/'run'
    with pytest.raises(BudgetError): run_benchmark(manifest,out,ledger,execute=True,transport=truncated)
    entry=next(iter(json.loads(ledger.read_text())['entries'].values()))
    assert entry['status']=='terminal_failure'
    assert entry['charge_nano']==entry['reserved_nano']
    assert entry['result']['outcome']=='truncated'
    with pytest.raises(BudgetError): run_benchmark(manifest,out,ledger,execute=True,transport=completed)
