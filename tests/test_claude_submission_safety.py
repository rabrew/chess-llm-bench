"""Offline regression checks for fail-closed paid batch submission."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from src import claude_batch as cb
from scripts import run_claude_batch as runner


def setup_run(tmp_path, create):
    ledger = cb.BatchLedger(tmp_path / 'batches.json')
    store = cb.ResponseStore(tmp_path / 'responses.jsonl')
    batches = NS(create=create, retrieve=Mock(return_value=NS(processing_status='ended')), results=Mock(return_value=[]))
    return ledger, store, NS(messages=NS(batches=batches))


def submit(client, ledger, store):
    cb.run_round(client, [{'custom_id': 'request-r1'}], 1, ledger, store, {}, {}, .5, 0)


def test_reservation_is_durable_before_create(tmp_path, monkeypatch):
    synced = []
    original = cb.os.fsync
    monkeypatch.setattr(cb.os, 'fsync', lambda fd: (synced.append(fd), original(fd))[-1])
    def create(**kwargs):
        rows = json.loads(ledger.path.read_text())
        assert rows[0]['state'] == cb.SUBMISSION_UNKNOWN
        assert rows[0]['request_ids'] == ['request-r1']
        assert len(synced) >= 2
        return NS(id='batch_ok')
    ledger, store, client = setup_run(tmp_path, create)
    submit(client, ledger, store)
    assert ledger.open_batches() == []
    client.messages.batches.retrieve.assert_called_once_with('batch_ok')


def test_failed_create_blocks_restart_and_collection(tmp_path):
    ledger, store, client = setup_run(tmp_path, Mock(side_effect=RuntimeError('secret-provider-detail')))
    with pytest.raises(cb.SubmissionSafetyError) as error:
        submit(client, ledger, store)
    assert 'secret-provider-detail' not in str(error.value)
    with pytest.raises(cb.SubmissionSafetyError):
        submit(client, ledger, store)
    with pytest.raises(cb.SubmissionSafetyError):
        cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    assert client.messages.batches.create.call_count == 1
    client.messages.batches.retrieve.assert_not_called()


def test_accepted_then_persistence_failure_leaves_unknown(tmp_path, monkeypatch):
    ledger, store, client = setup_run(tmp_path, Mock(return_value=NS(id='batch_ok')))
    original = ledger._save
    def save(rows):
        if any(row.get('id') == 'batch_ok' for row in rows):
            raise OSError('private-details')
        original(rows)
    monkeypatch.setattr(ledger, '_save', save)
    with pytest.raises(cb.SubmissionSafetyError):
        submit(client, ledger, store)
    with pytest.raises(cb.SubmissionSafetyError):
        ledger.open_batches()
    client.messages.batches.retrieve.assert_not_called()


@pytest.mark.parametrize('value', ['{', '{}', '[{}]', '[{"id":"b","round":true}]', '[{"id":"b","round":3}]', '[{"state":"unknown"}]'])
def test_invalid_ledgers_fail_closed(tmp_path, value):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.path.write_text(value)
    with pytest.raises(cb.SubmissionSafetyError):
        cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    client.messages.batches.retrieve.assert_not_called()
    client.messages.batches.create.assert_not_called()


def test_legacy_open_batch_still_collects(tmp_path):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.path.write_text('[{"id":"batch_old","round":1}]')
    cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    assert ledger.open_batches() == []
    client.messages.batches.retrieve.assert_called_once_with('batch_old')


def test_sdk_factory_disables_retries_without_credentials():
    sdk = NS(Anthropic=Mock())
    runner.make_client(sdk, 'fake-test-key')
    sdk.Anthropic.assert_called_once_with(api_key='fake-test-key', max_retries=0)


def test_provider_never_called_before_ledger_exists(tmp_path):
    def create(**kwargs):
        assert ledger.path.is_file(), 'provider called before any durable reservation'
        return NS(id='batch_ok')
    ledger, store, client = setup_run(tmp_path, create)
    submit(client, ledger, store)


def test_crash_leaves_reservation(tmp_path):
    ledger, store, client = setup_run(tmp_path, Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        submit(client, ledger, store)
    with pytest.raises(cb.SubmissionSafetyError):
        cb.run_round(client, [], 1, ledger, store, {}, {}, .5, 0)
    assert json.loads(ledger.path.read_text())[0]['state'] == cb.SUBMISSION_UNKNOWN


def test_failed_reservation_never_submits(tmp_path, monkeypatch):
    ledger, store, client = setup_run(tmp_path, Mock())
    monkeypatch.setattr(cb.os, 'fsync', Mock(side_effect=OSError('private-details')))
    with pytest.raises(cb.SubmissionSafetyError) as error:
        submit(client, ledger, store)
    assert 'private-details' not in str(error.value)
    client.messages.batches.create.assert_not_called()


def test_reservation_after_legacy_batch_blocks_all_polling(tmp_path):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.path.write_text(json.dumps([{'id': 'batch_old', 'round': 1},
                                      {'state': cb.SUBMISSION_UNKNOWN, 'round': 1, 'request_ids': ['r1']}]))
    with pytest.raises(cb.SubmissionSafetyError):
        cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    client.messages.batches.retrieve.assert_not_called()


def test_provider_poll_error_is_safe_and_retains_batch(tmp_path):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.add('batch_old', 1)
    client.messages.batches.retrieve.side_effect = RuntimeError('private-provider-header')
    with pytest.raises(cb.SubmissionSafetyError) as error:
        cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    assert 'private-provider-header' not in str(error.value)
    assert ledger.open_batches() == [{'id': 'batch_old', 'round': 1}]


def test_second_cli_instance_blocks_before_credentials_or_transport(tmp_path, monkeypatch):
    config = {'claude': {'budget_usd': 1}, 'paths': {'responses_file': str(tmp_path/'r.jsonl'),
              'batches_file': str(tmp_path/'b.json')}, 'benchmark': {'prompt_formats': []}}
    monkeypatch.setattr(runner, 'parse_args', lambda: NS(config='fake', status=False, dry_run=False, variants=''))
    monkeypatch.setattr(runner, 'setup_logging', lambda: None)
    monkeypatch.setattr(runner, 'load_config', lambda path: config)
    monkeypatch.setattr(cb, 'load_variants', lambda config: [])
    key = Mock(side_effect=AssertionError('credentials must not be accessed'))
    transport = Mock(side_effect=AssertionError('transport must not be created'))
    monkeypatch.setattr(cb, 'load_api_key', key)
    monkeypatch.setattr(runner, 'make_client', transport)
    with cb.execution_lock(tmp_path/'b.json'):
        with pytest.raises(SystemExit, match='already running'):
            runner.main()
    key.assert_not_called()
    transport.assert_not_called()


def test_response_sync_precedes_ledger_removal(tmp_path, monkeypatch):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.add('batch_old', 1)
    events = []
    original_sync = cb.os.fsync
    def sync(fd):
        events.append(('sync', cb.os.readlink(f'/proc/self/fd/{fd}')))
        original_sync(fd)
    monkeypatch.setattr(cb.os, 'fsync', sync)
    original_remove = ledger.remove
    def remove(batch_id):
        assert ('sync', str(store.path)) in events
        assert ('sync', str(store.path.parent)) in events
        events.append(('remove', batch_id))
        original_remove(batch_id)
    monkeypatch.setattr(ledger, 'remove', remove)
    cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    assert ledger.open_batches() == []


def test_response_sync_failure_retains_open_batch(tmp_path, monkeypatch):
    ledger, store, client = setup_run(tmp_path, Mock())
    ledger.add('batch_old', 1)
    monkeypatch.setattr(cb.os, 'fsync', Mock(side_effect=OSError('disk-failure')))
    with pytest.raises(OSError):
        cb.collect_open_batches(client, ledger, store, {}, {}, .5, 0)
    assert ledger.open_batches() == [{'id': 'batch_old', 'round': 1}]


def test_lock_is_exclusive_across_processes_and_released(tmp_path):
    import subprocess
    import sys
    ledger_path = tmp_path/'b.json'
    code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
            'fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)')
    with cb.execution_lock(ledger_path):
        blocked = subprocess.run([sys.executable, '-c', code, str(ledger_path)+'.lock'], capture_output=True)
        assert blocked.returncode != 0
    acquired = subprocess.run([sys.executable, '-c', code, str(ledger_path)+'.lock'], capture_output=True)
    assert acquired.returncode == 0


@pytest.mark.parametrize('status', [True, False])
def test_readonly_cli_does_not_create_lock_or_access_credentials(tmp_path, monkeypatch, status):
    config = {'claude': {'budget_usd': 1}, 'paths': {'responses_file': str(tmp_path/'r.jsonl'),
              'batches_file': str(tmp_path/'b.json'), 'data_dir': str(tmp_path)},
              'benchmark': {'prompt_formats': []}}
    monkeypatch.setattr(runner, 'parse_args', lambda: NS(config='fake', status=status, dry_run=not status,
                                                       variants='', positions_per_tier=0))
    monkeypatch.setattr(runner, 'setup_logging', lambda: None)
    monkeypatch.setattr(runner, 'load_config', lambda path: config)
    monkeypatch.setattr(cb, 'load_variants', lambda config: [])
    monkeypatch.setattr(cb, 'sample_positions', lambda *args: [])
    monkeypatch.setattr(cb, 'build_jobs', lambda *args: [])
    monkeypatch.setattr(cb, 'request_lookup', lambda *args: {})
    monkeypatch.setattr(cb, 'pilot_positions', lambda *args: [])
    monkeypatch.setattr(cb, 'estimate_usd', lambda *args: 0)
    key = Mock(side_effect=AssertionError('credentials must not be accessed'))
    monkeypatch.setattr(cb, 'load_api_key', key)
    runner.main()
    key.assert_not_called()
    assert list(tmp_path.iterdir()) == []
