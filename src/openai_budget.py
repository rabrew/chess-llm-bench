"""Fail-closed, persistent session accounting in integer nano-USD."""
import hashlib
import json
import os
import re
import tempfile
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

NANO_USD = 1_000_000_000


class BudgetError(RuntimeError):
    """A safe stop; callers must not retry paid requests automatically."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def money(value):
    decimal = Decimal(str(value))
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError('Money settings must be finite and positive')
    return int((decimal * NANO_USD).to_integral_value(rounding=ROUND_CEILING))


def cost(tokens_in, tokens_out, price):
    amount = Decimal(tokens_in)*Decimal(price['input']) + Decimal(tokens_out)*Decimal(price['output'])
    return int((amount*1000).to_integral_value(rounding=ROUND_CEILING))


def priced_model(requested, resolved):
    return isinstance(resolved,str) and re.fullmatch(re.escape(requested)+r"(?:-\d{4}-\d{2}-\d{2})?",resolved) is not None


def reservation(job, settings):
    bound = len(json.dumps(job['payload']['input'],ensure_ascii=False).encode('utf-8')) + settings['input_overhead_tokens']
    base = cost(bound,settings['max_output_tokens'],settings['prices'][job['model']])
    reserved = int((Decimal(base)*Decimal(settings['cost_margin'])).to_integral_value(rounding=ROUND_CEILING))
    return bound, reserved


def safe_path(path):
    original = Path(path).expanduser().absolute()
    if any(candidate.is_symlink() for candidate in (original,*original.parents)):
        raise ValueError('Symlink state/output paths are unsupported')
    return original.resolve()


def atomic_json(path, value):
    path = safe_path(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name+'.',suffix='.pending',dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def valid_usage(usage):
    if not isinstance(usage,dict) or any(type(usage.get(key)) is not int or usage[key]<0 for key in ('input_tokens','output_tokens')):
        return False
    if 'total_tokens' in usage and (type(usage['total_tokens']) is not int or usage['total_tokens'] != usage['input_tokens']+usage['output_tokens']):
        return False
    for details,field,total in (('input_tokens_details','cached_tokens','input_tokens'),('output_tokens_details','reasoning_tokens','output_tokens')):
        if details not in usage:
            continue
        if not isinstance(usage[details],dict):
            return False
        if field in usage[details] and (type(usage[details][field]) is not int or not 0<=usage[details][field]<=usage[total]):
            return False
    return True


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise BudgetError('Corrupt or unreadable persistent state') from error


class BudgetLedger:
    """Caller owns the exclusive file lock for the lifetime of this instance."""
    def __init__(self, path, settings):
        self.path = Path(path)
        self.settings = settings
        self.cap = money(settings['max_usd'])
        self.pilot_cap = money(settings['pilot_max_usd'])
        identity = {key: settings[key] for key in ('max_usd','pilot_max_usd','prices','pricing_verified_at',
                    'pricing_url','cost_margin','input_overhead_tokens','max_output_tokens')}
        self.identity = fingerprint(identity)
        if self.path.exists():
            self.state = read_json(path)
            self._validate()
        else:
            self.state = dict(version=1, settings_hash=self.identity, entries={})
            self.save()

    def _validate(self):
        if not isinstance(self.state, dict) or self.state.get('version') != 1 or self.state.get('settings_hash') != self.identity:
            raise BudgetError('Ledger configuration changed or ledger is corrupt')
        entries = self.state.get('entries')
        if not isinstance(entries, dict):
            raise BudgetError('Invalid ledger entries')
        for key, entry in entries.items():
            if not isinstance(key, str) or len(key) != 64 or not isinstance(entry, dict):
                raise BudgetError('Invalid ledger entry')
            if entry.get('status') not in ('reserved','complete','stopped') or entry.get('phase') not in ('pilot','main'):
                raise BudgetError('Invalid ledger status')
            for field in ('charge_nano','reserved_nano'):
                if type(entry.get(field)) is not int or entry[field] < 0:
                    raise BudgetError('Invalid ledger charge')
            self._validate_entry(key,entry)

    def _validate_entry(self, key, entry):
        try:
            job = entry['request']
            bound, reserved = reservation(job,self.settings)
            if job['id'] != key or fingerprint(dict(payload=job['payload'],position=job['position'])) != key or reserved != entry['reserved_nano']:
                raise ValueError()
            if entry['status'] != 'complete':
                if entry['charge_nano'] < reserved:
                    raise ValueError()
                return
            result, body = entry['result'], entry['result']['raw_response']
            usage = body['usage']
            if result['job_id'] != key or body.get('status') != 'completed' or body.get('service_tier') != 'default' or not priced_model(job['model'],body.get('model')):
                raise ValueError()
            if not valid_usage(usage):
                raise ValueError()
            charge = cost(usage['input_tokens'],usage['output_tokens'],self.settings['prices'][job['model']])
            if charge != entry['charge_nano'] or charge > reserved or usage['input_tokens'] > bound or usage['output_tokens'] > self.settings['max_output_tokens']:
                raise ValueError()
        except (KeyError,TypeError,ValueError,AttributeError,ArithmeticError):
            raise BudgetError('Ledger request/charge invariant violated') from None

    def save(self):
        atomic_json(self.path, self.state)

    def preflight(self, jobs, phase):
        entries = self.state['entries']
        if any(entries.get(job['id'], {}).get('status') in ('reserved','stopped') for job in jobs):
            raise BudgetError('Previous paid request stopped or is uncertain; no automatic retry')
        additional = sum(job['reserved_nano'] for job in jobs if job['id'] not in entries)
        total = sum(entry['charge_nano'] for entry in entries.values())
        pilot = sum(entry['charge_nano'] for entry in entries.values() if entry['phase'] == 'pilot')
        if total + additional > self.cap or (phase == 'pilot' and pilot + additional > self.pilot_cap):
            raise BudgetError('Entire remaining cohort exceeds session or pilot budget')

    def reserve(self, job, phase):
        self.preflight([job], phase)
        if job['id'] in self.state['entries']:
            raise BudgetError('Duplicate paid request identity')
        self.state['entries'][job['id']] = dict(status='reserved', phase=phase,
                    reserved_nano=job['reserved_nano'], charge_nano=job['reserved_nano'],request=job)
        self.save()

    def finish(self, job, result, charge, reliable):
        entry = self.state['entries'][job['id']]
        entry.update(status='complete' if reliable else 'stopped', result=result,
                     charge_nano=charge if reliable else max(charge, entry['reserved_nano']))
        self.save()
