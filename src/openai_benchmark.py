"""Standalone matched OpenAI benchmark; planning never calls the API."""
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import chess
import requests
from filelock import FileLock

from .evaluator import score_all
from .llm_client import (build_prompt, build_eval_prompt, build_explanation_prompt, build_move_prompt,
                        parse_response, parse_eval_response, parse_explanation_response, extract_move_from_text,
                        EVAL_SYSTEM_PROMPT, EXPLANATION_SYSTEM_PROMPT, MOVE_SYSTEM_PROMPT)
from .openai_budget import BudgetError, BudgetLedger, atomic_json, cost, fingerprint, money, read_json, priced_model, safe_path, valid_usage, verified_truncation

TIERS = ('easy','medium','hard','extreme')
API_URL = 'https://api.openai.com/v1/responses'
REPO = Path(__file__).resolve().parent.parent
MODELS = ('gpt-6-luna','gpt-6.1-sol','gpt-6-astra')
VERIFIED_PRICES = {'gpt-6-luna':('0.10','0.50'),'gpt-6.1-sol':('2','10'),'gpt-6-astra':('10','50')}
FORMATS = ('fen_only','pgn+fen','cot','eval_only','move_only','explanation_only')


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    validate_settings(settings)
    return settings


def validate_settings(settings):
    allowed={'models','positions_per_tier','pilot_per_tier','formats','seed','max_output_tokens','input_overhead_tokens',
             'cost_margin','max_usd','pilot_max_usd','timeout_seconds','pricing_verified_at','pricing_url','prices'}
    if not isinstance(settings,dict) or set(settings)!=allowed:
        raise ValueError('Only documented non-secret configuration fields are accepted')
    if set(settings['prices'])!=set(MODELS) or any(set(value)!={'input','output','url'} for value in settings['prices'].values()):
        raise ValueError('Only documented price metadata is accepted')
    if not settings['models'] or len(set(settings['models'])) != len(settings['models']) or any(model not in MODELS for model in settings['models']):
        raise ValueError('Invalid or duplicate models')
    if not settings['formats'] or len(set(settings['formats'])) != len(settings['formats']) or any(fmt not in FORMATS for fmt in settings['formats']):
        raise ValueError('Invalid or duplicate formats')
    for key in ('positions_per_tier','pilot_per_tier','max_output_tokens','input_overhead_tokens','timeout_seconds'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError('Invalid positive integer setting: '+key)
    if type(settings['seed']) is not int or settings['pilot_per_tier'] > settings['positions_per_tier']:
        raise ValueError('Invalid sampling configuration')
    if settings['max_output_tokens'] > 2048 or settings['input_overhead_tokens'] < 8192:
        raise ValueError('Output cap or conservative input overhead changed beyond safe defaults')
    if money(settings['max_usd']) > money('20') or money(settings['pilot_max_usd']) > money('1') or money(settings['pilot_max_usd']) > money(settings['max_usd']):
        raise ValueError('Session/pilot caps exceed conservative limits')
    margin = Decimal(settings['cost_margin'])
    if not margin.is_finite() or margin < Decimal('1.10'):
        raise ValueError('Insufficient cost margin')
    for model in MODELS:
        for direction in ('input','output'):
            money(settings['prices'][model][direction])
        if any(Decimal(settings['prices'][model][direction]) != Decimal(expected) for direction,expected in zip(('input','output'),VERIFIED_PRICES[model])):
            raise ValueError('Prices must match the verified official table')
        if settings['prices'][model]['url'] != 'https://developers.openai.com/api/docs/models/'+model:
            raise ValueError('Invalid model pricing provenance')
    if not settings['pricing_url'].startswith('https://developers.openai.com/') or not settings['pricing_verified_at']:
        raise ValueError('Pricing provenance required')


def selected_positions(source):
    unique, digest = {}, hashlib.sha256()
    with open(source, 'rb') as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            fen = record.get('fen')
            if not isinstance(fen, str) or record.get('difficulty') not in TIERS:
                raise ValueError('Invalid primary position')
            chess.Board(fen)
            position = {key: record.get(key) for key in ('position_id','fen','difficulty','theme','phase','source','pgn_moves')}
            position['stockfish_eval'] = record.get('t1_stockfish_eval')
            position['reference_provenance'] = 'historical engine provenance unverified'
            if type(position['stockfish_eval']) not in (int,float) or not math.isfinite(position['stockfish_eval']):
                raise ValueError('Missing numeric historical reference')
            if fen in unique and unique[fen] != position:
                raise ValueError('Position metadata/reference conflict')
            unique[fen] = position
    return list(unique.values()), digest.hexdigest()


def prompt_messages(position, fmt):
    fen, pgn = position['fen'], position.get('pgn_moves')
    if fmt == 'pgn+fen' and not pgn:
        raise ValueError('PGN history unavailable; cannot reproduce pgn+fen')
    builders = {'eval_only': (build_eval_prompt, EVAL_SYSTEM_PROMPT),
                'move_only': (build_move_prompt, MOVE_SYSTEM_PROMPT),
                'explanation_only': (build_explanation_prompt, EXPLANATION_SYSTEM_PROMPT)}
    if fmt in builders:
        builder, system = builders[fmt]
        return [dict(role='system',content=system),dict(role='user',content=builder(fen))]
    return [dict(role='user',content=build_prompt(fen,pgn,fmt))]


def make_job(position, model, fmt, settings):
    messages = prompt_messages(position,fmt)
    payload = dict(model=model,input=messages,max_output_tokens=settings['max_output_tokens'],
                   reasoning={'effort':'low'},service_tier='default',store=False)
    bound = len(json.dumps(messages,ensure_ascii=False).encode('utf-8')) + settings['input_overhead_tokens']
    amount = Decimal(cost(bound,settings['max_output_tokens'],settings['prices'][model])) * Decimal(settings['cost_margin'])
    identity = fingerprint(dict(payload=payload,position=position))
    return dict(id=identity,position=position,model=model,prompt_format=fmt,payload=payload,
                input_token_bound=bound,reserved_nano=int(amount.to_integral_value(rounding=ROUND_CEILING)))


def build_manifest(source, settings):
    validate_settings(settings)
    settings = json.loads(json.dumps(settings))
    positions, digest = selected_positions(source)
    randomizer, chosen, pilot_fens = random.Random(settings['seed']), [], []
    for tier in TIERS:
        population = sorted((row for row in positions if row['difficulty']==tier),key=lambda row:row['fen'])
        randomizer.shuffle(population)
        if len(population) < settings['positions_per_tier']:
            raise ValueError('Insufficient positions in stratum '+tier)
        chosen.extend(population[:settings['positions_per_tier']])
        pilot_fens.extend(row['fen'] for row in population[:settings['pilot_per_tier']])
    jobs = [make_job(position,model,fmt,settings) for position in chosen for model in settings['models'] for fmt in settings['formats']]
    versions = {name: hashlib.sha256((REPO/'src'/name).read_bytes()).hexdigest() for name in ('llm_client.py','evaluator.py','openai_benchmark.py','openai_budget.py')}
    return dict(version=1,settings=settings,source_sha256=digest,positions=chosen,pilot_fens=pilot_fens,
                jobs=jobs,code_sha256=versions,reference_provenance='unverified; fresh T2 quality rescoring required')


def validate_manifest(manifest):
    settings, positions = manifest['settings'], manifest['positions']
    required_versions={'llm_client.py','evaluator.py','openai_benchmark.py','openai_budget.py'}
    source_hash=manifest.get('source_sha256')
    if manifest.get('version') != 1 or set(manifest['code_sha256']) != required_versions or not isinstance(source_hash,str) or len(source_hash)!=64 or any(character not in '0123456789abcdef' for character in source_hash):
        raise ValueError('Frozen manifest schema/version is invalid')
    validate_settings(settings)
    if len({row['fen'] for row in positions}) != 4*settings['positions_per_tier']:
        raise ValueError('Frozen sample has invalid position count')
    expected_pilot=[]
    for tier in TIERS:
        cohort=[row for row in positions if row['difficulty']==tier]
        if len(cohort)!=settings['positions_per_tier']:
            raise ValueError('Frozen sample stratum changed')
        expected_pilot.extend(row['fen'] for row in cohort[:settings['pilot_per_tier']])
    expected=[make_job(row,model,fmt,settings) for row in positions for model in settings['models'] for fmt in settings['formats']]
    if manifest['jobs'] != expected or manifest['pilot_fens'] != expected_pilot or len({job['id'] for job in expected}) != len(expected):
        raise ValueError('Frozen requests, reservation or pilot changed')
    versions={name:hashlib.sha256((REPO/'src'/name).read_bytes()).hexdigest() for name in manifest['code_sha256']}
    if manifest['code_sha256'] != versions:
        raise ValueError('Frozen code version changed')


def validate_output(path):
    resolved = safe_path(path)
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError('Output and ledger must be outside the research repository')
    if resolved.is_symlink():
        raise ValueError('Symlink outputs are unsupported')
    return resolved


def http_response(payload, key, timeout):
    session = requests.Session()
    session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
    try:
        response = session.post(API_URL,json=payload,headers={'Authorization':'Bearer '+key},
                                timeout=(10,timeout),allow_redirects=False)
        if response.status_code != 200:
            raise BudgetError('HTTP request rejected; reservation retained')
        return response.json()
    finally:
        session.close()


def parsed_text(text, position, fmt):
    if fmt == 'eval_only':
        return parse_eval_response(text)
    if fmt == 'explanation_only':
        return parse_explanation_response(text)
    if fmt == 'move_only':
        return dict(move=extract_move_from_text(position['fen'],text),eval=None,explanation=None)
    return parse_response(text)


def base_record(job):
    record={key:job['position'].get(key) for key in ('position_id','fen','difficulty','theme','phase','source')}
    record.update(job_id=job['id'],model=job['model'],model_requested=job['model'],model_family='openai',model_size_b=None,
                  prompt_format=job['prompt_format'],first_attempt=True,timestamp=datetime.now(timezone.utc).isoformat(),
                  reference_provenance='unverified',t2_rescue_enabled=False)
    return record


def interpret_response(body, job, settings):
    if not isinstance(body,dict):
        raise ValueError('Invalid response object')
    content = [part for item in body.get('output',[]) if item.get('type')=='message' for part in item.get('content',[])]
    text = '\n'.join(part.get('text','') for part in content if part.get('type')=='output_text')
    parsed = parsed_text(text,job['position'],job['prompt_format'])
    scores = score_all(parsed,job['position'],engine=None)
    scores['t2_cpl'] = None
    usage, reliable, charge = body.get('usage'), True, job['reserved_nano']
    if not valid_usage(usage):
        reliable = False
    else:
        charge = cost(usage['input_tokens'],usage['output_tokens'],settings['prices'][job['model']])
        reliable = usage['input_tokens'] <= job['input_token_bound'] and usage['output_tokens'] <= settings['max_output_tokens'] and charge <= job['reserved_nano']
    required = {'eval_only':('eval',),'move_only':('move',),'explanation_only':('explanation',)}.get(job['prompt_format'],('eval','move','explanation'))
    parse_ok = all(parsed.get(key) is not None for key in required)
    reliable = reliable and body.get('status')=='completed' and body.get('service_tier')=='default' and priced_model(job['model'],body.get('model')) and not any(part.get('type')=='refusal' for part in content)
    result = base_record(job)
    result.update(model_resolved=body.get('model'),raw_response=body,parsed_first_attempt=parsed,**scores)
    result.update(timestamp=datetime.now(timezone.utc).isoformat(),usage=usage,reference_provenance='unverified',
                  outcome=('complete' if parse_ok else 'parse_failure') if reliable else 'stopped_unreliable',
                  parse_quality='complete' if parse_ok else 'failure',
                  t2_move_attempt_1=scores.get('t2_move') if job['prompt_format'] not in ('eval_only','explanation_only') else None,
                  t2_legal_attempt_1=scores.get('t2_legal') if job['prompt_format'] not in ('eval_only','explanation_only') else None,
                  t2_rescue_path=('not_applicable' if job['prompt_format'] in ('eval_only','explanation_only') else
                                  'legal_list_scan' if job['prompt_format']=='move_only' else 'disabled'),
                  reserved_nano=job['reserved_nano'],observed_charge_nano=charge)
    return result, charge, bool(reliable)


def mark_truncated(result):
    result['partial_scores']={key:value for key,value in result.items() if key.startswith(('t1_','t2_','t3_'))}
    for key in result['partial_scores']:
        if key not in ('t2_rescue_enabled','t2_rescue_path'):
            result[key]=None
    move_asked=result['prompt_format'] not in ('eval_only','explanation_only')
    result.update(outcome='truncated',parse_quality='failure',terminal_reason='max_output_tokens',
                  t2_move_attempt_1=None,t2_legal_attempt_1=False if move_asked else None,
                  t2_legal=False if move_asked else None)
    return result


def execute_job(job, settings, ledger, transport, key, phase):
    ledger.reserve(job,phase)
    body = None
    try:
        body = transport(job['payload'],key) if transport else http_response(job['payload'],key,settings['timeout_seconds'])
        result, charge, reliable = interpret_response(body,job,settings)
    except (requests.RequestException, BudgetError, ValueError, KeyError, TypeError):
        result = base_record(job)
        result.update(outcome='stopped_ambiguous',raw_response=body,parse_quality='unavailable',
                      t2_move_attempt_1=None,t2_legal_attempt_1=None,t2_rescue_path='disabled')
        charge, reliable = job['reserved_nano'], False
    if verified_truncation(job,body,settings):
        mark_truncated(result)
    ledger.finish(job,result,charge,reliable)
    return result, reliable or (phase=='main' and result.get('outcome')=='truncated')


def prepare_output(output, ledger_path, manifest):
    for path in (output/'records',output/'manifest.json'):
        safe_path(path)
    existing = output.exists() and any(output.iterdir())
    if existing and (not ledger_path.exists() or not (output/'manifest.json').exists()):
        raise BudgetError('Existing output lacks matching manifest or session ledger')
    if existing and read_json(output/'manifest.json') != manifest:
        raise BudgetError('Frozen manifest changed')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    (output/'records').mkdir(exist_ok=True,mode=0o700)
    if not existing:
        atomic_json(output/'manifest.json',manifest)


def save_report(output, report, jobs, ledger):
    report['charged_nano']=sum(entry['charge_nano'] for entry in ledger.state['entries'].values())
    report['models']={}
    for job in jobs:
        bucket=report['models'].setdefault(job['model'],dict(complete=0,parse_failure=0,terminal_failure=0,interrupted_unknown=0,stopped=0,pending=0,legal_first=0))
        entry=ledger.state['entries'].get(job['id'])
        if not entry:
            bucket['pending']+=1
            continue
        result=entry.get('result',{})
        bucket['interrupted_unknown' if entry['status']=='interrupted_unknown' else 'terminal_failure' if entry['status']=='terminal_failure' else 'stopped' if entry['status']!='complete' else 'parse_failure' if result.get('parse_quality')=='failure' else 'complete']+=1
        bucket['legal_first']+=int(result.get('t2_legal_attempt_1') is True)
    atomic_json(output/'summary.json',report)


def run_benchmark(manifest, output_dir, ledger_path, phase='pilot', execute=False, transport=None):
    validate_manifest(manifest)
    settings = manifest['settings']
    if phase not in ('pilot','main'):
        raise ValueError('Unknown phase')
    jobs = [job for job in manifest['jobs'] if phase=='main' or job['position']['fen'] in manifest['pilot_fens']]
    report = dict(phase=phase,request_count=len(jobs),upper_bound_nano=sum(job['reserved_nano'] for job in jobs),
                  session_cap_nano=money(settings['max_usd']),pilot_sublimit_nano=money(settings['pilot_max_usd']))
    output, ledger_file = validate_output(output_dir), validate_output(ledger_path)
    if ledger_file==output or output in ledger_file.parents:
        raise ValueError('Budget ledger must be outside the run output directory')
    if not execute:
        return report
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise BudgetError('OPENAI_API_KEY is not set')
    validate_output(str(ledger_file)+'.lock')
    ledger_file.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with FileLock(str(ledger_file)+'.lock',timeout=0):
        prepare_output(output,ledger_file,manifest)
        ledger = BudgetLedger(ledger_file,settings)
        pilot_ids = [job['id'] for job in manifest['jobs'] if job['position']['fen'] in manifest['pilot_fens']]
        if phase=='main' and any(ledger.state['entries'].get(key,{}).get('status')!='complete' or ledger.state['entries'].get(key,{}).get('result',{}).get('parse_quality')!='complete' for key in pilot_ids):
            raise BudgetError('Successful pilot required before main execution')
        ledger.preflight(manifest['jobs'],'main')
        ledger.preflight(jobs,phase)
        for job in jobs:
            entry = ledger.state['entries'].get(job['id'])
            result, reliable = (entry['result'],True) if entry else execute_job(job,settings,ledger,transport,key,phase)
            atomic_json(output/'records'/(job['id']+'.json'),result)
            save_report(output,report,jobs,ledger)
            if not reliable or (phase=='pilot' and result.get('parse_quality')!='complete'):
                raise BudgetError('Paid request stopped; reservation retained; no automatic retry')
        save_report(output,report,jobs,ledger)
    return report


KNOWN_ORIGINAL_RUNNER = {
    'openai_benchmark.py':'dd7011fea90d248b0d68890d7e0f8aa2e3be8e3efe55d1b53b84f2167142decf',
    'openai_budget.py':'e011ac8476d4dcf33a8856b9b035709777c499dd0b459abcc69ca972340faa6f',
}


KNOWN_RECOVERY_RUNNER = {
    'openai_benchmark.py':'9997a9660ebc32ae6949664877b22eafab097007a10a04a7fdac8b1c9f0f64c9',
    'openai_budget.py':'267023922e859600b1c3bd5e4c7d427c42bc4e23c1cc07d7c6876f055c334355',
}


def frozen_manifest_for_run(source, settings, output):
    expected=build_manifest(source,settings)
    path=validate_output(output)/'manifest.json'
    if not path.exists():
        return expected
    frozen=read_json(path)
    validate_manifest(frozen)
    if {key:value for key,value in frozen.items() if key!='runtime_migrations'} != expected:
        raise BudgetError('Frozen source, configuration or requests changed')
    return frozen


def migration_candidate(source, settings, output):
    frozen=read_json(output/'manifest.json')
    current=build_manifest(source,settings)
    stripped=lambda manifest:{key:value for key,value in manifest.items() if key not in ('code_sha256','runtime_migrations')}
    if stripped(frozen)!=stripped(current):
        raise BudgetError('Migration cannot change frozen source, sample, settings or payloads')
    for name, digest in frozen['code_sha256'].items():
        if digest!=current['code_sha256'][name] and (name not in KNOWN_ORIGINAL_RUNNER or digest not in (KNOWN_ORIGINAL_RUNNER[name],KNOWN_RECOVERY_RUNNER[name])):
            raise BudgetError('Unknown original code fingerprint; migration refused')
    if set(frozen['code_sha256'])!=set(current['code_sha256']):
        raise BudgetError('Incomplete original code fingerprints')
    validate_manifest(current)
    return frozen,current


def persist_terminal_migration(output, ledger, original, current, promoted, policy="known_max_output_terminal_v1"):
    manifest_hash=fingerprint(original)
    backup=output/('manifest.before-terminal-'+manifest_hash+'.json')
    if not backup.exists():
        atomic_json(backup,original)
    elif read_json(backup)!=original:
        raise BudgetError('Original manifest snapshot conflict')
    existing=next((row for row in ledger.state.get('terminal_policy_migrations',[])
                   if row.get('previous_manifest_sha256')==manifest_hash and row.get('new_code_sha256')==current['code_sha256'] and row.get('policy')==policy),None)
    journal=existing or dict(policy=policy,previous_manifest_sha256=manifest_hash,
                 previous_code_sha256=original['code_sha256'],new_code_sha256=current['code_sha256'],
                 promoted_ids=promoted,previous_status='reserved' if policy=='shutdown_interrupted_v1' else 'stopped',timestamp=datetime.now(timezone.utc).isoformat())
    current['runtime_migrations']=original.get('runtime_migrations',[])+[journal]
    if not existing:
        ledger.state.setdefault('terminal_policy_migrations',[]).append(journal)
    ledger.save()
    atomic_json(output/'manifest.json',current)


def accept_known_truncations(source, settings, output_dir, ledger_path, *, abandon_interrupted=False):
    """Explicit offline main policy migration; no key or transport access."""
    output, ledger_file=validate_output(output_dir),validate_output(ledger_path)
    if output==ledger_file or output in ledger_file.parents or not ledger_file.exists():
        raise BudgetError('Existing external shared ledger is required')
    for path in (output/'records',output/'manifest.json',Path(str(ledger_file)+'.lock')):
        safe_path(path)
    with FileLock(str(ledger_file)+'.lock',timeout=0):
        original,current=migration_candidate(source,settings,output)
        ledger=BudgetLedger(ledger_file,settings)
        jobs={job['id']:job for job in original['jobs']}
        promoted=[]
        policy='shutdown_interrupted_v1' if abandon_interrupted else 'known_max_output_terminal_v1'
        for identity,entry in ledger.state['entries'].items():
            if entry['status']!=('reserved' if abandon_interrupted else 'stopped') or identity not in jobs:
                continue
            if entry['request']!=jobs[identity] or entry['phase']!='main' or identity in {job['id'] for job in original['jobs'] if job['position']['fen'] in original['pilot_fens']}:
                raise BudgetError('Only non-pilot frozen main requests can be reconciled')
            if abandon_interrupted:
                if entry.get('result') is not None:
                    raise BudgetError('Reservation already has an observed result')
                result=base_record(jobs[identity])
                result.update(outcome='interrupted_unknown',parse_quality='unavailable',raw_response=None,
                              t2_move=None,t2_legal=None,t2_move_attempt_1=None,t2_legal_attempt_1=None,
                              previous_status='reserved',interruption_policy=policy)
                entry.update(status='interrupted_unknown',result=result)
                promoted.append(identity)
                continue
            if not verified_truncation(jobs[identity],entry.get('result',{}).get('raw_response'),settings):
                raise BudgetError('Stopped request is not a verified terminal truncation')
            entry['result']=mark_truncated(entry['result'])
            entry['status']='terminal_failure'
            promoted.append(identity)
        code_changed=original['code_sha256']!=current['code_sha256']
        if promoted or code_changed:
            backup=ledger_file.with_name(ledger_file.name+('.before-interrupted-v1.json' if abandon_interrupted else '.before-terminal-v1.json'))
            if not backup.exists():
                atomic_json(backup,read_json(ledger_file))
            persist_terminal_migration(output,ledger,original,current,promoted,policy)
            for identity in promoted:
                atomic_json(output/'records'/(identity+'.json'),ledger.state['entries'][identity]['result'])
        return dict(promoted=len(promoted),charged_nano=sum(entry['charge_nano'] for entry in ledger.state['entries'].values()),
                    policy=policy,api_calls=0)
