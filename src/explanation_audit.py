"""Exploratory human/proxy audit with conservative, non-semantic board evidence."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import chess

from .evaluator import match_t3_theme, T3_THEME_MATCHER_VERSION
from .human_scoring_pack import checked, read_json, write_private

VERSION='explicit_occupancy_audit_v1'
DIMENSIONS=('factual_grounding','explanatory_support')
CLAIM=re.compile(r"\b(?:(White|Black)(?:['’]s)?\s+|(?:the|a)\s+)?(king|queen|rook|bishop|knight|pawn)\s+(?:on|at)\s+([a-h][1-8])\b",re.I)
CONTEXT=re.compile(r"""\b(if|unless|would|could|might|may|can|will|should|after|before|when|then|previously|earlier|formerly|once|was|were|had|been|captured|gone|lacks|lack|moved|not|no|never|without|cannot|suppose|imagine|hypothetical|potential|variation|sequence|line|absent|missing|nonexistent|non-existent|non existent|imaginary|fictional|mistaken|example|quoted)\b|n['’]t\b|["“”'‘’]|\b\d+\.(?:\.\.)?""",re.I)


def sha(path):
    digest=hashlib.sha256()
    with checked(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def board_facts(fen,text):
    """Check only accepted literal occupancy; never grade the whole explanation."""
    board=chess.Board(fen); claims=[]
    if not board.is_valid(): raise ValueError('Invalid chess position; no board evidence can be asserted')
    context_text=re.sub(r"(?<=\w)['’]s\b",'',text,flags=re.I) if isinstance(text,str) else ''
    context=bool(CONTEXT.search(context_text))
    for match in CLAIM.finditer(text or ''):
        color,piece_type,square=match.groups(); expected=dict(color=color.title() if color else None,type=piece_type.lower())
        piece=board.piece_at(chess.parse_square(square.lower()))
        actual=dict(color='White' if piece.color else 'Black',type=chess.piece_name(piece.piece_type),symbol=piece.symbol()) if piece else None
        status='unverified' if context else 'supported' if actual and all(value is None or actual[key]==value for key,value in expected.items()) else 'contradicted'
        claims.append(dict(literal=match.group(),span=[match.start(),match.end()],square=square.lower(),
                           asserted_color=expected['color'],expected_piece=expected,actual_piece=actual,status=status,
                           reason='context requires abstention' if context else 'literal piece occupancy only'))
    if not claims:
        claims=[dict(literal=None,span=None,status='unverified',reason='no accepted explicit occupancy syntax')]
    return dict(version=VERSION,claims=claims,semantic_factual_grounding=None,semantic_explanatory_support=None,
                semantic_status='requires_human_review')


def validate_ratings(document,pack_id,rubric,cards):
    if document.get('pack_id')!=pack_id or document.get('rubric_version')!=rubric:
        raise ValueError('Ratings pack/rubric mismatch')
    rows=document.get('ratings')
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
        raise ValueError('Invalid ratings array')
    ids=[row.get('answer_id') for row in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(cards):
        raise ValueError('Ratings must contain exact unique calibration IDs; heldout/unknown IDs forbidden')
    for row in rows:
        if row.get('availability')!=cards[row['answer_id']]['availability']:
            raise ValueError('Rating availability mismatch')
        for component in DIMENSIONS:
            value=row.get(component)
            if value is not None and value!='NA' and not (type(value) is int and 0<=value<=2):
                raise ValueError('Human component must be integer0/1/2, NA or null')
            if row['availability']!='available' and value not in (None,'NA'):
                raise ValueError('Unavailable answer cannot receive a numeric quality score')
        if not isinstance(row.get('notes',''),str) or not isinstance(row.get('uncertainty_flags',[]),list) or any(not isinstance(flag,str) for flag in row.get('uncertainty_flags',[])):
            raise ValueError('Invalid notes/uncertainty metadata')
    return {row['answer_id']:row for row in rows}


def source_record(record,frame,models,answers):
    if record.get('model') not in models or record.get('prompt_format')!='fen_only': return
    fen=record.get('fen'); identity=record.get('position_id')
    expected={position['position_id']:position['fen'] for position in frame.values()}
    if identity in expected and fen!=expected[identity]: raise ValueError('Source FEN conflict')
    if fen not in frame: return
    if identity!=frame[fen]['position_id']: raise ValueError('Source position ID conflict')
    pair=(record['model'],fen)
    if pair in answers: raise ValueError('Duplicate source model/FEN/format join')
    text=record.get('t3_explanation')
    if text is not None and not isinstance(text,str): raise ValueError('Invalid explanation text')
    legacy=record.get('t3_score')
    if legacy is not None and (type(legacy) is not int or legacy not in (0,1,2)): raise ValueError('Invalid saved legacy T3')
    outcome=record.get('outcome')
    availability=outcome if outcome in ('truncated','interrupted_unknown') else 'available' if text and text.strip() else 'missing'
    answers[pair]=dict(model=record['model'],fen=fen,job_id=record.get('job_id'),text=text,
                       availability=availability,legacy_t3=legacy,theme=record.get('theme',frame[fen].get('theme')))


def load_sources(local_results,commercial_root,manifest,local_models):
    frame={row['fen']:row for row in manifest['positions']}
    if len(frame)!=len(manifest['positions']) or len({row['position_id'] for row in frame.values()})!=len(frame):
        raise ValueError('Duplicate/conflicting frozen positions')
    for fen in frame:
        if not chess.Board(fen).is_valid(): raise ValueError('Invalid frozen chess position')
    models=list(local_models)+list(manifest['settings']['models'])
    if not models or len(set(models))!=len(models): raise ValueError('Declared models must be unique')
    answers={}; local_digest=hashlib.sha256(); commercial_digest=hashlib.sha256()
    with checked(local_results).open('rb') as stream:
        for line in stream:
            local_digest.update(line)
            if line.strip(): source_record(json.loads(line),frame,local_models,answers)
    for path in sorted(checked(commercial_root/'records').glob('*.json')):
        content=checked(path).read_bytes(); commercial_digest.update(path.name.encode()+content)
        source_record(json.loads(content),frame,manifest['settings']['models'],answers)
    return answers,models,dict(local_sha256=local_digest.hexdigest(),commercial_records_sha256=commercial_digest.hexdigest(),
                               commercial_manifest_sha256=sha(commercial_root/'manifest.json'))


def compare_calibration(rows):
    report={}
    for row in rows:
        bucket=report.setdefault(row['model'],dict(n=0,legacy_zero_human_support_two=0,legacy_two_human_support_zero=0,components={}))
        bucket['n']+=1
        support=row['human']['explanatory_support']
        bucket['legacy_zero_human_support_two']+=int(row['legacy_t3']==0 and support==2)
        bucket['legacy_two_human_support_zero']+=int(row['legacy_t3']==2 and support==0)
        for component in DIMENSIONS:
            target=bucket['components'].setdefault(component,dict(human_counts=Counter(),legacy_t3_contingency=Counter(),keyword_contingency=Counter()))
            human='missing' if row['human'][component] is None else str(row['human'][component])
            legacy='missing' if row['legacy_t3'] is None else str(row['legacy_t3'])
            keyword='missing' if row['keyword_current'] is None else str(row['keyword_current'])
            target['human_counts'][human]+=1
            target['legacy_t3_contingency'][legacy+' -> '+human]+=1
            target['keyword_contingency'][keyword+' -> '+human]+=1
    return report


def calibration_rows(key,public,corrected,original,answers):
    private=key['answers']; ids=[row['answer_id'] for row in private]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate private answer IDs')
    selected={row['answer_id']:row for row in private if row['split']=='calibration'}
    if len({(row['model'],row['fen']) for row in selected.values()})!=len(selected): raise ValueError('Duplicate calibration model/FEN pair')
    cards={row['answer_id']:row for row in public['cards']}
    if len(cards)!=len(public['cards']) or set(cards)!=set(selected): raise ValueError('Public/private calibration mismatch')
    corrected_map=validate_ratings(corrected,key['pack_id'],key['rubric_version'],cards)
    original_map=validate_ratings(original,key['pack_id'],key['rubric_version'],cards)
    rows=[]
    for identity,item in selected.items():
        record=answers.get((item['model'],item['fen']))
        text=record['text'] if record else None
        availability=record['availability'] if record else 'missing'
        if item['availability']!=availability or cards[identity]['availability']!=('available' if availability=='available' else 'unavailable'):
            raise ValueError('Private/public/source availability mismatch')
        if item['explanation']!=text or cards[identity]['explanation']!=text or cards[identity]['fen']!=item['fen']:
            raise ValueError('Exact saved explanation/position join mismatch')
        if record and record['job_id']!=item.get('job_id'): raise ValueError('Saved source identity mismatch')
        rows.append(dict(answer_id=identity,model=item['model'],fen=item['fen'],text=text,
                         legacy_t3=record['legacy_t3'] if record else None,
                         keyword_current=match_t3_theme(record['theme'],text) if record else None,
                         human=corrected_map[identity],original_human=original_map[identity]))
    if len(rows)!=24 or len({row['fen'] for row in rows})!=4: raise ValueError('Expected24 calibration cards across4 boards')
    return rows


def write_diagnostics(path,manifest,models,answers):
    coverage={model:dict(planned=len(manifest['positions']),available=0,missing=0,truncated=0,interrupted_unknown=0,
                         supported_claims=0,contradicted_claims=0,unverified_claims=0,
                         available_answers_with_extracted_claims=0,available_answers_no_extracted_claims=0) for model in models}
    with path.open('x',encoding='utf-8') as stream:
        path.chmod(0o600)
        for position in manifest['positions']:
            for model in models:
                record=answers.get((model,position['fen']))
                availability=record['availability'] if record else 'missing'
                coverage[model][availability]+=1
                evidence=board_facts(position['fen'],record['text'] if record and availability=='available' else None)
                literal_claims=[claim for claim in evidence['claims'] if claim['literal'] is not None]
                if availability=='available':
                    coverage[model]['available_answers_with_extracted_claims' if literal_claims else 'available_answers_no_extracted_claims']+=1
                    for claim in literal_claims: coverage[model][claim['status']+'_claims']+=1
                payload=dict(model=model,fen=position['fen'],prompt_format='fen_only',availability=availability,
                             saved_explanation=record['text'] if record else None,**evidence)
                stream.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
    return coverage


def audit(local_results,commercial_root,pack_root,ratings,original_ratings,output_dir,*,local_models):
    output=checked(output_dir); root=checked(commercial_root); pack=checked(pack_root)
    repo=Path(__file__).resolve().parent.parent
    if output==repo or repo in output.parents or output.exists(): raise ValueError('Output must be new and outside repository')
    key=read_json(pack/'private/key.json'); public=read_json(pack/'public/manifest.json')
    if key['pack_id']!=public['pack_id'] or key['rubric_version']!=public['rubric_version']: raise ValueError('Pack metadata mismatch')
    corrected=read_json(ratings); original=read_json(original_ratings); manifest=read_json(root/'manifest.json')
    answers,models,provenance=load_sources(local_results,root,manifest,local_models)
    if provenance!=key['provenance']: raise ValueError('Source provenance hash mismatch; no silent source changes allowed')
    rows=calibration_rows(key,public,corrected,original,answers)
    calibration=compare_calibration(rows)
    provenance.update(corrected_ratings_sha256=sha(ratings),original_ratings_sha256=sha(original_ratings),
                      private_key_sha256=sha(pack/'private/key.json'),public_pack_sha256=sha(pack/'public/manifest.json'),
                      audit_code_sha256=sha(Path(__file__)),evaluator_code_sha256=sha(Path(__file__).with_name('evaluator.py')),
                      python_version=sys.version.split()[0],python_chess_version=chess.__version__,lexical_matcher_version=T3_THEME_MATCHER_VERSION)
    changes={component:sum(row['human'][component]!=row['original_human'][component] for row in rows) for component in DIMENSIONS}
    output.mkdir(parents=True,mode=0o700); (output/'private').mkdir(mode=0o700)
    coverage=write_diagnostics(output/'private/frame-diagnostics.jsonl',manifest,models,answers)
    summary=dict(version=VERSION,scope=dict(calibration_cards=len(rows),independent_calibration_boards=4,
                 assessors=1,frame_positions=len(manifest['positions']),declared_models=models),
                 limitations=['Exploratory one-assessor calibration, not validation or a model ranking.',
                              'Human dimensions, legacy side-plus-keyword sum, and binary mentions are different constructs.',
                              'No fitted weights, semantic grading, accuracy claims or significance tests.',
                              'Specific/verbose answers expose more claims; vague answers can escape checks. Fewest flags is not a quality ranking.',
                              'Private heldout diagnostics must not be inspected to tune these frozen rules.'],
                 calibration=calibration,corrected_component_changes=changes,coverage=coverage,provenance=provenance)
    write_private(output/'private/calibration.json',json.dumps(rows,ensure_ascii=False,indent=2))
    write_private(output/'summary.json',json.dumps(summary,ensure_ascii=False,indent=2))
    return dict(output_dir=str(output),calibration_cards=len(rows),frame_cases=len(manifest['positions'])*len(models),semantic_status='requires_human_review')
