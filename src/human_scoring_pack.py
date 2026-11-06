"""Build a blinded, offline worksheet from saved explanation snippets only."""
import hashlib
import html
import json
import os
import random
import re
import secrets
from pathlib import Path

import chess
import chess.svg

MODELS=('llama3.2:3b','qwen2.5:7b','gemma3:12b','llama3.3:70b','gpt-6-luna','gpt-6.1-sol')
TIERS=('easy','medium','hard','extreme')
RUBRIC_VERSION='human_grounding_support_calibration_v1'
REPO=Path(__file__).resolve().parent.parent
IDENTITY=re.compile(r'(?i)\b(?:openai|gpt(?:[-\s]?\d(?:\.\d+)?)?|(?:llama|qwen|gemma)(?:[-\s]?\d+(?:\.\d+)*)?|chatgpt|luna|sol|meta|alibaba|google)\b')


def checked(path):
    candidate=Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (candidate,*candidate.parents)):
        raise ValueError('Symlink paths are unsupported')
    return candidate.resolve()


def read_json(path):
    return json.loads(checked(path).read_text())


def select_positions(manifest, excluded, seed=42):
    positions=manifest['positions']
    if len({row['fen'] for row in positions})!=len(positions) or len({row['position_id'] for row in positions})!=len(positions):
        raise ValueError('Duplicate or conflicting manifest positions')
    randomizer=random.Random(seed); calibration=[]; heldout=[]
    for tier in TIERS:
        population=sorted((row for row in positions if row['difficulty']==tier and row['fen'] not in excluded),key=lambda row:row['fen'])
        if len(population)<2:
            raise ValueError('Insufficient unexposed positions in '+tier)
        chosen=randomizer.sample(population,2)
        calibration.append(chosen[0]); heldout.append(chosen[1])
    return calibration,heldout


def collect_record(record, frame, answers):
    if record.get('model') not in MODELS or record.get('prompt_format')!='fen_only':
        return
    fen=record.get('fen'); identity=record.get('position_id')
    expected={row['position_id']:row['fen'] for row in frame.values()}
    if identity in expected and fen!=expected[identity]:
        raise ValueError('Conflicting FEN for a frozen position ID')
    if fen not in frame:
        return
    if identity!=frame[fen]['position_id']:
        raise ValueError('Conflicting position ID for a frozen FEN')
    key=(record['model'],fen)
    if key in answers:
        raise ValueError('Duplicate matched answer')
    text=record.get('t3_explanation')
    if text is not None and not isinstance(text,str):
        raise ValueError('Saved explanation must be text or null')
    answers[key]=dict(model=record['model'],fen=fen,job_id=record.get('job_id'),explanation=text,
                      availability=('available' if text and text.strip() else
                                    record.get('outcome') if record.get('outcome') in ('truncated','interrupted_unknown') else 'missing'))


def load_answers(local_results, commercial_root, manifest):
    frame={row['fen']:row for row in manifest['positions']}; answers={}; digest=hashlib.sha256()
    with checked(local_results).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip(): collect_record(json.loads(line),frame,answers)
    commercial_digest=hashlib.sha256()
    for path in sorted(checked(commercial_root/'records').glob('*.json')):
        content=checked(path).read_bytes(); commercial_digest.update(path.name.encode()+content)
        record=json.loads(content)
        if record.get('model') not in MODELS[-2:]:
            raise ValueError('Unexpected commercial record model')
        collect_record(record,frame,answers)
    return answers,dict(local_sha256=digest.hexdigest(),commercial_records_sha256=commercial_digest.hexdigest(),
                        commercial_manifest_sha256=hashlib.sha256(checked(commercial_root/'manifest.json').read_bytes()).hexdigest())


def blinded_cases(calibration, heldout, answers, randomizer):
    public_positions=[]; public_cards=[]; private_answers=[]
    for split,positions in (('calibration',calibration),('heldout',heldout)):
        for position in positions:
            position_id='P-'+format(randomizer.getrandbits(64),'016x')
            if split=='calibration': public_positions.append(dict(position_id=position_id,fen=position['fen']))
            for model in MODELS:
                answer=answers.get((model,position['fen']),dict(model=model,fen=position['fen'],job_id=None,explanation=None,availability='missing'))
                answer_id='A-'+format(randomizer.getrandbits(96),'024x')
                private_answers.append(dict(answer_id=answer_id,position_id=position_id,split=split,**answer))
                if split=='calibration':
                    if answer['explanation'] and IDENTITY.search(answer['explanation']):
                        raise ValueError('Saved snippet discloses model/provider identity; publication refused')
                    public_cards.append(dict(answer_id=answer_id,position_id=position_id,fen=position['fen'],
                                             explanation=answer['explanation'],availability='available' if answer['availability']=='available' else 'unavailable'))
    randomizer.shuffle(public_cards)
    randomizer.shuffle(public_positions)
    for number,position in enumerate(public_positions): position["label"]="Board "+"ABCD"[number]
    for number,card in enumerate(public_cards,1): card["label"]="Answer "+str(number)
    return public_positions,public_cards,private_answers


def selector(name, missing):
    if missing:
        return f'<select name="{name}" disabled><option value="" selected>NA — unavailable</option></select>'
    options='<option value="" selected>Choose…</option>'+''.join(f'<option value="{value}">{value}</option>' for value in ('0','1','2','NA'))
    return f'<select name="{name}">{options}</select>'


def render_card(card, board_label):
    board=chess.Board(card['fen']); missing=card['availability']!='available'
    graphic=chess.svg.board(board,orientation=chess.WHITE,size=300)
    text=html.escape(card['explanation']) if card['explanation'] is not None else 'No saved answer is available. Leave this case unscored.'
    position=html.escape(board_label); answer=html.escape(card['answer_id']); label=html.escape(card['label'])
    flags=''.join(f'<label><input type="checkbox" name="{flag}"> {label}</label>' for flag,label in
                  (('board_uncertain','Board fact uncertain'),('variation_uncertain','Variation uncertain'),('strategic_unverified','Strategic verdict unverified')))
    return (f'<article data-answer="{answer}" data-missing="{str(missing).lower()}"><h2>{label}</h2>'
            f'<p>{position} · {"White" if board.turn else "Black"} to move</p>{graphic}'
            f'<details><summary>FEN</summary><code>{html.escape(card["fen"])}</code></details><pre>{text}</pre>'
            f'<label>Factual grounding {selector("factual_grounding",missing)}</label>'
            f'<label>Explanatory support {selector("explanatory_support",missing)}</label>'
            f'<fieldset><legend>Uncertainty flags</legend>{flags}</fieldset>'
            '<label>Optional notes<textarea name="notes" rows="3"></textarea></label></article>')


SCRIPT=r'''
const articles=[...document.querySelectorAll('article[data-answer]')];
const packId=document.body.dataset.pack, rubric=document.body.dataset.rubric;
const storageKey='human-chess-ratings:'+packId;
const alias=document.getElementById('assessor');
function rating(value){return value===''?null:value==='NA'?'NA':Number(value);}
function snapshot(){return {pack_id:packId,rubric_version:rubric,assessor_alias:alias.value,
 ratings:articles.map(card=>({answer_id:card.dataset.answer,availability:card.dataset.missing==='true'?'unavailable':'available',
 factual_grounding:rating(card.querySelector('[name=factual_grounding]').value),
 explanatory_support:rating(card.querySelector('[name=explanatory_support]').value),
 notes:card.querySelector('[name=notes]').value,
 uncertainty_flags:[...card.querySelectorAll('input:checked')].map(input=>input.name)}))};}
function notice(){document.getElementById('status').textContent='Browser saving unavailable. Export JSON to keep your ratings.';}
function progress(){const available=articles.filter(card=>card.dataset.missing!=='true');const done=available.filter(card=>['factual_grounding','explanatory_support'].every(field=>card.querySelector('[name='+field+']').value!==''));document.getElementById('progress').textContent=done.length+'/'+available.length+' available answers rated · '+(articles.length-available.length)+' unavailable';}
function save(){progress();try {localStorage.setItem(storageKey,JSON.stringify(snapshot()));}catch(error){notice();}}
try {
 const saved=JSON.parse(localStorage.getItem(storageKey)||'null');
 if(saved&&saved.pack_id===packId&&saved.rubric_version===rubric&&Array.isArray(saved.ratings)){
 alias.value=typeof saved.assessor_alias==='string'?saved.assessor_alias:'';
 for(const item of saved.ratings){const card=articles.find(card=>card.dataset.answer===item.answer_id);if(!card)continue;
 for(const field of ['factual_grounding','explanatory_support']){const select=card.querySelector('[name='+field+']');
 if(!select.disabled&&[0,1,2,'NA'].includes(item[field]))select.value=String(item[field]);}
 card.querySelector('[name=notes]').value=typeof item.notes==='string'?item.notes:'';
 for(const input of card.querySelectorAll('input'))input.checked=Array.isArray(item.uncertainty_flags)&&item.uncertainty_flags.includes(input.name);}}
}catch(error){notice();}
progress();
document.addEventListener('input',save);document.addEventListener('change',save);
document.getElementById('export').addEventListener('click',()=>{
 const url=URL.createObjectURL(new Blob([JSON.stringify(snapshot(),null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download=packId+'-ratings.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
'''


def render_html(public):
    labels={position['position_id']:position['label'] for position in public['positions']}
    cards=''.join(render_card(card,labels[card['position_id']]) for card in public['cards'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Chess explanation calibration</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;background:#f3f1eb;color:#182b32;font:17px system-ui}article{background:white;padding:1.5rem;margin:1.5rem 0;border-radius:12px}pre{white-space:pre-wrap;font:inherit;background:#f3f5f6;padding:1rem}label{display:block;margin:.7rem 0}select,textarea,input,button{font:inherit}textarea{display:block;width:95%}fieldset label{display:inline-block;margin:.4rem}code{overflow-wrap:anywhere}button{padding:.7rem 1.3rem}.toolbar{position:sticky;top:0;background:#f3f1eb;padding:.5rem;z-index:2}svg{max-width:100%;height:auto}</style>'
        f'<body data-pack="{public["pack_id"]}" data-rubric="{RUBRIC_VERSION}"><h1>Chess explanation calibration</h1>'
        '<p>24 saved answers on four boards. These exploratory ratings help calibrate a rubric; they are not a validated model ranking.</p>'
        '<p><strong>Factual grounding:</strong> 0 central board or move claim is wrong; 1 mixed accuracy, minor errors or partial grounding; 2 specific checkable facts are accurate; NA no checkable claims or cannot verify.</p><p><strong>Explanatory support:</strong> 0 absent or invalid central justification; 1 valid but generic or incomplete; 2 specific coherent tactical or positional mechanism; NA cannot assess. Blank means not yet rated. Unavailable answers export null ratings, with availability recorded separately.</p>'
        '<p>Do not reward length, confidence or keywords. Check variations sequentially from the displayed position, preserving preceding moves and branches. Mark uncertain strategic verdicts NA when you lack verified reference evidence. Missing answers are unavailable, not wrong.</p>'
        '<section class="toolbar"><label>Assessor alias <input id="assessor" autocomplete="off"></label><button id="export">Export JSON</button> <output id="progress"></output></section>'
        '<p id="status">Progress saves in this browser when permitted. Export JSON before closing and send the downloaded ratings file back.</p>'
        +cards+'<script>'+SCRIPT+'</script></body></html>')


def write_private(path, value):
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream: stream.write(value)


def build_pack(local_results, commercial_root, output_dir, exclusions_file, seed=42):
    output=checked(output_dir); root=checked(commercial_root)
    if output==REPO or REPO in output.parents or output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new/empty and outside the research repository')
    excluded=read_json(exclusions_file)
    if not isinstance(excluded,list) or any(not isinstance(fen,str) for fen in excluded):
        raise ValueError('Exclusions must be an explicit JSON list of FEN strings')
    manifest=read_json(root/'manifest.json')
    calibration,heldout=select_positions(manifest,excluded,seed)
    answers,provenance=load_answers(local_results,root,manifest)
    blinding_seed=secrets.randbits(128); randomizer=random.Random(blinding_seed)
    positions,cards,key=blinded_cases(calibration,heldout,answers,randomizer)
    pack_id='pack-'+secrets.token_hex(12)
    public=dict(pack_id=pack_id,rubric_version=RUBRIC_VERSION,positions=positions,cards=cards)
    private=dict(pack_id=pack_id,rubric_version=RUBRIC_VERSION,selection_seed=seed,blinding_seed=blinding_seed,
                 models=list(MODELS),calibration_positions=calibration,heldout_positions=heldout,answers=key,
                 exclusions=[dict(fen=fen,reason='prior_exposure_not_quality') for fen in excluded],provenance=provenance)
    document=render_html(public)
    output.mkdir(parents=True,exist_ok=True,mode=0o700); output.chmod(0o700)
    for folder in ('public','private'): (output/folder).mkdir(mode=0o700)
    write_private(output/'private'/'key.json',json.dumps(private,ensure_ascii=False,indent=2))
    write_private(output/'public'/'manifest.json',json.dumps(public,ensure_ascii=False,indent=2))
    write_private(output/'public'/'calibration.html',document)
    return dict(pack_id=pack_id,calibration_positions=4,calibration_cards=24,heldout_positions=4,
                public_html=str(output/'public'/'calibration.html'))
