"""Offline blinding, sampling and exact-snippet preservation regressions."""
import json
from pathlib import Path
import chess
import pytest
from src.human_scoring_pack import build_pack, select_positions, MODELS


@pytest.fixture
def fixture_sources(tmp_path):
    root=tmp_path/'commercial'; root.mkdir(); (root/'records').mkdir()
    positions=[]; local=[]
    for tier_index,tier in enumerate(('easy','medium','hard','extreme')):
        for offset in range(3):
            board=chess.Board(); board.fullmove_number=100+tier_index*3+offset
            position=dict(fen=board.fen(),position_id=tier_index*3+offset,difficulty=tier,theme='SECRET_THEME')
            positions.append(position)
            for model in MODELS:
                record=dict(model=model,fen=position['fen'],position_id=position['position_id'],prompt_format='fen_only',
                            job_id='PRIVATEJOB'+model+str(position['position_id']),t3_explanation='Exact saved explanation.',outcome='complete')
                if model.startswith('gpt-'):
                    (root/'records'/(str(position['position_id'])+model+'.json')).write_text(json.dumps(record))
                else: local.append(record)
    manifest=dict(positions=positions,settings=dict(models=list(MODELS[-2:])))
    (root/'manifest.json').write_text(json.dumps(manifest))
    source=tmp_path/'local.jsonl'; source.write_text(''.join(json.dumps(row)+'\n' for row in local))
    exclusions=tmp_path/'exclude.json'; exclusions.write_text('[]')
    return source,root,exclusions,manifest


def test_selection_is_seeded_stratified_and_disjoint(fixture_sources):
    _,_,_,manifest=fixture_sources
    first,heldout=select_positions(manifest,[],42)
    assert (first,heldout)==select_positions(manifest,[],42)
    assert len(first)==len(heldout)==4
    assert {row['difficulty'] for row in first}=={'easy','medium','hard','extreme'}
    assert not {row['fen'] for row in first}&{row['fen'] for row in heldout}
    excluded=first[0]['fen']
    changed=select_positions(manifest,[excluded],42)
    assert excluded not in {row['fen'] for cohort in changed for row in cohort}


def test_public_packet_is_blinded_heldout_free_and_exact(fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources; out=tmp_path/'packet'
    build_pack(source,root,out,exclusions)
    public=json.loads((out/'public'/'manifest.json').read_text())
    private=json.loads((out/'private'/'key.json').read_text())
    html=(out/'public'/'calibration.html').read_text()
    assert len(public['cards'])==24 and len(public['positions'])==4
    assert [row['label'] for row in public['positions']]==['Board A','Board B','Board C','Board D']
    assert [row['label'] for row in public['cards']]==['Answer '+str(number) for number in range(1,25)]
    assert '<h2>Answer 1</h2>' in html and 'id="progress"' in html
    assert {row['explanation'] for row in public['cards']}=={'Exact saved explanation.'}
    for secret in [*MODELS,'PRIVATEJOB','SECRET_THEME',*[row['fen'] for row in private['heldout_positions']]]:
        assert secret not in html and secret not in json.dumps(public)
    assert {row['model'] for row in private['answers']}==set(MODELS)
    assert 'try {' in html and 'localStorage' in html and 'Export JSON' in html
    assert 'value="" selected' in html
    assert 'factual_grounding' in html and 'explanatory_support' in html
    assert (out/'private').stat().st_mode & 0o777==0o700
    assert (out/'private'/'key.json').stat().st_mode & 0o777==0o600


def test_html_injection_escaped_missing_stays_unscored(fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    evil='</script><img src=x onerror="alert(1)"> & exact\ntext'
    for row in rows: row['t3_explanation']=None if row['model']==MODELS[0] else evil
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    out=tmp_path/'packet'; build_pack(source,root,out,exclusions)
    public=json.loads((out/'public'/'manifest.json').read_text()); html=(out/'public'/'calibration.html').read_text()
    assert evil in {row['explanation'] for row in public['cards']}
    assert '<img src=x' not in html and '&lt;/script&gt;' in html
    missing=[row for row in public['cards'] if row['explanation'] is None]
    assert len(missing)==4 and all(row['availability']=='unavailable' for row in missing)
    assert html.count('data-missing="true"')==4


def test_duplicate_and_output_reuse_fail(fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources
    source.write_text(source.read_text()+source.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='Duplicate'): build_pack(source,root,tmp_path/'packet',exclusions)
    assert not (tmp_path/'packet').exists()


def test_identity_disclosure_and_symlink_fail(fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    for row in rows: row['t3_explanation']='As an OpenAI model, I think so.'
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='identity'): build_pack(source,root,tmp_path/'packet',exclusions)
    alias=tmp_path/'alias'; alias.symlink_to(tmp_path/'real')
    with pytest.raises(ValueError,match='Symlink'): build_pack(source,root,alias,exclusions)


@pytest.mark.parametrize('text',['I am llama3.2:3b.','As qwen2.5:7b I answer.','gemma3:12b says so.','I am Luna.','I am Sol.','I am a Google model.','This is an Alibaba model.','As a Meta model.'])
def test_explicit_model_disclosure_rejected(text,fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    for row in rows: row['t3_explanation']=text
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='identity'): build_pack(source,root,tmp_path/'packet',exclusions)


def test_identity_detector_does_not_match_incidental_words():
    from src.human_scoring_pack import IDENTITY
    assert IDENTITY.search('The solution is a claim about material.') is None


def test_output_reuse_and_fen_conflict_refused(fixture_sources,tmp_path):
    source,root,exclusions,_=fixture_sources; out=tmp_path/'packet'
    build_pack(source,root,out,exclusions)
    with pytest.raises(ValueError,match='new/empty'): build_pack(source,root,out,exclusions)
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    rows[0]['fen']=chess.STARTING_FEN
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='Conflicting FEN'): build_pack(source,root,tmp_path/'other',exclusions)
