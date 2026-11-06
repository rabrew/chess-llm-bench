"""Conservative evidence and human/proxy join validation, entirely synthetic."""
import hashlib
import json
import chess
import pytest
from src.explanation_audit import board_facts, validate_ratings, compare_calibration, audit
from src.human_scoring_pack import build_pack, MODELS


FEN=chess.STARTING_FEN


@pytest.mark.parametrize('text,status',[
    ("White's rook on a1 supports the queen.",'supported'),
    ('Black’s knight at b8 is developed.','supported'),
    ("White's rook on d1 controls the file.",'contradicted'),
    ("If White's rook on d1 were active, it would win.",'unverified'),
    ("White's rook on a1 was moved earlier.",'unverified'),
    ("White's rook on a1 does not exist.",'unverified'),
    ("After 1.e4, White's rook on d1 is active.",'unverified'),
    ('A pawn can promote after a sequence.','unverified'),
])
def test_explicit_occupancy_only_and_context_abstention(text,status):
    result=board_facts(FEN,text)
    assert result['semantic_factual_grounding'] is None
    assert result['semantic_explanatory_support'] is None
    assert result['semantic_status']=='requires_human_review'
    assert result['claims'][0]['status']==status
    if status=='contradicted':
        assert result['claims'][0]['actual_piece']['type']=='queen'
        assert result['claims'][0]['actual_piece']['color']=='White'
        assert text[slice(*result['claims'][0]['span'])]==result['claims'][0]['literal']


def ratings_document(value=2):
    return dict(pack_id='p',rubric_version='r',ratings=[dict(answer_id='a',availability='available',
            factual_grounding=value,explanatory_support=2,notes='',uncertainty_flags=[])])


def test_strict_ratings_ids_missing_and_component_types():
    cards={'a':dict(availability='available')}
    assert validate_ratings(ratings_document(), 'p','r',cards)['a']['factual_grounding']==2
    for invalid in (True,2.0,'2',3):
        with pytest.raises(ValueError): validate_ratings(ratings_document(invalid),'p','r',cards)
    for key,value in (('pack_id','wrong'),('rubric_version','wrong')):
        doc=ratings_document(); doc[key]=value
        with pytest.raises(ValueError): validate_ratings(doc,'p','r',cards)
    doc=ratings_document(); doc['ratings']*=2
    with pytest.raises(ValueError): validate_ratings(doc,'p','r',cards)
    doc=ratings_document(); doc['ratings'][0]['answer_id']='heldout'
    with pytest.raises(ValueError): validate_ratings(doc,'p','r',cards)


def test_comparison_is_contingency_not_fitted_semantic_score():
    rows=[dict(model='m',legacy_t3=0,keyword_current=1,human=dict(factual_grounding=1,explanatory_support=2)),
          dict(model='m',legacy_t3=2,keyword_current=0,human=dict(factual_grounding='NA',explanatory_support=0))]
    report=compare_calibration(rows)
    assert report['m']['legacy_zero_human_support_two']==1
    assert report['m']['legacy_two_human_support_zero']==1
    assert report['m']['components']['factual_grounding']['human_counts']['NA']==1
    assert 'mae' not in json.dumps(report).lower()
    assert 'accuracy' not in json.dumps(report).lower()


@pytest.fixture
def sources(tmp_path):
    root=tmp_path/'commercial'; root.mkdir(); (root/'records').mkdir()
    positions=[]; local=[]
    for index,tier in enumerate(('easy','medium','hard','extreme')):
        for offset in range(2):
            board=chess.Board(); board.fullmove_number=100+index*2+offset
            position=dict(fen=board.fen(),position_id=index*2+offset,difficulty=tier,theme='fork')
            positions.append(position)
            for model in (*MODELS,'extra-local'):
                row=dict(model=model,fen=position['fen'],position_id=position['position_id'],prompt_format='fen_only',
                         job_id=model+str(position['position_id']),t3_explanation="White's rook on a1 is active.",
                         t3_score=0,t3_p1_side_correct=0,t3_p2_theme_correct=0,theme='fork',outcome='complete')
                if model in MODELS[-2:]: (root/'records'/(row['job_id']+'.json')).write_text(json.dumps(row))
                else: local.append(row)
    (root/'manifest.json').write_text(json.dumps(dict(positions=positions,settings=dict(models=list(MODELS[-2:])))))
    local_path=tmp_path/'local.jsonl'; local_path.write_text(''.join(json.dumps(row)+'\n' for row in local))
    excluded=tmp_path/'exclusions.json'; excluded.write_text('[]'); pack=tmp_path/'pack'
    build_pack(local_path,root,pack,excluded)
    public=json.loads((pack/'public/manifest.json').read_text())
    doc=dict(pack_id=public['pack_id'],rubric_version=public['rubric_version'],assessor_alias='tester',
             ratings=[dict(answer_id=row['answer_id'],availability=row['availability'],factual_grounding=1,
                           explanatory_support=2,notes='',uncertainty_flags=[]) for row in public['cards']])
    original=tmp_path/'original.json'; original.write_text(json.dumps(doc))
    corrected=tmp_path/'corrected.json'; corrected.write_text(json.dumps(doc))
    return local_path,root,pack,original,corrected


def test_audit_preserves_private_holdout_and_full_declared_models(sources,tmp_path):
    local,root,pack,original,corrected=sources; output=tmp_path/'audit'
    audit(local,root,pack,corrected,original,output,local_models=[*MODELS[:4],'extra-local'])
    summary=json.loads((output/'summary.json').read_text())
    assert summary['scope']['calibration_cards']==24 and summary['scope']['independent_calibration_boards']==4
    assert summary['coverage']['extra-local']['planned']==8
    assert summary['coverage']['extra-local']['available']==8
    key=json.loads((pack/'private/key.json').read_text())
    for row in key['heldout_positions']: assert row['fen'] not in json.dumps(summary)
    assert 'A-' not in json.dumps(summary)
    assert len((output/'private/frame-diagnostics.jsonl').read_text().splitlines())==56
    assert summary['calibration']['llama3.2:3b']['legacy_zero_human_support_two']==4
    with pytest.raises(ValueError,match='new'): audit(local,root,pack,corrected,original,output,local_models=list(MODELS[:4]))


def test_changed_source_and_human_snippet_join_fail_closed(sources,tmp_path):
    local,root,pack,original,corrected=sources
    local.write_text(local.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'): audit(local,root,pack,corrected,original,tmp_path/'audit',local_models=list(MODELS[:4]))
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('text,status',[
    ('The rook on a8 controls the file.','supported'),
    ('The rook at d1 controls the file.','contradicted'),
    ('The rook on d4 is active.','contradicted'),
    ('If the rook on d1 arrives, it wins.','unverified'),
    ('There is no rook on d1.','unverified'),
    ('The rook on d1 is missing.','unverified'),
    ('The rook on d1 is absent.','unverified'),
    ('The rook on d1 is nonexistent.','unverified'),
    ('The rook on d1 doesn’t exist.','unverified'),
    ('"The rook on d1" is an example.','unverified'),
    ('The rook on a8 will move.','unverified'),
    ('White can place a rook on d1.','unverified'),
    ('The rook on d1 has been captured.','unverified'),
    ('White lacks a rook on d1.','unverified'),
    ('The non-existent rook on d1 dominates.','unverified'),
    ('‘The rook on d1’ is false.','unverified'),
    ("'The rook on d1' is false.",'unverified'),
])
def test_optional_owner_does_not_infer_color_and_context_abstains(text,status):
    result=board_facts(FEN,text)
    claim=result['claims'][0]
    assert claim['status']==status
    assert claim['asserted_color'] is None


def test_unavailable_ratings_cannot_be_numeric():
    doc=ratings_document(0); doc['ratings'][0]['availability']='unavailable'
    with pytest.raises(ValueError): validate_ratings(doc,'p','r',{'a':dict(availability='unavailable')})
    with pytest.raises(ValueError): validate_ratings(ratings_document(),'p','r',{'a':dict(availability='unavailable')})


def test_snippet_join_mismatch_even_with_unchanged_source_hashes(sources,tmp_path):
    local,root,pack,original,corrected=sources
    key_path=pack/'private/key.json'; key=json.loads(key_path.read_text())
    next(row for row in key['answers'] if row['split']=='calibration')['explanation']='Changed saved snippet.'
    key_path.write_text(json.dumps(key))
    with pytest.raises(ValueError,match='join mismatch'):
        audit(local,root,pack,corrected,original,tmp_path/'audit',local_models=list(MODELS[:4]))
