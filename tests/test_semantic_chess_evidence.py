"""Deterministic quoted-claim evidence; no inference or engine calls."""
import copy
import chess
import pytest
from src.semantic_chess_evidence import validate_extraction, evidence_for, EXTRACTION_SCHEMA


def context(identity='root',parent=None,moves=None):
    return dict(id=identity,parent_id=parent,moves=moves or [],assumption='Explicit extracted context; interpretation provisional.')


def quoted_move(text,san):
    start=text.index(san)
    return dict(san=san,start=start,end=start+len(san),quote=san)


def claim(text,**overrides):
    row=dict(id='c',context_id='root',kind='occupancy',start=0,end=len(text),quote=text,
             color='White',piece='rook',square='a1',count=None,value=True,san=None)
    row.update(overrides)
    return row


def extraction(claims=None,contexts=None):
    return dict(contexts=contexts or [context()],claims=claims or [],uncertainties=[])


def test_exact_root_piece_claim_and_factual_inventory():
    text="White's rook is on a1."
    result=evidence_for(chess.STARTING_FEN,text,extraction([claim(text)]))
    assert result['claims'][0]['status']=='supported'
    assert result['root_board']['side_to_move']=='White'
    assert result['root_board']['inventory']['White']['pawn']==8
    assert 'e4' in result['root_board']['legal_san']
    assert result['claims'][0]['interpretation_provisional'] is True
    assert 'score' not in result


@pytest.mark.parametrize('change',[dict(quote='invented'),dict(start=True),dict(end=999),dict(unknown=1),dict(value=1)])
def test_quote_type_span_and_extra_field_rejected(change):
    text="White's rook is on a1."
    row=claim(text); row.update(change)
    with pytest.raises(ValueError): validate_extraction(text,extraction([row]))


def test_top_level_extra_and_duplicate_ids_rejected():
    text='a1'; rows=extraction([claim(text)])
    with pytest.raises(ValueError): validate_extraction(text,{**rows,'grade':2})
    with pytest.raises(ValueError): validate_extraction(text,extraction([claim(text),claim(text)]))


def test_moves_must_be_exact_verbatim_tokens():
    text='e4 e5'
    bad=quoted_move(text,'e4'); bad['san']='Nf3'
    with pytest.raises(ValueError): validate_extraction(text,extraction(contexts=[context(),context('line','root',[bad])]))
    with pytest.raises(ValueError): validate_extraction(text,extraction([claim(text,kind='legality',san='Nf3')]))


def test_parent_and_branch_moves_are_sequential_not_root_checked():
    text='e4 e5 Nf3'
    contexts=[context(),context('prefix','root',[quoted_move(text,'e4'),quoted_move(text,'e5')]),
              context('child','prefix',[quoted_move(text,'Nf3')])]
    row=claim(text,context_id='child',piece='knight',square='f3')
    result=evidence_for(chess.STARTING_FEN,text,extraction([row],contexts))
    assert result['claims'][0]['status']=='supported'
    assert result['claims'][0]['context_provisional'] is True
    assert all(item['status']=='supported' for item in result['contexts'])


def test_variation_siblings_keep_independent_boards():
    text='e4 e5 or c5'
    contexts=[context(),context('prefix','root',[quoted_move(text,'e4')]),
              context('kingpawn','prefix',[quoted_move(text,'e5')]),context('sicilian','prefix',[quoted_move(text,'c5')])]
    rows=[claim(text,id='a',context_id='kingpawn',color='Black',piece='pawn',square='e5'),
          claim(text,id='b',context_id='sicilian',color='Black',piece='pawn',square='c5')]
    result=evidence_for(chess.STARTING_FEN,text,extraction(rows,contexts))
    assert [row['status'] for row in result['claims']]==['supported','supported']


def test_illegal_prefix_never_falls_back_to_initial_board():
    text='e5'
    contexts=[context(),context('bad','root',[quoted_move(text,'e5')])]
    row=claim(text,context_id='bad')
    result=evidence_for(chess.STARTING_FEN,text,extraction([row],contexts))
    assert result['claims'][0]['status']=='unverified'
    assert result['contexts'][1]['status']=='unverified'


@pytest.mark.parametrize('san',['Nf3+','Nf3#'])
def test_incorrect_check_suffix_is_not_silently_accepted(san):
    row=claim(san,kind='legality',san=san,value=True)
    result=evidence_for(chess.STARTING_FEN,san,extraction([row]))
    assert result['claims'][0]['status']=='contradicted'
    assert result['claims'][0]['actual']['legal'] is True
    assert result['claims'][0]['actual']['annotation_status']=='contradicted'


def test_wrong_suffix_in_prefix_abstains_descendant():
    text='e4+ e5'
    contexts=[context(),context('wrong','root',[quoted_move(text,'e4+')]),context('child','wrong',[quoted_move(text,'e5')])]
    result=evidence_for(chess.STARTING_FEN,text,extraction([claim(text,context_id='child')],contexts))
    assert result['claims'][0]['status']=='unverified'
    assert result['contexts'][1]['move_checks'][0]['annotation_status']=='contradicted'


@pytest.mark.parametrize('which',['unknown','cycle'])
def test_unknown_or_cyclic_context_is_unverified(which):
    text='rook'
    contexts=[context(),context('a','b'),context('b','a')] if which=='cycle' else [context()]
    row=claim(text,context_id='a' if which=='cycle' else 'missing')
    result=evidence_for(chess.STARTING_FEN,text,extraction([row],contexts))
    assert result['claims'][0]['status']=='unverified'


def test_material_and_strategy_are_distinct():
    text='White has eight pawns and will win.'
    rows=[claim(text,id='inventory',kind='material',piece='pawn',square=None,count=8,value=None),
          claim(text,id='opinion',kind='strategic',color=None,piece=None,square=None,value=None)]
    result=evidence_for(chess.STARTING_FEN,text,extraction(rows))
    assert [row['status'] for row in result['claims']]==['supported','unverified']
    assert any('repetition' in note for note in result['limitations'])


def test_schema_is_strict_at_each_object():
    assert EXTRACTION_SCHEMA['additionalProperties'] is False
    for name in ('contexts','claims'):
        item=EXTRACTION_SCHEMA['properties'][name]['items']
        assert item['additionalProperties'] is False
        assert set(item['required'])==set(item['properties'])


@pytest.mark.parametrize('san',['--','0000'])
def test_parser_null_moves_are_not_legal_chess_moves(san):
    row=claim(san,kind='legality',san=san,value=True)
    result=evidence_for(chess.STARTING_FEN,san,extraction([row]))
    assert result['claims'][0]['status']=='contradicted'
    assert result['claims'][0]['actual']['legal'] is False
