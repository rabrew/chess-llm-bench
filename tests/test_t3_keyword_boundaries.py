"""Lexical theme credit must not come from embedded character sequences."""
import pytest
from src.evaluator import score_t3
from src.metrics import rescore_t3_theme


@pytest.mark.parametrize('theme,text',[
    ('mate','Black has a material advantage.'),('pin','Pieces are spinning around.'),
    ('fork','A forklift carries the pieces.'),('equality','There are seven pieces.'),
    ('mate','émate mateé mate_idea premate mate2'),('novelTheme','prenovel themeish'),
])
def test_embedded_substrings_receive_no_theme_credit(theme,text):
    assert rescore_t3_theme(theme,text)==0
    assert score_t3(text,'White',100,theme)['t3_p2_theme_correct']==0


@pytest.mark.parametrize('theme,text',[
    ('mate','White delivers CHECKMATE!'),('pin','The knight is pinned.'),
    ('pin','Pinning the defender wins.'),('fork','A double attack wins.'),
    ('advancedPawn','The passed pawn will promote.'),('novelTheme','A novel theme appears.'),
    ('novel_theme','A novel theme appears.'),('mate','♟mate!'),('mate','No mate exists.'),
])
def test_aliases_normalization_and_literal_mentions_remain(theme,text):
    assert rescore_t3_theme(theme,text)==1
    assert score_t3(text,'White',100,theme)['t3_p2_theme_correct']==1


@pytest.mark.parametrize('theme,text',[(None,'mate'),('mate',None),(1,'mate'),('mate',1)])
def test_rescore_missing_invalid_inputs_remain_null(theme,text):
    assert rescore_t3_theme(theme,text) is None


def test_shared_helper_has_explicit_lexical_version():
    from src.evaluator import match_t3_theme, T3_THEME_MATCHER_VERSION
    assert T3_THEME_MATCHER_VERSION=='keyword_boundaries_v1'
    assert match_t3_theme('mate','material')==0
    assert score_t3('An equal position.','Equal',0,None)['t3_score']==1
