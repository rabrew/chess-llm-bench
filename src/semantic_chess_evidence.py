"""Exact source-bound extraction and deterministic chess evidence, never grades."""
import re

import chess

EVIDENCE_VERSION = 'source_bound_chess_evidence_v1'
PIECES = ('king', 'queen', 'rook', 'bishop', 'knight', 'pawn')
KINDS = ('occupancy', 'material', 'legality', 'check', 'checkmate', 'strategic')
MAX_CONTEXTS = 16
MAX_CLAIMS = 48
MAX_CONTEXT_MOVES = 16
MAX_TEXT_FIELD = 2048
MAX_ID_LENGTH = 64
MAX_PIECE_COUNT = 32


def _object(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


MOVE_SCHEMA = _object(dict(san={'type':'string'}, start={'type':'integer'},
                           end={'type':'integer'}, quote={'type':'string'}))
CONTEXT_SCHEMA = _object(dict(id={'type':'string'}, parent_id={'type':['string','null']},
    moves={'type':'array','items':MOVE_SCHEMA}, assumption={'type':'string'}))
CLAIM_SCHEMA = _object(dict(id={'type':'string'}, context_id={'type':'string'},
    kind={'type':'string','enum':list(KINDS)}, start={'type':'integer'}, end={'type':'integer'},
    quote={'type':'string'}, color={'type':['string','null'],'enum':['White','Black',None]},
    piece={'type':['string','null'],'enum':[*PIECES,None]}, square={'type':['string','null']},
    count={'type':['integer','null']}, value={'type':['boolean','null']}, san={'type':['string','null']}))
EXTRACTION_SCHEMA = _object(dict(contexts={'type':'array','items':CONTEXT_SCHEMA},
    claims={'type':'array','items':CLAIM_SCHEMA}, uncertainties={'type':'array','items':{'type':'string'}}))


def _exact_object(value, schema):
    if not isinstance(value,dict) or set(value)!=set(schema['properties']):
        raise ValueError('Object must contain exactly the required extraction fields')


def _string(value, maximum=MAX_TEXT_FIELD):
    if not isinstance(value,str) or not value or len(value)>maximum:
        raise ValueError('Expected a nonempty bounded string')


def _span(text, item):
    start,end=item['start'],item['end']
    if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text):
        raise ValueError('Invalid source span')
    _string(item['quote'])
    if text[start:end]!=item['quote']:
        raise ValueError('Extraction quote does not match its exact source span')


def _san_bound(text, san, start, end):
    _string(san,MAX_ID_LENGTH)
    if text[start:end]!=san or (start and re.match(r'\w',text[start-1])):
        raise ValueError('SAN must be an exact source token')
    if end<len(text) and re.match(r'[\w+#]',text[end]):
        raise ValueError('SAN extraction cannot omit a word or check suffix')


def _validate_context(text, item):
    _exact_object(item,CONTEXT_SCHEMA); _string(item['id'],MAX_ID_LENGTH)
    if item['parent_id'] is not None: _string(item['parent_id'],MAX_ID_LENGTH)
    if not isinstance(item['assumption'],str) or len(item['assumption'])>MAX_TEXT_FIELD:
        raise ValueError('Invalid context assumption')
    if not isinstance(item['moves'],list) or len(item['moves'])>MAX_CONTEXT_MOVES:
        raise ValueError('At most16 source-bound moves per context')
    previous=-1
    for move in item['moves']:
        _exact_object(move,MOVE_SCHEMA); _span(text,move)
        _san_bound(text,move['san'],move['start'],move['end'])
        if move['quote']!=move['san'] or move['start']<previous:
            raise ValueError('Context moves must be verbatim and ordered in their source')
        previous=move['end']


def _kind_attributes(item):
    required,unused={
        'occupancy':(('piece','square','value'),('count','san')),
        'material':(('color','piece','count'),('value','square','san')),
        'legality':(('san','value'),('color','piece','square','count')),
        'check':(('value',),('piece','square','count','san')),
        'checkmate':(('value',),('piece','square','count','san')),
        'strategic':((),('color','piece','square','count','value','san')),
    }[item['kind']]
    if any(item[key] is None for key in required) or any(item[key] is not None for key in unused):
        raise ValueError('Claim kind has missing or inapplicable typed attributes')


def _validate_claim(text, item):
    _exact_object(item,CLAIM_SCHEMA); _span(text,item)
    for field in ('id','context_id'): _string(item[field],MAX_ID_LENGTH)
    if item['kind'] not in KINDS or item['color'] not in ('White','Black',None) or item['piece'] not in (*PIECES,None):
        raise ValueError('Unknown claim type/color/piece')
    _kind_attributes(item)
    if item['square'] is not None and (not isinstance(item['square'],str) or not re.fullmatch('[a-h][1-8]',item['square'])):
        raise ValueError('Invalid square')
    if item['count'] is not None and (type(item['count']) is not int or not 0<=item['count']<=MAX_PIECE_COUNT):
        raise ValueError('Invalid extracted material count')
    if item['value'] is not None and type(item['value']) is not bool:
        raise ValueError('Claim polarity must be Boolean or null')
    if item['san'] is not None:
        _string(item['san'],MAX_ID_LENGTH)
        matches=list(re.finditer(r'(?<!\w)'+re.escape(item['san'])+r'(?![\w+#])',item['quote']))
        if not matches:
            raise ValueError('Claim SAN is absent from its bound source quote')


def validate_extraction(text, extraction):
    """Reject unknown fields, unbound quotes/moves, duplicate IDs and wrong types."""
    if not isinstance(text,str): raise ValueError('Explanation must be text')
    _exact_object(extraction,EXTRACTION_SCHEMA)
    contexts,claims=extraction['contexts'],extraction['claims']
    if not isinstance(contexts,list) or not 1<=len(contexts)<=MAX_CONTEXTS or not isinstance(claims,list) or len(claims)>MAX_CLAIMS:
        raise ValueError('Invalid bounded context/claim arrays')
    for item in contexts: _validate_context(text,item)
    for item in claims: _validate_claim(text,item)
    for items in (contexts,claims):
        if len({item['id'] for item in items})!=len(items): raise ValueError('Duplicate extraction IDs')
    roots=[item for item in contexts if item['id']=='root']
    if len(roots)!=1 or roots[0]['parent_id'] is not None or roots[0]['moves']:
        raise ValueError('Exactly one unmoved root context is required')
    if any(item['id']!='root' and item['parent_id'] is None for item in contexts):
        raise ValueError('Only root may have no parent')
    uncertainties=extraction['uncertainties']
    if not isinstance(uncertainties,list) or len(uncertainties)>MAX_CLAIMS or any(not isinstance(item,str) or len(item)>MAX_TEXT_FIELD for item in uncertainties):
        raise ValueError('Invalid extraction uncertainties')
    return extraction


def _move_result(board,san):
    try:
        move=board.parse_san(san)
        if move not in board.legal_moves:
            raise ValueError('Parser null moves are not legal chess moves')
    except (ValueError,chess.IllegalMoveError,chess.AmbiguousMoveError):
        return None,dict(san=san,legal=False,status='contradicted',annotation_status='unverified',reason='Illegal or ambiguous SAN in this context')
    after=board.copy(); after.push(move)
    annotation_ok=(not san.endswith('#') or after.is_checkmate()) and (not san.endswith('+') or after.is_check())
    return after,dict(san=san,legal=True,status='supported' if annotation_ok else 'contradicted',
                      annotation_status='supported' if annotation_ok else 'contradicted',
                      gives_check=after.is_check(),gives_checkmate=after.is_checkmate())


class _Contexts:
    def __init__(self,board,contexts):
        self.root=board; self.items={item['id']:item for item in contexts}; self.cache={}; self.visiting=set()

    def resolve(self,identity):
        if identity in self.cache: return self.cache[identity]
        if identity not in self.items or identity in self.visiting:
            return None,dict(id=identity,status='unverified',reason='Unknown or cyclic context',move_checks=[]),0
        item=self.items[identity]
        if identity=='root':
            result=(self.root.copy(),dict(id='root',status='supported',reason='Exact initial FEN',move_checks=[]),0)
        else:
            self.visiting.add(identity)
            parent,_,last_end=self.resolve(item['parent_id'])
            result=self._advance(item,parent,last_end)
            self.visiting.remove(identity)
        self.cache[identity]=result
        return result

    def _advance(self,item,board,last_end):
        checks=[]; reason='Sequential quoted moves under a provisional extracted branch'
        for move in item['moves']:
            if board is None or move['start']<last_end:
                board=None; reason='Unverified parent or source-order context'; break
            board,check=_move_result(board,move['san']); checks.append(check); last_end=move['end']
            if check['status']!='supported':
                board=None; reason='Illegal move or incorrect check/mate suffix in prefix'; break
        return board,dict(id=item['id'],status='supported' if board is not None else 'unverified',
                          reason=reason,assumption=item['assumption'],context_provisional=True,move_checks=checks),last_end


def _piece_info(piece):
    return None if piece is None else dict(color='White' if piece.color else 'Black',piece=chess.piece_name(piece.piece_type))


def _claim_actual(board,claim):
    kind=claim['kind']; color=claim['color']; piece=claim['piece']
    if kind=='occupancy' and claim['square'] and piece and claim['value'] is not None:
        actual=_piece_info(board.piece_at(chess.parse_square(claim['square'])))
        present=actual is not None and actual['piece']==piece and (color is None or actual['color']==color)
        return present==claim['value'],dict(occupant=actual,square=claim['square'],asserted_present=claim['value'])
    if kind=='material' and color and piece and claim['count'] is not None:
        count=len(board.pieces(chess.PIECE_NAMES.index(piece),color=='White'))
        return count==claim['count'],dict(color=color,piece=piece,count=count,asserted_count=claim['count'])
    if kind=='legality' and claim['san'] and claim['value'] is not None:
        _,actual=_move_result(board,claim['san'])
        return actual['legal']==claim['value'] and actual['annotation_status']!='contradicted',actual
    if kind in ('check','checkmate') and claim['value'] is not None:
        if color is not None and (color=='White')!=board.turn:
            return None,dict(reason='Check/checkmate evidence only verifies the side to move')
        target=board.turn if color is None else color=='White'
        if kind=='check':
            square=board.king(target); actual=bool(square is not None and board.is_attacked_by(not target,square))
        else: actual=target==board.turn and board.is_checkmate()
        return actual==claim['value'],dict(color='White' if target else 'Black',value=actual,asserted_value=claim['value'])
    return None,dict(reason='Strategic, underspecified or unsupported interpretation; no deterministic truth established')


def _root_info(board):
    inventory={color:{piece:len(board.pieces(chess.PIECE_NAMES.index(piece),color=='White')) for piece in PIECES} for color in ('White','Black')}
    return dict(side_to_move='White' if board.turn else 'Black',in_check=board.is_check(),
                checkmate=board.is_checkmate(),inventory=inventory,legal_san=sorted(board.san(move) for move in board.legal_moves))


def evidence_for(fen,text,extraction):
    """Expose exact chess facts plus provisional claim interpretation, never grades."""
    validate_extraction(text,extraction)
    board=chess.Board(fen)
    if not board.is_valid(): raise ValueError('Invalid initial chess position')
    contexts=_Contexts(board,extraction['contexts']); context_rows=[]
    for item in extraction['contexts']: context_rows.append(contexts.resolve(item['id'])[1])
    claims=[]
    for claim in extraction['claims']:
        state,_,_=contexts.resolve(claim['context_id'])
        matched,actual=_claim_actual(state,claim) if state is not None else (None,dict(reason='Unverified context; initial board was not substituted'))
        claims.append(dict(id=claim['id'],context_id=claim['context_id'],kind=claim['kind'],
            literal=claim['quote'],span=[claim['start'],claim['end']],
            status='unverified' if matched is None else 'supported' if matched else 'contradicted',actual=actual,
            context_provisional=claim['context_id']!='root',interpretation_provisional=True,
            assumption='Claim attributes, polarity and branch association are extracted interpretations, not verified prose semantics'))
    return dict(version=EVIDENCE_VERSION,root_board=_root_info(board),contexts=context_rows,claims=claims,
                uncertainties=extraction['uncertainties'],limitations=[
                    'FEN does not establish repetition history, forced draw, winning play or an engine verdict.',
                    'Quoted token binding does not prove that extracted context or polarity represents the prose correctly.',
                    'These checks are fact evidence, not whole-explanation grounding or support grades.'])
