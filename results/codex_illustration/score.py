"""Score the GPT replies like the worker, and compare with stored answers on the same positions."""
import json
import sys

sys.path.insert(0, ".")
from src import claude_batch as cb
from src.evaluator import score_all
from src.metrics import _is_legal_san
from src.llm_client import parse_response

items = json.load(open(sys.argv[1]))
needs = []
for it in items:
    job = {"fen": it["fen"]}
    parsed, retry = cb.first_answer(job, it["reply"])
    it["first_legal"] = _is_legal_san(it["fen"], parse_response(it["reply"]).get("move"))
    it["needs_retry"] = retry
    if retry:
        needs.append(it["position_id"])
    else:
        s = score_all(parsed_response=parsed, position={"id": it["position_id"], "fen": it["fen"],
                      "stockfish_eval": it["stockfish_eval"], "stockfish_best_move": "", "theme": ""})
        it["scores"] = {k: s.get(k) for k in ("t1_model_eval", "t1_stockfish_eval", "t2_move", "t2_legal",
                                               "t3_side_claimed", "t3_p1_side_correct")}
print(json.dumps({"needs_retry": needs, "items": [{k: v for k, v in it.items() if k not in ("prompt", "retry_prompt", "reply")} for it in items]}))
