"""GPT (via Codex, 12 positions) next to Claude and local models on the same positions, FEN prompt."""
import json
import sys

import pandas as pd

sys.path.insert(0, ".")
from src import claude_batch as cb
from src.evaluator import score_all
from src.llm_client import parse_response
from src.metrics import _is_legal_san, _spearman

items = json.load(open(sys.argv[1]))
ids = {it["position_id"] for it in items}
gpt = []
for it in items:
    parsed, needs = cb.first_answer({"fen": it["fen"]}, it["reply"])
    if needs:
        parsed = cb.apply_retry(parsed, it["fen"], it["retry_reply"])
    s = score_all(parsed_response=parsed, position={"id": it["position_id"], "fen": it["fen"],
                  "stockfish_eval": it["stockfish_eval"], "stockfish_best_move": "", "theme": ""})
    gpt.append({"model": "gpt-6.1-sol (Codex)", "position_id": it["position_id"], "fen": it["fen"],
                "raw_response": it["reply"], "t1_model_eval": s["t1_model_eval"], "t1_stockfish_eval": s["t1_stockfish_eval"],
                "t2_legal": s["t2_legal"], "t3_p1_side_correct": s["t3_p1_side_correct"]})


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["position_id"] in ids and r["prompt_format"] == "fen_only"]


rows = gpt + load("results/claude/evaluations.jsonl") + load("/home/rabrew/Desktop/clb-analysis/results/plain_board/evaluations.jsonl")
df = pd.DataFrame(rows)
df = df[~df["model"].str.startswith("claude-sonnet")]
out = []
for model, g in df.groupby("model"):
    if len(g) < len(ids):
        continue
    first = sum(_is_legal_san(r.fen, parse_response(r.raw_response or "").get("move")) for r in g.itertuples())
    out.append({"model": model, "first_legal": f"{first}/12", "final_legal": f"{int(g.t2_legal.sum())}/12",
                "side_correct": f"{int(g.t3_p1_side_correct.fillna(0).sum())}/12",
                "eval_rho": round(_spearman(g.dropna(subset=["t1_model_eval", "t1_stockfish_eval"])), 2)})
print(pd.DataFrame(out).sort_values(["eval_rho"], ascending=False).to_string(index=False))
