# Chess LLM Benchmark — Simple Summary

**Ryan Brew · draft of 2026-10-29**

---

## The question

**Do language models actually reason about a chess position, or do they just produce text that looks
like chess?**

Language models learn from huge amounts of internet text, which includes a lot of chess: game records,
articles and puzzle solutions. Ask one a chess question and you get something that *sounds* right.
Chess is a good way to check whether there is real reasoning underneath, because:

- every position has an objective answer that a chess engine can compute
- a move is either legal or illegal, with no grey area
- you have to keep track of where every piece is

## The experiment

- **22 open-source models**, from 2 billion to 70 billion parameters, all run on my own PC.
- **4,000 positions** from Lichess puzzles, 1,000 in each of four difficulty tiers.
- **Three questions per position:**
  1. Who is winning, and by how much?
  2. What is the best move?
  3. Why is that side better?
- **Six ways of asking.** One of them turned out to be an exact repeat of another (see below).
- **526,662 answers** in total, all checked against **Lc0**, a neural-network chess engine far
  stronger than any human.

## What I found

The numbers come straight from the results files (`results/metrics/summary.json`):

<!-- table:headline -->
| Measure | Value |
|---|---|
| Records / models / positions | 526,662 / 22 / 4,000 |
| T1: Spearman ρ, model eval vs engine (pooled) | 0.11 |
| T1: eval error ÷ error of always answering 0 | 3.04 |
| T2: legal move, final (after re-asking with the legal-move list) | 97.0% |
| T2: legal move, model's own first answer | 9.3% (5 re-run models, combined prompts) |
| T2: blunder rate (clamped CPL ≥ 1000), 95% CI | 41.9% (41.7–42.0%) |
| T2: plays the engine's best move | 10.2% vs 8.8% for a random legal move (1.16×) |
| T2: plays the human setup move (the move that led into the puzzle) | 11.5% |
| T3: names the better side | 30.3% vs 54.2% for always naming the usual side (κ = 0.07) |
| T3: names a tactical theme of the puzzle | 6.0% |
<!-- /table:headline -->

### 1. On their own, the models almost never give a legal move

When a model has to give an evaluation, a move and an explanation in one answer, its move is legal
only about **1 time in 10**. The models mostly write a common opening move like "e4" or "Nf6",
whatever the position. The final 97% legality only appears because the system asks again and hands
the model a list of every legal move to choose from.

### 2. Their moves are barely better than picking at random

- **The engine's best move:** models play it about 10% of the time. A random legal move would hit it
  about 9% of the time.
- **The human's move:** models more often play the move a human actually played in that position,
  usually the mistake that set up the puzzle.

That looks like remembering familiar moves, not working out good ones.

### 3. Moves are either perfect or disasters

About 19% of moves are essentially perfect, and about 42% throw away the position. Very little falls
in between.

### 4. They can't judge who is winning

- **The number they give** for "who is winning" barely follows the real position.
- **A fixed guess does better:** "always say it's level" beats every model's estimate.
- **Naming the better side:** models get it right less often than "always name the side that is
  usually better" does.

### 5. Bigger models are mostly not better

From 2B to 70B the picture is the same for Llama, Qwen and DeepSeek-R1. The exception is Google's
Gemma models, which do improve with size. That points to how a model was trained, not how big it is.

### 6. Asking them to "think step by step" doesn't help

The chain-of-thought prompt gives the same results as the plain one. And when I asked the *same*
question twice, a model gave the same move only about 69% of the time. Model answers are not
repeatable.

## Can teaching the board fix it?

**Experiment 1: done.**
- I fine-tuned five of the models to list exactly where every piece is.
- **What they learned:** board-reading went from almost nothing to about 90–99% correct.
- **What didn't change:** nothing else got better. Their moves got slightly *worse*, and they gave
  *more* illegal moves, even with the list of legal moves in front of them.
- **What it means:** knowing where the pieces are didn't help them play.

**Experiment 2: done.** The same five models were taught the piece positions *and every legal move*.
- I wrote down beforehand what would count as success. The first test was whether they actually
  learned the legal moves: at least 90% right.
- They didn't. They learned the pieces, but their lists of legal moves were only about 52–81% right.
- By the rule I set in advance, that means no conclusion about whether rules knowledge helps.
- Still, their answers in the benchmark became *less* legal, not more, just as in Experiment 1.

## The conclusion

These models have learned what chess **looks like** (the notation, the vocabulary, the usual moves),
not how it **works**. On every measure the evidence is more consistent with pattern matching than
with reasoning. Being bigger mostly doesn't change that, and neither did teaching them where the
pieces are.

## Honest limits

- **Scope:** only open-source models up to 70B, and only chess. This says nothing about commercial
  models or about reasoning in general.
- **Possibly seen before:** the puzzles are public, so models may have seen them in training. That
  makes the failures more striking, not less.
- **Setup position, not the puzzle:** the benchmark showed each puzzle's *starting* position, one move
  before the puzzle begins. That affects the "name the tactic" mark and the difficulty tiers, but not
  the move or evaluation scores.
- **Not repeatable answer by answer:** the dataset and scoring can be reproduced exactly, but the
  models' answers vary from run to run.
- **The second experiment didn't finish the job:** the models only half-learned the legal moves,
  so whether knowing the rules would help is still an open question.
- **Earlier numbers were wrong:** an audit found and fixed several measurement mistakes. Some numbers
  in older drafts were wrong, and they are listed in `docs/FINDINGS.md`.

---

*Full details, data and code: `docs/FINDINGS.md` and github.com/rabrew/chess-llm-bench*
