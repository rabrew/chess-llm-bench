# Chess LLM Benchmark — Simple Summary

**Ryan Brew · May 2026**

---

## The Question

**Can today's open-source language models actually play chess, or do they just sound like they can?**

Large language models are trained on enormous amounts of internet text, which includes a lot of chess: game records, articles, books, Wikipedia. So when you ask one a chess question, it produces output that *looks* like a chess answer. But is the model really thinking about the board, or is it just pattern-matching the words?

Chess is a perfect testbed for this question because:
- It has a single objective truth (Stockfish, the world's strongest engine, can tell you the right answer for any position).
- The rules are completely formal — a move is either legal or illegal, no grey area.
- It requires spatial reasoning that isn't reducible to language patterns.

---

## The Experiment

I tested **22 open-source language models**, ranging from 2 billion parameters (small) to 70 billion (large), on **4,000 chess positions** from the Lichess puzzle database — 1,000 each at four difficulty tiers (easy, medium, hard, extreme).

For every position, each model was asked three questions:

1. **T1 — Evaluation:** "Who's winning, and by how much?" (in centipawns, where 100 = one pawn)
2. **T2 — Move:** "What's the best move here?"
3. **T3 — Explanation:** "Why is this side better?"

I scored each answer against the strongest engines available — **Stockfish 17 at depth 22** for evaluation truth, and **Lc0 (a neural-network engine) at 800 nodes** for move quality. Both are far stronger than any human grandmaster.

Each position was tested with 6 different prompt formats (to make sure the result wasn't just an artefact of how the question was phrased), giving a total of **526,662 evaluations** across 22 models × 4,000 positions × 6 prompt formats.

The benchmark was run locally on consumer hardware (no API costs, no commercial models).

---

## The Results

### 1. When asked for a move, models produce a legal one ~98% of the time.

When the prompt explicitly asks for a move, models produce a valid chess move on average **97.9% of the time**. The largest models do best (Qwen2.5-32B at 99.94%, Llama3.3-70B at 99.67%); the worst is the DeepSeek-R1 reasoning model at 90% (its long chain-of-thought output makes the move harder to extract).

So **legality is mostly NOT the problem.** Models can do chess notation. The problem is move *quality*.

### 2. The legal moves they play are wildly inconsistent.

The legal moves models produce do not look like "weak but consistent" amateur play. The distribution is **bimodal**:

- **About 18% of the time**, the model plays essentially the best move (top-3 engine choice)
- **About 10% of the time**, the model plays *exactly* the engine's #1 move — ~3× better than random
- **About 39% of the time**, the model plays a catastrophic blunder (equivalent to hanging a queen or walking into mate)
- Mean centipawn loss is ~700, but that's the average of "occasionally brilliant" and "frequently catastrophic"

Among non-catastrophic moves, mean centipawn loss is **449** — within the range of a weak amateur. The best architecture (Gemma 4) finds the engine's #1 move **14% of the time**, ~4× random chance.

The headline issue isn't that models play uniformly badly — it's that they're **unreliable**. They flip between recognising the right move and walking into disaster, with no apparent middle ground.

### 3. Models can tell when one side is *clearly* winning, but struggle on close positions.

This depends heavily on what you mean by "winning":

| Threshold | All-model accuracy | Random baseline |
|---|---|---|
| Exact agreement on direction | 44.5% | 33% |
| Borderline: ±50 cp from neutral | 46.1% | 33% |
| One-pawn advantage required | **61.6%** | 33% |
| Two-pawn (clearly decisive) advantage | **78.9%** | 33% |

So when one side has a *clear* (2-pawn) advantage, models agree with the engine on who's winning **79% of the time** — much better than the headline number of 46% (which used a poor centipawn threshold). On positions that are *near-equal* (under 1 pawn difference), models fail roughly half the time — which is what you'd expect if the model isn't actually evaluating the position, just guessing.

Strikingly, **the largest model in the study (Llama3.3-70B) is wrong about who's winning even on clearly decisive positions** — at the 1-pawn threshold it agrees with the engine only 22% of the time, well *below* the 33% random baseline. It's somehow systematically *anti-correlated* with truth.

### 4. Bigger models are not better.

There's no consistent relationship between model size and chess ability:
- A 7B-parameter model (Mistral) **outplays** a 70B-parameter model (LLaMA 3.3) on move quality.
- The largest models (32B, 31B, 70B) are at the **bottom** of the rankings for understanding who's winning.
- The best architecture in the study (Google's Gemma 4) tops the rankings at 26B and 31B, but its 2B variant is the *worst* model on move quality — proving it's the architecture, not the size, that matters.

### 5. Reasoning ≠ Playing.

The most interesting architecture-level finding: **the models that play best are NOT the models that reason best about chess.**

- **Gemma 4 31B** picks the best move 14% of the time (highest in the study) — but is below-random at saying who's winning.
- **DeepSeek-R1** (a chain-of-thought reasoning model) is best at saying who's winning (77% direction accuracy at the 1-pawn threshold) — but worst at producing legal moves and only middling at move quality.

The two failure modes — "can't pick a good move" and "can't say who's winning" — are addressed by different architectural choices, suggesting they are mechanistically distinct kinds of failures.

### 6. Talking about chess is easier than playing it.

Some models can produce surprisingly good positional explanations — describing the right concepts, naming the right tactics — while still playing terribly. They've absorbed chess **vocabulary** from training text without acquiring the underlying spatial reasoning those words describe.

---

## The Conclusion

**Today's open-source language models, up to 70 billion parameters, do not play chess at a competent level — even though they can produce legal moves and recognise clearly-decisive positions.**

They have learned the *surface form* of chess — the notation, the vocabulary, the way chess writing flows — and at the largest sizes they have learned it almost perfectly (99% legal-move rates). But the deeper task of consistently picking *good* moves is not learned at any scale.

The fact that this failure is **uniform across two orders of magnitude of model size** is the key finding. It strongly suggests that scaling general-purpose language models on internet text is not, by itself, a path to chess competence. Whatever cognitive structure is needed for consistent spatial reasoning over a chess board is not emerging just by making the model bigger.

The bimodal failure pattern — "occasionally brilliant, frequently catastrophic" — is consistent with the hypothesis that models are pattern-matching on familiar position shapes rather than maintaining a working board representation. When the position resembles something in training data, performance is decent. When it doesn't, the model walks into a forced mate.

This is a small, controlled example of a much bigger question: *which kinds of intelligence can be acquired by next-token prediction on text, and which cannot?* Chess says: surface fluency is acquired easily; consistent grounded reasoning is not.

---

## What this study does NOT show

- **It doesn't show LLMs can't reason at all.** It shows that *these* models, on *this* task, fail.
- **It doesn't include commercial models** (Claude, GPT-4o). Those may perform differently — that's the natural next step.
- **It doesn't test chess-fine-tuned models.** A model trained specifically on chess data would almost certainly do better; the question this study answers is about general-purpose pre-training.
- **It uses Lichess puzzles**, which are tactical by construction. Performance on quiet middlegame or endgame positions may differ.

---

## Bottom line

If you ever wondered whether ChatGPT and friends are *really* thinking or *just predicting words* — chess is a clean, specific, testable instance where you can watch the difference. They CAN produce legal chess moves. They CANNOT consistently choose good ones. They CAN tell when one side is overwhelmingly winning. They CANNOT distinguish a slight advantage from a slight disadvantage. They CAN talk about chess concepts. They CANNOT use those concepts to play.

22 models. 70 billion parameters at the top end. 4,000 positions, half a million evaluations. None of them play chess like a human who actually understands the game.

---

*Full data, code, and methodology: github.com/rabrew/chess-llm-bench*
