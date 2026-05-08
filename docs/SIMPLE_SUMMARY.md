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

I scored each answer against **Stockfish 17 at depth 22** — the gold-standard chess engine, far stronger than any human grandmaster.

Each position was tested with 6 different prompt formats (to make sure the result wasn't just an artefact of how the question was phrased), giving a total of **526,662 evaluations** across 22 models × 4,000 positions × 6 prompt formats.

The benchmark was run locally on consumer hardware (no API costs, no commercial models).

---

## The Results

### 1. About 35% of moves the models suggest are **illegal**.

Not "bad" — *illegal*. Like trying to move a bishop in a straight line, or moving into check. This rate is the same across all 22 models, regardless of size. The 70-billion-parameter model is no better than the 2-billion-parameter model at producing legal chess moves.

When models that produced illegal moves were given the **complete list of legal moves** and asked to pick one, **a third of them still failed to choose a valid option**. So the failure isn't about formatting — they don't know what moves are legal in the first place.

### 2. When they do play legally, they are wildly inconsistent.

The legal moves models produce do not look like "weak but consistent" amateur play. The distribution is **bimodal**:

- **About 18% of the time**, the model plays essentially the best move (CPL < 25 — top-3 Stockfish choice)
- **About 10% of the time**, the model plays *exactly* Stockfish's #1 move — ~3× better than random
- **About 39% of the time**, the model plays a catastrophic blunder (CPL ≥ 1000 — equivalent to hanging a queen or walking into mate)
- The mean is ~700 CPL, but that's the average of "occasionally brilliant" and "frequently catastrophic"

Among non-catastrophic moves, mean CPL is **449** — within the range of a weak amateur (200–400 CPL is beginner level). The best architecture (Gemma 4) finds Stockfish's #1 move **14% of the time**, ~4× random chance.

The headline issue isn't that models play uniformly badly — it's that they're **unreliable**. They flip between recognising the right move and walking into disaster, with no apparent middle ground.

### 3. They can't tell who's winning.

When asked "is White or Black better?", models get the answer right **46% of the time**. Random guessing would be 33%. So they're barely above chance — and the **largest model in the study (70B) is the *worst* at this**, scoring 34%.

### 4. Bigger models are not better.

This is the surprising finding. There's no consistent relationship between model size and chess ability:
- A 7B-parameter model (Mistral) **outplays** a 70B-parameter model (LLaMA 3.3) on move quality.
- The largest models (32B, 31B, 70B) are at the **bottom** of the rankings for understanding who's winning.
- The best architecture in the study (Google's Gemma 4) tops the rankings at 26B and 31B, but its 2B variant is the *worst* model overall — proving it's the architecture, not the size, that matters.

### 5. Talking about chess is easier than playing it.

Some models can produce surprisingly good positional explanations — describing the right concepts, naming the right tactics — while still playing terribly. They've absorbed chess **vocabulary** from training text without acquiring the underlying spatial reasoning those words describe.

---

## The Conclusion

**Today's open-source language models, up to 70 billion parameters, do not understand chess.**

They have learned the *surface form* of chess — the notation, the vocabulary, the way chess writing flows — but they have not learned the *substance* of chess. They produce text that looks like a chess move without first verifying it against any internal model of the board.

The fact that this failure is **uniform across two orders of magnitude of model size** is the key finding. It strongly suggests that scaling general-purpose language models on internet text is not, by itself, a path to chess competence. Whatever cognitive structure is needed for spatial reasoning over a chess board is not emerging just by making the model bigger.

This is a small, controlled example of a much bigger question: *which kinds of intelligence can be acquired by next-token prediction on text, and which cannot?* Chess says: not all of them. Not even simple ones.

---

## What this study does NOT show

- **It doesn't show LLMs can't reason at all.** It shows that *these* models, on *this* task, fail.
- **It doesn't include commercial models** (Claude, GPT-4o). Those may perform differently — that's the natural next step.
- **It doesn't test chess-fine-tuned models.** A model trained specifically on chess data would almost certainly do better; the question this study answers is about general-purpose pre-training.

---

## Bottom line

If you ever wondered whether ChatGPT and friends are "really thinking" or "just predicting words" — chess is a clean, specific, testable instance where the answer is **just predicting words**, and you can see it failing in real time.

22 models. 70 billion parameters at the top end. 4,000 positions, half a million evaluations. None of them can play.

---

*Full data, code, and methodology: github.com/rabrew/chess-llm-bench*
