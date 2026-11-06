# Explanation-quality scoring redesign — proposal, not a validated scorer
## Goal
Replace the use of side agreement plus keyword mention as a headline explanation-quality score for ALL local and commercial models. Preserve old outputs and distinguish lexical diagnostics from grounded explanation assessment. This is a technical protocol; it is not competition-report prose.

## Inputs / Outputs
Inputs: immutable FEN, prompt format and task, exact saved t3_explanation snippet, availability category, and verified board/reference evidence where available. Human-facing grading packets hide model/provider identity and existing scores. Use the same extracted T3 field for local and commercial answers; historical local full raw responses are unavailable.
Outputs proposed: claim-level supported/contradicted/unverified labels with evidence, separate factual-grounding and justification ratings, missingness and uncertainty flags, assessor IDs, rubric version and adjudication record. New quality scores remain unvalidated/unpublished until calibration and held-out checking. Existing t3_score and keyword scores remain versioned legacy proxies.

## Verified defects motivating the redesign
- Whole-substring matching gives the mate point for material. Shared word-boundary repair addresses this lexical bug only.
- A saved Luna answer on FEN r1bq3r/pp1p1kpp/2n3n1/2p4Q/4PR2/2NP4/PPP3PP/R5K1 b - - 0 12 receives the legacy2/2 while citing Nxf4. python-chess rejects that move in this position.
- The scorer does not receive FEN, so it cannot check the explanation's board claims or variations.
- Mentioning a motif, denying it, or inventing it can all earn the same keyword point.
- Puzzle themes describe annotated puzzle features, not an exhaustive list of valid reasons to evaluate a position.
- Half the old sum measures side classification and overlaps T1; it is not an independent explanation measure.

## Proposed rubric components
Keep components separate initially; do not publish an arbitrary weighted composite before validation.
1. Factual grounding: assess salient piece/square, material, check/attack and move-legality claims. Ratings0 central contradiction,1 mixed accuracy/minor errors,2 accurate specific grounding. If there are no checkable claims, label insufficient evidence rather than assuming full correctness.
2. Explanatory support: ratings0 absent/invalid central justification,1 valid but generic or incomplete support,2 specific coherent tactical/positional mechanism supporting the explanation. A legal-sounding variation must be checked from the correct position with its preceding moves.
3. Conclusion/reference consistency: report supported/contradicted/unassessable separately. It must not be silently added again to T3 as another T1 point. Current uncertain engine provenance cannot establish a reliable strategic verdict.
4. Error/availability flags: distinguish hallucinated board facts, illegal continuations, contradiction, generic unsupported claims, empty output, token truncation and unobserved transport outcome.
Clarity can be recorded descriptively, but grammar, length, confidence, vocabulary and verbosity must not earn chess-quality credit.
Correct alternative justifications can score well without naming the stored theme. Negation alone does not establish error; verify what the sentence claims in context.

## Steps / Logic
1. Freeze and retain legacy records. Rescore the diagnostic lexical matcher consistently for local and OpenAI data in new explicit outputs; do not overwrite original research artifacts.
2. Build the primary comparison on the same200 positions and fen_only prompt used for OpenAI, using saved local T3 snippets. Other local prompts/cohorts can be exploratory, labelled separately.
3. Preserve coverage: show all planned cases and distinct complete/missing/truncated/unobserved counts. Report conditional quality among assessable explanations alongside coverage, never turn unknown outputs into model mistakes.
4. Select a small stratified calibration pilot BEFORE grading, with local and both OpenAI models represented and model names blinded. An initial feasibility example is8 positions,2per tier, with6preselected models. Do not select models because of the observed T3 leaderboard.
5. Split calibration and held-out validation by POSITION so different model answers to one board cannot leak across the split. Keep balanced difficulty/provider coverage; randomize answer presentation and avoid visible paired-model groupings.
6. Ideally two chess-experienced assessors label independently. Reconcile disagreements only after preserving their original ratings. Freeze rubric, examples and any aggregation rule before opening held-out labels.
7. Report agreement per dimension with sample sizes and uncertainty. The small initial pilot is feasibility evidence, not adequate proof of scorer validity. Expand validation if disagreement exposes ambiguous criteria.
8. If automation is wanted, implement an auditable claim-checking or judge workflow only after a validated human reference set exists. Benchmark it against held-out labels across model families. Record exact judge versions/prompts and potential self/style bias; blinding names cannot guarantee independence.
9. Apply the SAME frozen validated scorer to local and commercial explanations and publish disagreements/uncertainty. Do not present post-hoc metric revisions as pre-registered.
No new LLM inference is needed to recover saved explanations. No paid judge, human rating claims or engine workloads are authorized by this spec alone.

## Edge cases
Conditional moves need sequential or branching board context; checking every mentioned move against the original FEN is invalid. Unsupported extraction means unverified, not contradicted. Several valid positional explanations may exist. Scarce verified engine evidence makes conclusions unassessable. A generic true sentence demonstrates less support than a specific correct reason but must not be called factually false. Refusals and truncated snippets are availability outcomes rather than automatically fabricated chess claims.

## Dependencies and decisions still needed
Existing python-chess for unambiguous legality/board facts; documented common engine evidence for strategic claims; at least one chess-competent human reviewer, preferably an independent second assessor. Human access, grading effort and any automated judge budget remain to be decided. No semantic grader or new headline quality score is implemented by the keyword-boundary fix.

## Review clarifications
With only one assessor, label ratings as exploratory expert assessment; do not claim inter-rater reliability or a validated objective quality ranking. Define any common-complete cohort against an explicit frozen model set: the194positions answered by both OpenAI models are not automatically complete for every local model. Report each model's coverage against the full200position frame and avoid selecting cases after seeing quality scores.

## Calibration revision: separate correctness, support, and verdict

Protocol revision: human_grounding_support_v2_candidate. This revision is informed
by the first four calibration positions and is not preregistered. The 24 original
ratings remain v1 observations, with an independently preserved correction record:
Answer 4 explanatory support 0 -> 2; Answer 21 explanatory support 2 -> 1.
Neither factual-grounding score changed. Do not relabel the original observations
as v2 ratings or silently revise other scores.

### Factual grounding
- 0: A central, explicitly asserted board fact or continuation is contradicted.
- 1: A minor or noncentral factual error among otherwise supported claims.
  A central contradiction takes precedence and receives 0 even if other facts are true.
- 2: Specific checkable factual claims are supported in their stated contexts.
- NA: No sufficiently checkable factual claims, or insufficient evidence.

A wrong overall winner does not by itself make all supporting board facts false.
An omitted best move affects explanatory completeness; it is not automatically
a fabricated board fact. A hypothetical future piece location must not be tested
as though it describes the initial board.

### Explanatory support
- 0: No reason is offered, or the central mechanism is demonstrably invalid.
- 1: A relevant valid reason is generic, incomplete, or omits a critical resource.
- 2: A specific, coherent reason adequately supports the stated assessment,
  without an identified decisive omission.
- NA: The central mechanism cannot be assessed with the available evidence.

A slight advantage may receive 2 when the explanation accurately describes a
slight advantage; magnitude alone is not a reason for 0. A description of an
attack does not establish a winning position if a critical defensive resource
is omitted. Preserve uncertainty instead of equating an unverified engine verdict
or a human conjecture with an established chess fact. Do not require a particular
puzzle theme, exact move, long variation, or verbosity when another valid reason
suffices.

### Human evidence and automation limits
The corrected calibration file contains one assessor's judgments, not verified
engine truth. Flags, notes about perpetual checks, proposed best moves, and
material statements require independent context-sensitive checks before becoming
automated ground truth. Agreement with these labels measures agreement with this
assessor on four boards; it does not establish inter-rater reliability.

The first automated revision may extract only a documented narrow subset of
unambiguous present-board claims. Return evidence for each supported,
contradicted, or unverified claim, retain exact text and source fingerprints,
and report extraction coverage. Unsupported language, negation, conditional
moves, future/past claims and variation contexts must lead to abstention where
their meaning cannot be resolved. No extracted contradiction does not imply
correctness. Counting correct square mentions does not establish explanatory
support. The automatic whole-explanation grounding and support ratings remain
null until an independently evaluated method exists.

Legacy side-plus-keyword scores and the repaired lexical matcher remain separately
named diagnostics. Compare them descriptively against both human components;
do not fit weights or optimize thresholds on 24 answers. Do not combine the two
human components into an arbitrary total, or treat equal numeric scales as equal
constructs.

### Frozen evaluation boundary
Use only the four calibration positions for developing this revision. Do not
inspect held-out answer text or labels to tune rules. Before a held-out round,
freeze the implemented diagnostic version, parser scope and rubric instructions.
Prepare a new blinded packet explicitly labelled with the new rubric version.
Any held-out feedback used to change rules becomes development evidence and
requires a fresh independent evaluation set. Four more positions remain a small
feasibility check rather than sufficient validation of a general semantic scorer.
