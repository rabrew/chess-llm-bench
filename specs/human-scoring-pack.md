# Blinded offline human explanation calibration pack

## Goal

Let a human independently assess saved explanations using two separate exploratory
ratings. One assessor and24cards do not establish inter-rater reliability or a
validated quality ranking. No model calls, new explanations or automatic grades.

## Inputs / Outputs

Read the frozen200-position commercial manifest and exact saved fen_only T3
snippets for four predeclared local models and both OpenAI models. Stream local
JSONL once. Explicit exclusion JSON contains previously exposed FENs, excluded
only for blinding. Seed42 selects2positions per difficulty tier; the first of each
tier is calibration and the other held-out, before checking answer availability.
All6models remain represented; missing saved answers stay missing without resampling.

Create a new external output directory with private/key.json, containing8boards,
held-out snippets, identities, split, exclusions and source fingerprints. Public
calibration.html and manifest.json contain only4calibration boards and24opaque
answer IDs, never model/job/theme/rating/tier/heldout metadata. Private directories
mode700 and files600. Fail on reused output, symlinks, duplicate matched answers,
conflicting FEN/position metadata, malformed sources or self-identifying snippets.

## Logic

Use a private random blinding seed (retained in key) for opaque IDs and a global
24-card shuffle; selection itself remains reproducible with seed42. White-bottom
SVG boards show FEN and side-to-move. Display exact saved snippets with HTML
escaping; never expose full commercial raw chains. Missing answers have disabled
NA controls and are not called wrong. No theme/legacy score is disclosed.

Each card has factual grounding and explanatory support selectors: blank default,
0 central contradiction/invalid explanation,1mixed/generic/partial,2specific
accurate/coherent, NA cannot assess. Optional notes and uncertainty flags, assessor
alias, guarded pack-specific localStorage and JSON export. No combined sum.
Strategic uncertainty without verified reference means NA; check branch moves
sequentially, not each move against the initial board. No credit for verbosity or
keywords. All controls work offline without external assets or requests.

## Edge cases / Tests

Deterministic stratified selection, disjoint position splits, missing versus0,
no identities/held-out leakage in public source, HTML and closing-script injection,
original snippet preservation, duplicates/conflicts, output reuse/symlink rejection,
explicit blank defaults, guarded storage, ratings-only export metadata.

## Dependencies

Existing python-chess SVG and Python standard library. Tests use temporary
synthetic records only. Parent builds real output after independent review.

## Review refinements

Public availability is only available/unavailable, while detailed missingness
categories remain private. Export null ratings plus explicit neutral availability
for unavailable cards, distinguishing them from ungraded available cards. Reject
versioned model tags, display aliases and provider aliases at whole-word boundaries.
Friendly Answer1–24 labels accompany opaque exported IDs. Randomize board-list
order privately before assigning BoardA–D so label order cannot reveal difficulty.
A sticky export toolbar and progress counter expose no grading or identity hints.
