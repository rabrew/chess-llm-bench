# Human calibration versus lexical proxies and conservative board facts

## Goal

Audit corrected exploratory human calibration labels against saved legacy T3 and
current lexical diagnostics equally for local and commercial systems. Produce
board-fact evidence where narrowly justified; do not fabricate semantic grades.

## Inputs / Outputs

Explicit calibration pack/private key, corrected ratings, original ratings,
historical local JSONL, commercial frozen manifest/record directory, and declared
local model list. Validate pack/rubric and exact unique calibration IDs, input
hashes, model+FEN+fen_only joins and exact saved snippets. Held-out labels are
never accepted. Stream local records once across the full200-position frame for
all22local models plus both commercial models, not only six calibration models.

New external output: public summary.json with per-model aggregate calibration
counts and separate human dimensions, and coverage/evidence-count diagnostics.
private/calibration.json and private/frame-diagnostics.jsonl preserve joins/evidence.
No held-out text, answer identity mapping or FEN appears in public summary.

## Logic

Retain original/corrected file and source fingerprints. Validate numeric human
labels0/1/2, NA and missing independently; never treat unavailable as zero.
Legacy0–2 proxy versus each human dimension gets descriptive contingency,
numeric exact-match and MAE, explicitly different constructs rather than scorer
validation. Binary current keyword mention gets a contingency only: no invented
scale mapping, combined human score, fitted weights, significance test or ranking.
Only four independent calibration boards and one assessor: exploratory only.

A shared provider-neutral board checker recognizes explicit present-board
color+piece+on/at+square phrases. Compare only piece occupancy at that square.
Any hypothetical/conditional, negated, past/future or variation context causes
abstention. Unsupported claims stay unverified. Never test every SAN token on
the initial board; branches need sequential context. Supported extracted facts
or zero detected errors cannot establish whole-explanation accuracy/support.
Full semantic fields remain null with status requires_human_review.

## Edge cases / Tests

Duplicate IDs/source joins, wrong pack/rubric, held-out IDs, snippet mismatches,
invalid ratings, changed source hashes, missing/truncated/unobserved records,
conditional and negated board claims, past/branch moves, unsupported syntax,
privacy of public summary, symlinks and reused output. Fail closed on conflicts;
outputs only after validation. Tests use synthetic/temp files with no APIs,
engines, GPU, dataset generation or research-artifact writes.

## Dependencies

Existing python-chess, shared lexical matcher and JSON helpers; standard library.

## Review refinements before execution

Comparison outputs use descriptive cross-tabs and discordance counts only; MAE
and numeric exact-match are omitted because the constructs differ. No scorer
accuracy or same-construct agreement is claimed. Explicit unowned “the rook on
d1” checks only type/square with asserted_color null; never infer color from a
preceding evaluation verdict. Negated existence (absent/missing), quotations,
conditional/future/past/variation context abstain. Publish availability alongside
answers with/no extracted claims and literal-only evidence counts. Specific or
verbose answers expose more checkable claims; vague answers can escape checking.
Never rank models by fewest flags or interpret no flags as accurate explanation.
Record Python/python-chess versions and shared evaluator source fingerprint.
