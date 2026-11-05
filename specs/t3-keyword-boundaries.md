# T3 keyword boundary correction

## Goal

Remove accidental substring theme matches consistently for local and commercial
models. A keyword-presence score remains a lexical proxy, not semantic quality.

## Inputs / Outputs

Shared evaluator.match_t3_theme(theme, explanation) returns1 or0 for string
inputs and null for missing/invalid inputs. Matcher version keyword_boundaries_v1.
Existing score_t3 and metrics.rescore_t3_theme delegate to this single helper;
score_t3 preserves its previous zero-theme-credit behavior for missing themes.
No raw stored scores, records or research outputs are rewritten.

## Logic

Retain current exact-label synonyms, lowercase matching, camelCase-to-words and
underscore-to-space fallback candidates. Escape candidates as literal regex text;
match only when neither adjacent character is a Unicode word character, using
negative word-character lookarounds. Empty candidates never match. Explicit map
aliases such as checkmate, pinned and pinning remain valid.

## Edge cases

mate must not match material, pin must not match spinning, fork must not match
forklift. Equality's even must not match seven; Unicode letters, digits and
underscores block accidental embedded matches. Punctuation delimiters permit
legitimate mentions. Unknown themes retain literal/camel/underscore fallback.
Missing text remains null in rescoring; existing side-credit calculations stay
unchanged. Negation, factual chess correctness and explanation quality are not
resolved: “no mate” still mentions mate and receives lexical theme credit.

## Dependencies / Tests

Python standard re module, existing theme aliases. Tests cover false positives,
legitimate aliases, Unicode boundaries, invalid inputs and equality of both public
scoring paths. CPU-only temporary/unit fixtures; no model/engine/API calls.

## Curated inflections: keyword_boundaries_v2

Add explicit grammatical forms for common fork, pin and mate terms: forks,
forked, forking, pins, mates, mated, checkmates, checkmated and checkmating.
Whole-word boundaries remain mandatory; these are curated aliases, not stemming.
Bump matcher identifier to keyword_boundaries_v2 so existing v1 audit artifacts
remain distinguishable and untouched. Negation/semantic truth remain outside scope.
