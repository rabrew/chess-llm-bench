"""Docs must describe the engines actually used (D9, specs/metric-defect-remediation.md).

T1 truth is a single forward pass of the Lc0 network (no search), saturating at
±10,000 cp. Stockfish was never used in the shipped run, and nothing reaches ±16,000.
"""

from pathlib import Path

import pytest

DOCS = ["README.md", "docs/FINDINGS.md", "docs/SIMPLE_SUMMARY.md", "specs/benchmark.md", "specs/benchmark-detailed.md"]
STALE_CLAIMS = ["Stockfish 17", "Stockfish-17", "depth 22", "depth-22", "16,000", "±16000"]


@pytest.mark.parametrize("doc", DOCS)
def test_no_stale_engine_claims(doc):
    text = Path(doc).read_text()
    found = [claim for claim in STALE_CLAIMS if claim in text]
    assert not found, f"{doc} still says {found}"
