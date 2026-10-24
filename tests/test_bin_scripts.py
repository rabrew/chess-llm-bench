"""bin/ scripts locate the repo themselves (specs/repo-relative-scripts.md)."""

from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "bin"
SCRIPTS = sorted(BIN.glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_hardcoded_home_paths(script):
    assert "/home/" not in script.read_text(), f"{script.name} hardcodes a home directory"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_cd_targets_are_derived_from_script_location(script):
    for line in script.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("cd ") and not stripped.startswith("cd -"):
            assert "$REPO" in stripped or "dirname" in stripped, f"{script.name}: {stripped}"
