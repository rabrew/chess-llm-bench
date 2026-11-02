"""Safe static checks for lifecycle helpers; never execute workloads."""
import ast
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("script", sorted((ROOT / "bin").glob("*.sh")), ids=lambda path: path.name)
def test_shell_syntax_and_portability(script):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    assert "/home/rabrew/Desktop/chess-llm-bench" not in script.read_text()

def test_model_passed_as_argument():
    script = (ROOT / "bin/run_all.sh").read_text()
    assert "('$model',)" not in script
    assert "(sys.argv[1],)" in script

def test_parser_methods_are_unique():
    tree = ast.parse((ROOT / "tests/test_parse_response_formats.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
            assert len(names) == len(set(names)), node.name
