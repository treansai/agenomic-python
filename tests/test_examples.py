"""Smoke-test the offline examples by invoking them."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

OFFLINE_EXAMPLES = [
    "01_minimal_trace.py",
    "02_decorator_jsonl.py",
    "03_atep_local.py",
    "05_langgraph_traced.py",
    "07_offline_signed_release.py",
]


@pytest.mark.parametrize("script", OFFLINE_EXAMPLES)
def test_example_runs(script: str) -> None:
    path = EXAMPLES_DIR / script
    assert path.exists()
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
