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
    "09_rmp.py",
    "12_vault_langgraph.py",
    "12_managed_prompts_stategraph.py",
    "13_langgraph_subagent_prompts.py",
    "14_langgraph_async_streaming.py",
    "15_langgraph_interrupt_restart.py",
    "16_langgraph_promotion_pinning.py",
    "17_prompt_experiment_counterfactual.py",
    "12_hermes_agent/render_config.py",
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
        f"{script} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("variable", ["AGENOMIC_ENDPOINT", "AGENOMIC_BASE_URL"])
def test_tool_execution_example_reads_either_endpoint_variable(variable: str) -> None:
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENOMIC_")}
    env[variable] = "http://127.0.0.1:9"
    env["AGENOMIC_API_KEY"] = "key"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLES_DIR / "10_tool_execution.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert "to run this example" not in result.stdout
    assert "transport_error" in result.stderr
