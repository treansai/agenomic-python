#!/usr/bin/env bash
# Acceptance gate — runs the full pipeline end-to-end.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> install"
$PY -m pip install --quiet -e ".[dev,all]"

echo "==> ruff"
ruff check src tests examples
ruff format --check src tests examples

echo "==> mypy"
mypy src

echo "==> pytest"
pytest --cov=agenomic --cov-fail-under=85

echo "==> example: minimal trace"
$PY examples/01_minimal_trace.py > /dev/null

echo "==> example: decorator + jsonl"
$PY examples/02_decorator_jsonl.py > /dev/null

echo "==> example: atep local"
$PY examples/03_atep_local.py > /dev/null

echo "==> example: langgraph (mock shim)"
$PY examples/05_langgraph_traced.py > /dev/null

echo "==> example: offline signed release"
$PY examples/07_offline_signed_release.py > /dev/null

echo "==> CLI: keys generate"
agenomic-py keys generate "$WORK/key.pem"

echo "==> CLI: atep verify (golden)"
agenomic-py atep verify \
  tests/fixtures/golden_atep_segments/golden_v1.atep \
  --public-key tests/fixtures/golden_atep_segments/golden_pub.pem

echo "==> CLI: atep inspect (golden)"
agenomic-py atep inspect tests/fixtures/golden_atep_segments/golden_v1.atep > /dev/null

echo
echo "All checks passed."
