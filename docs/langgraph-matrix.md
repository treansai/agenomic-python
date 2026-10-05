# LangGraph version matrix

The `langgraph` and `all` extras accept `langgraph>=1.0.10,<2` with
`langchain-core>=1.6.3,<2`. Inside that range, only the points below are
tested with the managed prompts adapter (`bind_langgraph`) and the rest of
the test suite. Any other version installs and runs, and the adapter emits
one `AgenomicUntestedVersionWarning` per process. The tested versions are
`TESTED_LANGGRAPH` and `TESTED_LANGCHAIN_CORE` in
`agenomic.integrations.langgraph_binding`.

## Points

Each point is a constraint file under `ci/constraints/`. The SQLite saver
and `aiosqlite` are only needed by the restart tests and by example 15
(the `dev` extra).

| Package                       | Primary | Floor  |
| ----------------------------- | ------- | ------ |
| `langgraph`                   | 1.2.11  | 1.0.10 |
| `langgraph-checkpoint`        | 4.2.0   | 4.2.0  |
| `langgraph-prebuilt`          | 1.1.0   | 1.0.8  |
| `langgraph-sdk`               | 0.4.4   | 0.3.15 |
| `langchain-core`              | 1.6.3   | 1.6.3  |
| `langgraph-checkpoint-sqlite` | 3.1.1   | 3.1.1  |
| `aiosqlite`                   | 0.22.1  | 0.22.1 |

- Primary: `ci/constraints/langgraph-1.2.11.txt`.
- Floor: `ci/constraints/langgraph-1.0.10.txt`. It pins
  `langgraph-prebuilt==1.0.8` because `langgraph==1.0.10` alone installs
  `langgraph-prebuilt` 1.0.13, and `import langgraph.prebuilt` then fails
  with `ImportError: cannot import name 'ExecutionInfo' from
  'langgraph.runtime'`.

## Verified cells

Status on 2026-10-05. Every cell below ran the whole test suite
(`pytest --no-cov`) locally on macOS 26.3 (arm64), in a virtual environment
installed with `uv pip install -e ".[dev,all]" -c <constraint file>`, whose
packages of the table above match the point exactly.

| Point   | Python  | Result                |
| ------- | ------- | --------------------- |
| primary | 3.10.18 | 963 passed, 2 skipped |
| primary | 3.11.11 | 965 passed            |
| primary | 3.12.11 | 965 passed            |
| primary | 3.13.5  | 965 passed            |
| floor   | 3.10.18 | 963 passed, 2 skipped |
| floor   | 3.13.5  | 965 passed            |

On the primary point with Python 3.10, the full check also passed in an
environment without the SQLite saver: `ruff check`, `ruff format --check`,
`mypy src` and `pytest --cov=agenomic --cov-fail-under=85` (959 passed,
6 skipped, coverage 91.10 %).

The skipped tests:

- `test_langgraph_binding_resume.py::test_async_interrupt_resume` skips
  below Python 3.11: LangGraph's `interrupt()` inside asyncio tasks needs
  the context variable propagation of Python 3.11.
- `test_langgraph_binding_subgraphs.py::test_nested_bind_without_pin_refused_async`
  skips below Python 3.11: the LangChain run context variable does not reach
  asyncio tasks there, so a nested bind cannot be detected on Python 3.10
  async.
- `test_langgraph_binding_resume.py::test_interrupt_resume_after_real_process_restart_sqlite`
  (online and offline),
  `::test_factory_execution_scope_resume_after_real_process_restart_sqlite`
  and `::test_outage_after_real_process_restart_resumes_disk_cached_binding`
  skip when `langgraph-checkpoint-sqlite` is not installed.

## Differences between the points

- `astream_events(version="v3")` passes through on the primary point.
  LangGraph 1.0.10 raises `NotImplementedError` for it, and the proxy passes
  that error through.
- `create_react_agent` warns with `LangGraphDeprecatedSinceV10` on both
  points and works with `managed_prompt` and `AgentFactory`.
- `instrument_langgraph` and `instrument_langgraph_canonical` record nothing
  on a real `StateGraph` or compiled graph on either point (see
  [Integrations](integrations.md#langgraph)).

## Not verified

- Linux and Windows. The CI workflow runs the primary point on Python 3.10
  to 3.13 on Ubuntu, macOS and Windows, and the floor point on Python 3.10
  and 3.13 on Ubuntu, but no CI run of these cells is recorded here yet.
- The `langgraph-latest` job, run weekly or on demand, installs the newest
  `langgraph<2` and `langchain-core<2` without constraints. It may fail and
  only reports drift; nothing it installs is a tested point.
- `langchain.agents.create_agent` and its middleware, and graphs served by
  LangGraph Server or LangGraph Platform, which bypass the proxy.

## Adding a point

Add the constraint file, the CI job or matrix entry, the version to
`TESTED_LANGGRAPH` or `TESTED_LANGCHAIN_CORE`, and the cells to this file
together, and list a cell here only after its tests have passed.
