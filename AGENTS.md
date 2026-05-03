# agentlock-python — agent instructions

Public Python SDK for AgentLock. Apache-2.0.

## Product invariants

1. Works fully offline. No primitive requires network.
2. ATEP-native. Segments produced here are bit-for-bit compatible with
   agentlock-cli and agentlock-cloud.
3. Integrations are optional and lazy. Top-level imports never load
   openai, anthropic, langgraph or mcp.

## Engineering rules

- mypy strict. No `Any` in public APIs.
- pydantic v2 for all data models.
- No `print()` in library code. Use `logging.getLogger("agentlock.<module>")`.
- All public functions have docstrings with at least one example.
- Async-first for I/O-bound primitives (HTTP, file streams). Provide sync
  wrappers via `asyncio.run` only at the top level.
- No bare `except:`. Catch specific exceptions.

## Naming

- Package: `agentlock`
- CLI: `agentlock-py` (entry point)
- ATEP file extension: `.atep`
- Default config dir: `~/.config/agentlock/`

## Security defaults

- Redaction runs BEFORE any export.
- ed25519 PEM keys loaded with file mode check (warn if not 0600).
- HTTP client uses TLS verify=True. No `verify=False` flag exposed.
