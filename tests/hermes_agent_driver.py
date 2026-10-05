"""Runs one real Hermes ``AIAgent`` conversation in a fresh process.

Used by ``test_hermes_integration.py``: each test gets its own process so
``HERMES_HOME`` (set by the parent before anything imports Hermes) and the
plugin manager are isolated. Prints one ``RESULT:<json>`` line.

argv[1] is a JSON object: ``base_url``, ``api_key``, ``prompt``, ``shell_hooks`` (bool).
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    cfg = json.loads(sys.argv[1])
    assert os.environ.get("HERMES_HOME"), "HERMES_HOME must be set by the parent"
    from run_agent import AIAgent

    if cfg.get("shell_hooks"):
        from agent.shell_hooks import register_from_config
        from hermes_cli.config import load_config

        register_from_config(load_config(), accept_hooks=True)

    agent = AIAgent(
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key", "dummy"),
        provider="custom",
        api_mode="chat_completions",
        model="demo-model",
        enabled_toolsets=["file"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        max_iterations=4,
        session_db=None,
        platform="cli",
    )
    result = agent.run_conversation(cfg["prompt"], task_id="t1")

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    plugins = {
        key: {"source": lp.manifest.source, "enabled": lp.enabled, "error": lp.error}
        for key, lp in manager._plugins.items()
    }
    adapter_module = sys.modules.get("agenomic.integrations.hermes.plugin")
    adapter = adapter_module.current_adapter() if adapter_module else None
    if adapter is not None:
        adapter.exporter.flush(5.0)
    tool_messages = [
        m.get("content")
        for m in (result.get("messages") or [])
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    out = {
        "final": result.get("final_response"),
        "plugins": plugins,
        "tool_messages": tool_messages,
        "session_id": agent.session_id,
        "adapter_imported_by_entry_point": adapter is not None,
    }
    sys.stdout.write("RESULT:" + json.dumps(out, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
