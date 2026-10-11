"""One real Hermes conversation, run by the supervisor as the unprivileged child.

The Agenomic adapter is loaded by Hermes through its ``hermes_agent.plugins``
entry point because ``config.yaml`` lists it in ``plugins.enabled``. The
provider is the Agenomic Model Gateway; no provider key exists in this
process. Prints one ``RESULT:<json>`` line.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    prompt = sys.argv[1]
    from agent.shell_hooks import register_from_config
    from hermes_cli.config import load_config
    from run_agent import AIAgent

    cfg = load_config()
    register_from_config(cfg, accept_hooks=True)
    model = cfg.get("model", {})
    agent = AIAgent(
        base_url=model["base_url"],
        api_key=os.environ["AGENOMIC_HERMES_RUNTIME_TOKEN"],
        provider="custom",
        api_mode="chat_completions",
        model=model.get("default", "demo-model"),
        enabled_toolsets=["file", "terminal", "delegation"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        max_iterations=6,
        session_db=None,
        platform="cli",
    )
    result = agent.run_conversation(prompt, task_id="demo")
    from tools.delegate_tool_registry import list_active_subagents

    deadline = time.time() + 90
    while list_active_subagents() and time.time() < deadline:
        time.sleep(0.5)
    module = sys.modules.get("agenomic.integrations.hermes.plugin")
    adapter = module.current_adapter() if module else None
    if adapter is not None:
        adapter.exporter.flush(10.0)
    tools = [
        m.get("content")
        for m in (result.get("messages") or [])
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    sys.stdout.write(
        "RESULT:"
        + json.dumps(
            {
                "session_id": agent.session_id,
                "final": result.get("final_response"),
                "tool_messages": tools,
                "adapter_loaded": adapter is not None,
            },
            default=str,
        )
        + "\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
