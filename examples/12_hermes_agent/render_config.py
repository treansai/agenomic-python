"""Render and validate the Hermes configuration for an Agenomic controlled runtime.

Offline: prints the Hermes ``config.yaml`` fragment and checks that the adapter
config file in this directory is valid. Run with ``python render_config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from agenomic.integrations.hermes.config import build_config, render_hermes_config

HERE = Path(__file__).parent

rendered = render_hermes_config("https://agenomic.example", model="demo-model")
print(yaml.safe_dump(rendered, sort_keys=False))

env = {
    "AGENOMIC_HERMES_CONFIG": str(HERE / "agenomic-adapter.yaml"),
    "AGENOMIC_HERMES_RUNTIME_TOKEN": os.environ.get(
        "AGENOMIC_HERMES_RUNTIME_TOKEN", "agmhr_example"
    ),
}
config, _ = build_config({}, environ=env)
print(f"adapter config OK: endpoint={config.endpoint} capture={config.capture.content}")

template = yaml.safe_load((HERE / "hermes-config.yaml").read_text(encoding="utf-8"))
assert template["model"] == {**rendered["model"]}
assert template["hooks"] == rendered["hooks"]
print("hermes-config.yaml matches the rendered model and hooks blocks")
