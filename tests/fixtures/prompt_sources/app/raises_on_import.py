from pathlib import Path

Path(__file__).with_name("executed.marker").write_text("scanned code ran")
raise RuntimeError("discovery must never import this module")

ONBOARDING_PROMPT = "Welcome {customer} to the support desk."
