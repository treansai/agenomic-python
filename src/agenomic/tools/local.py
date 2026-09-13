"""In-memory tool execution engine used by ``client.tools`` in local mode.

It mirrors the cloud contract for what can honestly run in-process:
structural validation, preflight plans, the run lifecycle with the same
statuses and approval rule, the ``static``, ``rules`` and ``recorded`` mock
strategies, and the two-phase protocol for runtime-local functions. What
needs the gateway (``mcp`` and ``http`` adapters, ``scenario``,
``schema_generated`` and ``plugin`` strategies, connection tests) is refused
by the plan or the call with an explicit error, never degraded silently.

Hashes are ``sha256:`` over sorted compact JSON; they are not comparable to
the cloud's ``blake3:`` hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from agenomic.tools.models import (
    TOOL_EXECUTION_SCHEMA_VERSION,
    ToolExecutionError,
)

_ENV_REF = re.compile(r"(?<!\$)\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_POSITIONS = ("endpoint", "headers", "env", "query")
_LIVE_ADAPTERS = ("mcp", "http", "local")
_MOCK_STRATEGIES = ("static", "rules", "recorded", "scenario", "schema_generated", "plugin")
_OFFLINE_STRATEGIES = ("static", "rules", "recorded")
_WRITE_EFFECTS = ("reversible_write", "irreversible_write", "unknown")
_LOCAL_NOTE = "agenomic-python local engine: not a cloud gateway result"


def canonical_hash(value: Any) -> str:
    """``sha256:`` over sorted compact JSON.

    Example:
        >>> canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
        True
    """
    canon = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _invalid(message: str) -> ToolExecutionError:
    return ToolExecutionError("tool_execution_config_invalid", message, 400)


def _binding_invalid(tool: str, message: str) -> ToolExecutionError:
    return ToolExecutionError("tool_binding_invalid", f"bindings.{tool}: {message}", 400)


def _scan_refs(
    value: Any, path: str, forbidden: list[str], allowed: set[str], secret: bool
) -> None:
    if isinstance(value, str):
        names = _ENV_REF.findall(value)
        if names and not secret:
            forbidden.append(path)
        allowed.update(names) if secret else None
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_refs(item, f"{path}.{key}", forbidden, allowed, secret)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_refs(item, f"{path}[{index}]", forbidden, allowed, secret)


class LocalToolEngine:
    """State and behaviour of local-mode tool execution.

    Example:
        >>> engine = LocalToolEngine()
        >>> report = engine.validate({"schema_version": "agenomic.tool_execution/v1", "mode": "mock"}, 1)
        >>> report["config"]["mode"]
        'mock'
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[str, dict[str, Any]] = {}
        self._variables: dict[tuple[str, str], tuple[str, int]] = {}
        self._contracts: dict[str, dict[str, Any]] = {}
        self._fixture_sets: dict[str, dict[str, Any]] = {}
        self._scenarios: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._invocations: dict[str, list[dict[str, Any]]] = {}

    # ── profiles and variables ────────────────────────────────────────

    def create_profile(self, name: str, environment: str, allowed_env: list[str]) -> dict[str, Any]:
        with self._lock:
            if any(p["name"] == name for p in self._profiles.values()):
                raise ToolExecutionError("conflict", f"profile {name} already exists", 409)
            profile: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "org_id": "local",
                "name": name,
                "environment": environment,
                "allowed_env": list(allowed_env),
                "network": {"allow_private_destinations": False, "allowed_hosts": []},
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._profiles[profile["id"]] = profile
            return {"profile": dict(profile), "variables": self._variable_status(profile)}

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._profiles.values()]

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ToolExecutionError("not_found", "execution profile not found", 404)
            return {"profile": dict(profile), "variables": self._variable_status(profile)}

    def set_variable(self, profile_id: str, name: str, value: str) -> dict[str, Any]:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ToolExecutionError("not_found", "execution profile not found", 404)
            if name not in profile["allowed_env"]:
                raise _invalid(
                    f"variable {name} is not in the profile allowlist; add it to allowed_env first"
                )
            _, version = self._variables.get((profile_id, name), ("", 0))
            self._variables[(profile_id, name)] = (value, version + 1)
            return {"name": name, "source": "stored", "version": version + 1, "available": True}

    def _variable_status(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for name in profile["allowed_env"]:
            stored = self._variables.get((profile["id"], name))
            out.append(
                {
                    "name": name,
                    "source": "stored",
                    "version": stored[1] if stored else 0,
                    "available": stored is not None,
                    "updated_at": profile["updated_at"],
                }
            )
        return out

    # ── contracts, fixture sets, scenarios ────────────────────────────

    def create_contract(self, contract: dict[str, Any]) -> dict[str, Any]:
        ref = f"{contract['name']}@{contract['version']}"
        with self._lock:
            if ref in self._contracts:
                raise ToolExecutionError("conflict", f"contract {ref} already exists", 409)
            stored = {"id": str(uuid.uuid4()), "org_id": "local", **contract, "created_at": _now()}
            self._contracts[ref] = stored
            return {"contract": dict(stored)}

    def create_fixture_set(
        self, name: str, version: int, fixtures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ref = f"{name}@{version}"
        with self._lock:
            if ref in self._fixture_sets:
                raise ToolExecutionError("conflict", f"fixture set {ref} already exists", 409)
            seen: set[str] = set()
            normalized = []
            for fixture in fixtures:
                fixture_id = str(fixture.get("fixture_id", ""))
                request = dict(fixture.get("request") or {})
                if not fixture_id or fixture_id in seen or "tool" not in request:
                    raise _invalid("each fixture needs a unique fixture_id and a request.tool")
                seen.add(fixture_id)
                expected = canonical_hash(request.get("arguments", {}))
                supplied = str(request.get("arguments_hash") or "")
                if supplied and supplied != expected:
                    raise _invalid(
                        f"fixture {fixture_id} arguments_hash does not match its arguments"
                    )
                request["arguments_hash"] = expected
                normalized.append({**dict(fixture), "request": request})
            stored = {
                "id": str(uuid.uuid4()),
                "org_id": "local",
                "schema_version": "agenomic.tool_fixture/v1",
                "name": name,
                "version": version,
                "approved": False,
                "fixtures": normalized,
                "created_at": _now(),
            }
            self._fixture_sets[ref] = stored
            return {"fixture_set": dict(stored)}

    def approve_fixture_set(self, fixture_set_id: str) -> dict[str, Any]:
        with self._lock:
            for stored in self._fixture_sets.values():
                if stored["id"] == fixture_set_id:
                    stored["approved"] = True
                    return {"fixture_set": dict(stored)}
        raise ToolExecutionError("not_found", "fixture set not found", 404)

    def list_fixture_sets(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(s) for s in self._fixture_sets.values()]

    def create_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        ref = f"{scenario['name']}@{scenario['version']}"
        with self._lock:
            if ref in self._scenarios:
                raise ToolExecutionError("conflict", f"scenario {ref} already exists", 409)
            stored = {"id": str(uuid.uuid4()), "org_id": "local", **scenario, "created_at": _now()}
            self._scenarios[ref] = stored
            return {"scenario": dict(stored)}

    def adapters(self) -> dict[str, Any]:
        return {
            "adapter_interface_version": "agenomic.tool_adapter/v1",
            "adapters": [
                {
                    "kind": "local",
                    "version": "local-engine",
                    "capabilities": {"structured": True, "streaming": False, "artifacts": False},
                    "note": "runtime-local functions through the router; mcp and http adapters need Agenomic Cloud",
                }
            ],
        }

    # ── validation and plans ──────────────────────────────────────────

    def validate(self, config: Mapping[str, Any], repetitions: int) -> dict[str, Any]:
        if not isinstance(config, Mapping):
            raise _invalid("configuration must be a mapping")
        cfg = dict(config)
        if len(cfg) == 1 and isinstance(cfg.get("tool_execution"), Mapping):
            cfg = dict(cfg["tool_execution"])
        if not isinstance(repetitions, int) or repetitions < 1 or repetitions > 500:
            raise _invalid("repetitions must be an integer between 1 and 500")
        if cfg.get("schema_version") != TOOL_EXECUTION_SCHEMA_VERSION:
            raise _invalid(f"schema_version must be {TOOL_EXECUTION_SCHEMA_VERSION!r}")
        mode = cfg.get("mode")
        if mode not in ("mock", "live", "hybrid"):
            raise _invalid("mode must be mock, live or hybrid")
        if cfg.get("default_mode", "mock") != "mock":
            raise _invalid("default_mode must be mock")
        if cfg.get("on_unmatched", "error") != "error":
            raise _invalid("on_unmatched accepts only error")
        safety = dict(cfg.get("safety") or {})
        if safety.get("allow_implicit_fallback", False):
            raise _invalid(
                "safety.allow_implicit_fallback must be false: a missing mock never falls back to a live call"
            )
        limits = dict(cfg.get("limits") or {})
        if int(limits.get("max_concurrency", 1)) < 1:
            raise _invalid("limits.max_concurrency must be at least 1")
        bindings = cfg.get("bindings") or {}
        if not isinstance(bindings, Mapping):
            raise _invalid("bindings must be a mapping")
        warnings: list[str] = []
        required_env: dict[str, list[str]] = {}
        if int(limits.get("max_concurrency", 1)) > 1:
            warnings.append(
                "limits.max_concurrency is recorded in the plan but not enforced; the runtime controls concurrency"
            )
        for tool, raw in bindings.items():
            if not isinstance(raw, Mapping):
                raise _binding_invalid(tool, "binding must be a mapping")
            binding = dict(raw)
            binding_mode = binding.get("mode", "mock")
            if binding_mode not in ("mock", "live"):
                raise _binding_invalid(tool, "mode must be mock or live")
            if mode == "mock" and binding_mode == "live":
                raise _binding_invalid(tool, "live bindings are not allowed in mode: mock")
            if mode == "live" and binding_mode == "mock":
                raise _binding_invalid(
                    tool, "mock bindings are not allowed in mode: live; use mode: hybrid"
                )
            forbidden: list[str] = []
            names: set[str] = set()
            for key, value in binding.items():
                _scan_refs(
                    value, f"bindings.{tool}.{key}", forbidden, names, key in _SECRET_POSITIONS
                )
            if forbidden:
                raise ToolExecutionError(
                    "env_reference_forbidden_location",
                    f"environment references are not allowed at {forbidden[0]}",
                    400,
                )
            if binding_mode == "live":
                adapter = binding.get("adapter")
                if adapter not in _LIVE_ADAPTERS:
                    raise _binding_invalid(
                        tool, "live bindings require an adapter (mcp, http or local)"
                    )
                if adapter == "local" and not binding.get("function"):
                    raise _binding_invalid(tool, "local adapter requires a function")
                if adapter != "local" and not binding.get("endpoint"):
                    raise _binding_invalid(tool, f"{adapter} adapter requires an endpoint")
                if binding.get("strategy") is not None:
                    raise _binding_invalid(tool, "live bindings must not declare mock fields")
                if binding.get("effect", "unknown") == "unknown":
                    warnings.append(
                        f"bindings.{tool}: effect is unknown and will be treated as a write"
                    )
                if binding.get("policy_ref"):
                    warnings.append(
                        f"bindings.{tool}.policy_ref is recorded in the plan but not evaluated"
                    )
                if names:
                    required_env[tool] = sorted(names)
            else:
                strategy = binding.get("strategy")
                if strategy not in _MOCK_STRATEGIES:
                    raise _binding_invalid(tool, "mock bindings require a strategy")
                if binding.get("adapter") or binding.get("endpoint"):
                    raise _binding_invalid(tool, "mock bindings must not declare live fields")
                if strategy == "static" and not isinstance(binding.get("response"), Mapping):
                    raise _binding_invalid(tool, "static strategy requires a response")
                if strategy == "rules" and not isinstance(binding.get("rules"), list):
                    raise _binding_invalid(tool, "rules strategy requires a rules list")
                if strategy == "recorded" and not binding.get("fixture_set_ref"):
                    raise _binding_invalid(tool, "recorded strategy requires fixture_set_ref")
                if strategy == "scenario" and not binding.get("scenario_ref"):
                    raise _binding_invalid(tool, "scenario strategy requires scenario_ref")
                if binding.get("contract_ref") is None:
                    warnings.append(f"no contract for {tool}")
        return {
            "config": cfg,
            "config_hash": canonical_hash(cfg),
            "warnings": warnings,
            "required_env": required_env,
        }

    def preflight(self, config: Mapping[str, Any], repetitions: int) -> dict[str, Any]:
        report = self.validate(config, repetitions)
        cfg = report["config"]
        limits = {
            "max_live_calls": int((cfg.get("limits") or {}).get("max_live_calls", 0)),
            "max_concurrency": int((cfg.get("limits") or {}).get("max_concurrency", 1)),
            "timeout_ms": int((cfg.get("limits") or {}).get("timeout_ms", 15000)),
        }
        profile_name = cfg.get("environment_profile")
        with self._lock:
            profile = next((p for p in self._profiles.values() if p["name"] == profile_name), None)
        errors: list[str] = []
        missing_capabilities: list[str] = []
        tools: list[dict[str, Any]] = []
        live_tools: list[str] = []
        mock_tools: list[str] = []
        possible_effects: list[dict[str, Any]] = []
        if cfg["mode"] != "mock" and profile is None:
            errors.append(
                "environment_profile is required for live or hybrid runs and must name a profile of this client"
            )
        for tool, binding in (cfg.get("bindings") or {}).items():
            entry: dict[str, Any] = {
                "tool": tool,
                "binding_mode": binding.get("mode", "mock"),
                "required_env": report["required_env"].get(tool, []),
                "missing_env": [],
                "effect": binding.get("effect", "unknown"),
                "errors": [],
            }
            if binding.get("mode", "mock") == "live":
                live_tools.append(tool)
                entry["adapter"] = binding.get("adapter")
                entry["fidelity"] = "live"
                if binding.get("adapter") != "local":
                    cap = f"adapter {binding.get('adapter')} requires Agenomic Cloud"
                    missing_capabilities.append(cap)
                    entry["errors"].append(cap)
                effect = self._effective_effect(binding)
                possible_effects.append(
                    {
                        "tool": tool,
                        "effect": effect,
                        "simulated": False,
                        "per_run_calls_upper_bound": limits["max_live_calls"],
                    }
                )
            else:
                mock_tools.append(tool)
                strategy = binding.get("strategy")
                entry["strategy"] = strategy
                entry["fidelity"] = "contract_only"
                if strategy not in _OFFLINE_STRATEGIES:
                    cap = f"strategy {strategy} requires Agenomic Cloud"
                    missing_capabilities.append(cap)
                    entry["errors"].append(cap)
                elif strategy == "recorded":
                    fixture_set = self._fixture_sets.get(str(binding.get("fixture_set_ref")))
                    if fixture_set is None:
                        entry["errors"].append(
                            f"fixture set {binding.get('fixture_set_ref')} not found"
                        )
                    elif not fixture_set["approved"]:
                        entry["errors"].append(
                            f"fixture set {binding.get('fixture_set_ref')} is not approved"
                        )
                    else:
                        entry["fidelity"] = "recorded_response"
            tools.append(entry)
        plan_errors = errors + [
            f"{t['tool']}: {e}" for t in tools for e in t["errors"] if e not in missing_capabilities
        ]
        has_live = bool(live_tools)
        safety = dict(cfg.get("safety") or {})
        plan = {
            "plan_version": "agenomic.tool_execution_plan/v1",
            "engine": "agenomic-python-local",
            "mode": cfg["mode"],
            "repetitions": repetitions,
            "config_hash": report["config_hash"],
            "tools": tools,
            "live_tools": live_tools,
            "mock_tools": mock_tools,
            "has_live": has_live,
            "projected_live_calls": len(live_tools) * repetitions,
            "limits": limits,
            "possible_effects": possible_effects,
            "required_env": sorted({n for names in report["required_env"].values() for n in names}),
            "missing_env": [],
            "missing_capabilities": missing_capabilities,
            "errors": plan_errors,
            "warnings": list(report["warnings"]),
            "approval_required": has_live and bool(safety.get("require_approved_bindings", True)),
            "profile": {"name": profile["name"], "environment": profile["environment"]}
            if profile
            else None,
            "seed": int((cfg.get("mock_engine") or {}).get("seed", 0)),
            "recording_enabled": bool((cfg.get("recording") or {}).get("enabled", False)),
        }
        return {
            "plan": plan,
            "plan_hash": canonical_hash(plan),
            "runnable": not plan_errors and not missing_capabilities,
        }

    def _effective_effect(self, binding: Mapping[str, Any]) -> str:
        effect = str(binding.get("effect", "unknown"))
        if effect != "unknown":
            return effect
        contract = self._contracts.get(str(binding.get("contract_ref") or ""))
        return str(contract.get("effect", "unknown")) if contract else "unknown"

    # ── runs ──────────────────────────────────────────────────────────

    def create_run(
        self, name: str, config: Mapping[str, Any], repetitions: int, links: Mapping[str, Any]
    ) -> dict[str, Any]:
        preflight = self.preflight(config, repetitions)
        if not preflight["runnable"]:
            raise _invalid(
                "plan is not runnable: "
                + "; ".join(preflight["plan"]["errors"] + preflight["plan"]["missing_capabilities"])
            )
        plan = preflight["plan"]
        run = {
            "id": str(uuid.uuid4()),
            "org_id": "local",
            "name": name,
            "status": "planned" if plan["approval_required"] else "approved",
            "config": dict(self.validate(config, repetitions)["config"]),
            "config_hash": plan["config_hash"],
            "repetitions": repetitions,
            "plan": plan,
            "plan_hash": preflight["plan_hash"],
            "approval": None,
            "live_calls_used": 0,
            "cancel_requested": False,
            "links": dict(links),
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error_message": None,
        }
        with self._lock:
            self._runs[run["id"]] = run
            self._invocations[run["id"]] = []
        return dict(run)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ToolExecutionError("not_found", "tool run not found", 404)
            return dict(run)

    def approve_run(self, run_id: str, plan_hash: str) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            if run["status"] != "planned":
                raise ToolExecutionError("run_not_active", f"run {run_id} is {run['status']}", 400)
            if plan_hash != run["plan_hash"]:
                raise ToolExecutionError(
                    "plan_approval_required", "plan_hash does not match the plan of this run", 400
                )
            run["approval"] = {"plan_hash": plan_hash, "approved_at": _now()}
            run["status"] = "approved"
            return dict(run)

    def start_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            if run["status"] != "approved":
                code = "plan_approval_required" if run["status"] == "planned" else "run_not_active"
                raise ToolExecutionError(code, f"run {run_id} is {run['status']}", 400)
            run["status"] = "running"
            run["started_at"] = _now()
            return dict(run)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            run["cancel_requested"] = True
            if run["status"] in ("planned", "approved", "running"):
                run["status"] = "cancelled"
                run["finished_at"] = _now()
            return dict(run)

    def complete_run(
        self, run_id: str, failed: bool, error_message: Optional[str]
    ) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            if run["status"] != "running":
                raise ToolExecutionError("run_not_active", f"run {run_id} is {run['status']}", 400)
            run["status"] = "failed" if failed else "completed"
            run["error_message"] = error_message
            run["finished_at"] = _now()
            return dict(run)

    def _run(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            raise ToolExecutionError("not_found", "tool run not found", 404)
        return run

    def _active_run(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run["status"] != "running" or run["cancel_requested"]:
            raise ToolExecutionError("run_not_active", f"run {run_id} is {run['status']}", 400)
        return run

    @staticmethod
    def _check_identity(run: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
        repetition = int(identity.get("repetition", 0))
        if repetition < 1 or repetition > int(run["repetitions"]):
            raise _invalid(f"repetition must be between 1 and {run['repetitions']}")
        if not identity.get("logical_call_id") or int(identity.get("attempt", 0)) < 1:
            raise _invalid("logical_call_id and attempt >= 1 are required")

    def _binding(self, run: Mapping[str, Any], tool: str) -> dict[str, Any]:
        binding = (run["config"].get("bindings") or {}).get(tool)
        if binding is None:
            raise ToolExecutionError("tool_unknown", f"tool {tool} has no binding in this run", 400)
        return dict(binding)

    # ── invoke ────────────────────────────────────────────────────────

    def invoke(self, run_id: str, identity: Mapping[str, Any]) -> dict[str, Any]:
        tool = str(identity["tool"])
        arguments = dict(identity.get("arguments") or {})
        with self._lock:
            run = self._active_run(run_id)
            self._check_identity(run, identity)
            try:
                binding = self._binding(run, tool)
                if binding.get("mode", "mock") == "live":
                    raise ToolExecutionError(
                        "live_call_denied",
                        "local mode runs live bindings only through the router's local_functions",
                        400,
                    )
                occurrence = 1 + sum(
                    1
                    for r in self._invocations[run_id]
                    if r["repetition"] == identity["repetition"]
                    and r["tool"] == tool
                    and r["arguments_hash"] == canonical_hash(arguments)
                    and r["attempt"] == 1
                )
                envelope = self.mock_outcome(tool, binding, arguments, occurrence)
            except ToolExecutionError as error:
                self._record(
                    run,
                    identity,
                    "unrouted",
                    "error",
                    "none",
                    error.code,
                    {"error": {"code": error.code, "message": str(error)}},
                )
                raise
            record = self._record(
                run,
                identity,
                str(envelope["agenomic"]["provenance"]["source"]),
                str(envelope["agenomic"]["status"]),
                "none",
                None,
                envelope["result"],
                provenance=envelope["agenomic"]["provenance"],
            )
            envelope["agenomic"]["record_id"] = record["id"]
            return envelope

    def mock_outcome(
        self, tool: str, binding: Mapping[str, Any], arguments: Mapping[str, Any], occurrence: int
    ) -> dict[str, Any]:
        strategy = binding.get("strategy")
        if strategy == "static":
            response, source, extra = dict(binding.get("response") or {}), "static", {}
        elif strategy == "rules":
            response, rule_id = self._select_rule(tool, list(binding.get("rules") or []), arguments)
            source, extra = "static", {"rule_id": rule_id}
        elif strategy == "recorded":
            response, fixture_id = self._select_fixture(
                tool, str(binding.get("fixture_set_ref")), arguments, occurrence
            )
            source, extra = "recorded", {"fixture_id": fixture_id}
        else:
            raise ToolExecutionError(
                "capability_not_implemented", f"strategy {strategy} requires Agenomic Cloud", 400
            )
        return self._envelope(response, source, str(strategy), extra)

    @staticmethod
    def _select_rule(
        tool: str, rules: list[Any], arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        matches = []
        for rule in rules:
            when = dict((rule.get("when") or {}).get("args_match") or {})
            if all(arguments.get(key) == value for key, value in when.items()):
                matches.append(rule)
        if not matches:
            raise ToolExecutionError(
                "mock_unmatched", f"no rule of {tool} matches the arguments", 400
            )
        top = max(int(r.get("priority", 0)) for r in matches)
        best = [r for r in matches if int(r.get("priority", 0)) == top]
        if len(best) > 1:
            raise ToolExecutionError(
                "mock_ambiguous",
                f"rules {[r.get('id') for r in best]} of {tool} match with the same priority",
                400,
            )
        return dict(best[0].get("then") or {}), str(best[0].get("id", ""))

    def _select_fixture(
        self, tool: str, ref: str, arguments: Mapping[str, Any], occurrence: int
    ) -> tuple[dict[str, Any], str]:
        fixture_set = self._fixture_sets.get(ref)
        if fixture_set is None or not fixture_set["approved"]:
            raise ToolExecutionError(
                "mock_unmatched", f"fixture set {ref} is missing or not approved", 400
            )
        wanted = canonical_hash(arguments)
        candidates = [
            f
            for f in fixture_set["fixtures"]
            if f["request"]["tool"] == tool and f["request"]["arguments_hash"] == wanted
        ]
        exact = [f for f in candidates if f["request"].get("occurrence") == occurrence]
        wildcard = [f for f in candidates if f["request"].get("occurrence") is None]
        chosen = exact or wildcard
        if not chosen:
            raise ToolExecutionError(
                "mock_unmatched", f"no fixture of {ref} matches {tool} with these arguments", 400
            )
        if len(chosen) > 1:
            raise ToolExecutionError(
                "mock_ambiguous", f"several fixtures of {ref} match {tool}", 400
            )
        outcome = dict(chosen[0].get("outcome") or {})
        return outcome, str(chosen[0]["fixture_id"])

    @staticmethod
    def _envelope(
        response: Mapping[str, Any], source: str, strategy: str, extra: Mapping[str, Any]
    ) -> dict[str, Any]:
        kind = response.get("kind", "structured")
        status = "success"
        if kind == "structured":
            result: Any = response.get("data")
        elif kind == "text":
            result = {"content": [{"type": "text", "text": response.get("text", "")}]}
        elif kind == "business_error":
            result, status = {"error": dict(response.get("error") or {})}, "error"
        else:
            raise ToolExecutionError(
                "capability_not_implemented", f"response kind {kind} requires Agenomic Cloud", 400
            )
        return {
            "result": result,
            "agenomic": {
                "record_id": "",
                "status": status,
                "provenance": {
                    "source": source,
                    "fidelity": "recorded_response" if source == "recorded" else "contract_only",
                    "binding_mode": "mock",
                    "strategy": strategy,
                    "note": _LOCAL_NOTE,
                    **dict(extra),
                },
                "external_state": "none",
                "effects": [],
                "duration_ms": 0,
                "expected_error": False,
            },
        }

    # ── runtime-local two-phase protocol ──────────────────────────────

    def authorize_local(self, run_id: str, identity: Mapping[str, Any]) -> dict[str, Any]:
        tool = str(identity["tool"])
        with self._lock:
            run = self._active_run(run_id)
            self._check_identity(run, identity)
            binding = self._binding(run, tool)
            if binding.get("mode", "mock") != "live" or binding.get("adapter") != "local":
                return {"decision": "gateway"}
            if run["plan"]["approval_required"] and not run["approval"]:
                raise ToolExecutionError("plan_approval_required", "run has no approval", 400)
            effect = self._effective_effect(binding)
            live_writes = str((run["config"].get("safety") or {}).get("live_writes", "deny"))
            if effect in _WRITE_EFFECTS and live_writes == "deny":
                raise ToolExecutionError(
                    "live_call_denied",
                    f"tool {tool} has effect {effect} and safety.live_writes is deny",
                    400,
                )
            if self._find(run_id, identity) is not None:
                raise ToolExecutionError(
                    "conflict", "invocation already authorized or recorded", 409
                )
            if run["live_calls_used"] >= run["plan"]["limits"]["max_live_calls"]:
                raise ToolExecutionError(
                    "live_budget_exhausted", "the live call budget of this run is exhausted", 400
                )
            run["live_calls_used"] += 1
            record = self._record(
                run,
                identity,
                "runtime_local",
                "pending",
                "indeterminate",
                None,
                {"pending": True},
                effect=effect,
            )
            return {"decision": "local", "record_id": record["id"]}

    def report_local(
        self,
        run_id: str,
        identity: Mapping[str, Any],
        result: Any,
        is_error: bool,
        duration_ms: int,
    ) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            self._check_identity(run, identity)
            record = self._find(run_id, identity)
            if record is None:
                raise ToolExecutionError(
                    "live_call_denied",
                    "runtime-local call was not authorized; call authorize_local before executing",
                    400,
                )
            if record["status"] != "pending":
                raise ToolExecutionError("conflict", "invocation already settled", 409)
            if record["arguments_hash"] != canonical_hash(dict(identity.get("arguments") or {})):
                raise ToolExecutionError(
                    "conflict", "report does not match the authorized arguments", 409
                )
            record["status"] = "error" if is_error else "success"
            record["external_state"] = "confirmed"
            record["duration_ms"] = duration_ms
            record["envelope"] = {"outcome": {"error": result} if is_error else result}
            record["result_hash"] = canonical_hash(record["envelope"])
            return {"record_id": record["id"]}

    def _find(self, run_id: str, identity: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        for record in self._invocations[run_id]:
            if (
                record["repetition"] == identity["repetition"]
                and record["logical_call_id"] == identity["logical_call_id"]
                and record["attempt"] == identity["attempt"]
            ):
                return record
        return None

    def _record(
        self,
        run: dict[str, Any],
        identity: Mapping[str, Any],
        source: str,
        status: str,
        external_state: str,
        error_code: Optional[str],
        payload: Any,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
        effect: str = "unknown",
    ) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "run_id": run["id"],
            "repetition": int(identity["repetition"]),
            "logical_call_id": str(identity["logical_call_id"]),
            "attempt": int(identity["attempt"]),
            "tool": str(identity["tool"]),
            "source": source,
            "status": status,
            "external_state": external_state,
            "effect": effect,
            "arguments_hash": canonical_hash(dict(identity.get("arguments") or {})),
            "result_hash": canonical_hash(payload),
            "error_code": error_code,
            "expected_error": False,
            "provenance": dict(provenance or {"source": source, "note": _LOCAL_NOTE}),
            "started_at": _now(),
            "duration_ms": 0,
            "envelope": {"outcome": payload},
        }
        self._invocations[run["id"]].append(record)
        return record

    # ── reports ───────────────────────────────────────────────────────

    def report(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            records = [dict(r) for r in self._invocations[run_id]]
        by_source: dict[str, int] = {}
        by_status: dict[str, int] = {}
        uncovered, indeterminate = [], []
        unexpected_errors = 0
        for record in records:
            by_source[record["source"]] = by_source.get(record["source"], 0) + 1
            by_status[record["status"]] = by_status.get(record["status"], 0) + 1
            if record["source"] == "unrouted":
                uncovered.append(
                    {
                        "repetition": record["repetition"],
                        "logical_call_id": record["logical_call_id"],
                        "tool": record["tool"],
                        "error_code": record["error_code"],
                    }
                )
            if record["status"] not in ("success", "pending"):
                unexpected_errors += 1
            if record["external_state"] == "indeterminate":
                indeterminate.append(
                    {
                        "repetition": record["repetition"],
                        "logical_call_id": record["logical_call_id"],
                        "tool": record["tool"],
                        "status": record["status"],
                        "requires_reconciliation": True,
                    }
                )
        has_real = any(r["source"] in ("live", "runtime_local") for r in records)
        report = {
            "engine": "agenomic-python-local",
            "run_id": run_id,
            "status": run["status"],
            "calls_by_source": by_source,
            "calls_by_status": by_status,
            "has_real_calls": has_real,
            "uncovered_calls": uncovered,
            "expected_errors": 0,
            "unexpected_errors": unexpected_errors,
            "external_effects": [],
            "indeterminate_results": indeterminate,
            "warning": (
                "this run performed real tool calls: results are not a deterministic reproduction"
                if has_real
                else None
            ),
        }
        return {"run": dict(run), "report": report, "invocations": records}

    def export(self, run_id: str) -> dict[str, Any]:
        full = self.report(run_id)
        return {
            "export_version": "agenomic.tool_run_export/v1",
            "engine": "agenomic-python-local",
            "run": full["run"],
            "report": full["report"],
            "invocations": full["invocations"],
        }
