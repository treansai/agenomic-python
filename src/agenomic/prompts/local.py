from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union

import blake3
import ulid

from agenomic.crypto.signing import SigningKey
from agenomic.exceptions import ApiError
from agenomic.prompts.bundle import BUNDLE_SCHEMA, build_bundle
from agenomic.prompts.digest import (
    MANIFEST_SCHEMA,
    artifact_set_digest,
    manifest_digest,
    sorted_keys,
)
from agenomic.prompts.errors import api_error
from agenomic.prompts.models import ManagedPromptVersion, PromptVersionRecord, ResolvedFrom
from agenomic.prompts.refs import (
    ALIAS_PATTERN,
    PromptAliasRef,
    PromptReference,
    PromptVersionRef,
    is_prompt_id,
    is_uuid,
    parse_execution_ref,
)
from agenomic.prompts.render import content_secrets, validate_content

STATE_SCHEMA = "agenomic.local_prompt_engine/v1"
RELEASE_STATUSES = ("awaiting_approval", "approved", "production", "rejected", "rolled_back")
GOVERNED_STATUSES = ("approved", "production")
UNBINDABLE_STATUSES = ("rejected", "rolled_back")
PROTECTED_TARGETS = ("approved", "production")
UNPROTECTED_TARGETS = ("awaiting_approval", "approved", "production")
MAX_ALIASES = 32
_CHANNEL = re.compile(r"[a-z][a-z0-9-]{0,31}", re.ASCII)
_SLOT = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", re.ASCII)
_RUNTIME_NAMESPACE = uuid.UUID("2f8f3c4e-6d0a-4b9e-9a51-7c2d1e0b4a63")


def _error(code: str, status: int, message: str, **details: Any) -> ApiError:
    return api_error(code, status, message, details)


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _thread_key_reason(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= 256:
        return "grammar"
    if value != value.strip() or any(ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F for c in value):
        return "grammar"
    if value.startswith("exp:"):
        return "reserved_prefix"
    return None


class LocalPromptEngine:
    def __init__(
        self,
        workspace_id: Optional[str] = None,
        *,
        state_path: Optional[os.PathLike[str]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._path = Path(state_path) if state_path is not None else None
        if self._path is not None and self._path.is_file():
            state = json.loads(self._path.read_text(encoding="utf-8"))
            if state.get("schema") != STATE_SCHEMA:
                raise ValueError("unsupported local prompt engine state")
            if workspace_id is not None and workspace_id != state["workspace_id"]:
                raise _error("workspace_mismatch", 0, "the state file belongs to another workspace")
            self._state: dict[str, Any] = state
        else:
            workspace = workspace_id or str(uuid.uuid4())
            if not is_uuid(workspace):
                raise ValueError("workspace_id must be a lowercase uuid")
            self._state = {
                "schema": STATE_SCHEMA,
                "workspace_id": workspace,
                "prompts": {},
                "releases": {},
                "channels": {},
                "bindings": {},
            }

    @property
    def workspace_id(self) -> str:
        return str(self._state["workspace_id"])

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(self._state, ensure_ascii=False, sort_keys=True))
        os.replace(temporary, self._path)

    def _now(self) -> str:
        return _timestamp(datetime.now(timezone.utc))

    def _prompt(self, prompt_id: str) -> dict[str, Any]:
        prompt = self._state["prompts"].get(prompt_id)
        if prompt is None:
            raise _error("prompt_not_found", 404, f"prompt {prompt_id} not found")
        return dict(prompt)

    def _record(self, prompt_id: str, version: int) -> Optional[PromptVersionRecord]:
        prompt = self._state["prompts"].get(prompt_id)
        entry = prompt["versions"].get(str(version)) if prompt is not None else None
        if entry is None:
            return None
        return PromptVersionRecord.model_validate(entry)

    def _fragment_source(
        self, prompt_id: str, version: int, digest: str
    ) -> Optional[dict[str, Any]]:
        record = self._record(prompt_id, version)
        if record is None:
            return None
        return {"prompt_kind": record.prompt_kind, "content": record.content}

    def create_prompt(
        self,
        prompt_id: str,
        *,
        kind: Literal["text", "chat", "fragment"],
        name: str,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not is_prompt_id(prompt_id):
            raise _error("prompt_ref_invalid", 400, "invalid prompt id", reason="invalid_prompt_id")
        if kind not in ("text", "chat", "fragment"):
            raise _error("validation_error", 400, "kind must be text, chat or fragment")
        with self._lock:
            if prompt_id in self._state["prompts"]:
                raise _error("prompt_id_taken", 409, f"prompt {prompt_id} already exists")
            now = self._now()
            self._state["prompts"][prompt_id] = {
                "prompt_id": prompt_id,
                "kind": kind,
                "name": name,
                "description": description,
                "owner": owner,
                "tags": list(tags),
                "status": "active",
                "latest_version": None,
                "metadata_revision": 1,
                "created_at": now,
                "updated_at": now,
                "versions": {},
                "draft": None,
                "aliases": {},
            }
            self._save()
            return self.prompt(prompt_id)

    def prompt(self, prompt_id: str) -> dict[str, Any]:
        prompt = self._prompt(prompt_id)
        return {
            key: copy.deepcopy(value)
            for key, value in prompt.items()
            if key not in ("versions", "draft", "aliases")
        }

    def publish(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        parent_version: Optional[int],
        change_message: str,
        variable_descriptions: Optional[Mapping[str, str]] = None,
    ) -> ManagedPromptVersion:
        with self._lock:
            prompt = self._prompt(prompt_id)
            report = validate_content(
                content, fragments=self._fragment_source, prompt_kind=prompt["kind"]
            )
            report.raise_for_errors(400)
            digest = str(report.content_digest)
            descriptions = dict(variable_descriptions or {})
            if any(name not in report.declared for name in descriptions):
                raise _error(
                    "validation_error", 400, "variable descriptions name undeclared variables"
                )
            for entry in prompt["versions"].values():
                if entry["parent_version"] == parent_version and entry["content_digest"] == digest:
                    return self.get_version(prompt_id, entry["version"])
            latest = prompt["latest_version"]
            if parent_version != latest:
                raise _error(
                    "prompt_version_conflict",
                    409,
                    "parent_version is not the latest version",
                    current=latest,
                )
            if latest is not None and prompt["versions"][str(latest)]["content_digest"] == digest:
                raise _error(
                    "prompt_version_unchanged", 409, "the content equals the parent version"
                )
            number = (latest or 0) + 1
            now = self._now()
            stored = self._state["prompts"][prompt_id]
            stored["versions"][str(number)] = {
                "prompt_id": prompt_id,
                "version": number,
                "prompt_kind": prompt["kind"],
                "content_digest": digest,
                "content": report.content,
                "parent_version": parent_version,
                "change_message": change_message,
                "author": None,
                "created_at": now,
                "variable_descriptions": descriptions,
            }
            stored["latest_version"] = number
            stored["updated_at"] = now
            self._save()
            return self.get_version(prompt_id, number)

    def get_version(self, prompt_id: str, version: int) -> ManagedPromptVersion:
        self._prompt(prompt_id)
        record = self._record(prompt_id, version)
        if record is None:
            raise _error(
                "prompt_version_not_found", 404, f"prompt version {prompt_id}:{version} not found"
            )
        return ManagedPromptVersion.from_record(
            record, workspace_id=self.workspace_id, lookup=self._record
        )

    def get(self, ref: Union[str, PromptReference]) -> ManagedPromptVersion:
        parsed = (
            parse_execution_ref(ref, workspace_id=self.workspace_id)
            if isinstance(ref, str)
            else ref
        )
        if isinstance(parsed, PromptAliasRef):
            alias = self.get_alias(parsed.prompt_id, parsed.alias)
            version = self.get_version(parsed.prompt_id, alias["version"])
            return version.model_copy(
                update={"resolved_from": ResolvedFrom(parsed.alias, alias["generation"])}
            )
        if not isinstance(parsed, PromptVersionRef):
            parsed = parsed.to_version_ref(self.workspace_id)
        return self.get_version(parsed.prompt_id, parsed.version)

    def get_draft(self, prompt_id: str) -> dict[str, Any]:
        draft = self._prompt(prompt_id)["draft"]
        if draft is None:
            raise _error("prompt_draft_not_found", 404, f"prompt {prompt_id} has no draft")
        return copy.deepcopy(dict(draft))

    def save_draft(
        self,
        prompt_id: str,
        content: Mapping[str, Any],
        *,
        base_version: Optional[int],
        expected_revision: int,
    ) -> dict[str, Any]:
        with self._lock:
            prompt = self._prompt(prompt_id)
            current = prompt["draft"]["revision"] if prompt["draft"] is not None else 0
            if expected_revision != current:
                raise _error(
                    "prompt_draft_conflict",
                    409,
                    "the draft revision changed",
                    current=current,
                )
            if base_version is not None and str(base_version) not in prompt["versions"]:
                raise _error("prompt_version_not_found", 404, "base version not found")
            report = validate_content(
                content, fragments=self._fragment_source, prompt_kind=prompt["kind"]
            )
            if report.secret_findings:
                report.raise_for_errors(400)
            now = self._now()
            self._state["prompts"][prompt_id]["draft"] = {
                "prompt_id": prompt_id,
                "revision": current + 1,
                "base_version": base_version,
                "origin": "editor",
                "content": copy.deepcopy(dict(content)),
                "validation": {
                    "ok": report.ok,
                    "errors": [issue.to_dict() for issue in report.errors],
                    "warnings": [issue.to_dict() for issue in report.warnings],
                    "content_digest": report.content_digest,
                },
                "updated_at": now,
            }
            self._save()
            return self.get_draft(prompt_id)

    def get_alias(self, prompt_id: str, alias: str) -> dict[str, Any]:
        found = self._prompt(prompt_id)["aliases"].get(alias)
        if found is None:
            raise _error("prompt_alias_not_found", 404, f"alias {prompt_id}@{alias} not found")
        return dict(found)

    def move_alias(
        self, prompt_id: str, alias: str, *, version: int, expected_generation: int
    ) -> dict[str, Any]:
        if ALIAS_PATTERN.fullmatch(alias) is None:
            raise _error("prompt_ref_invalid", 400, "invalid alias", reason="invalid_alias")
        with self._lock:
            prompt = self._prompt(prompt_id)
            current = prompt["aliases"].get(alias)
            generation = current["generation"] if current is not None else 0
            if expected_generation != generation:
                raise _error(
                    "prompt_alias_conflict", 409, "the alias generation changed", current=generation
                )
            target = prompt["versions"].get(str(version))
            if target is None:
                raise _error("prompt_version_not_found", 404, "alias target not found")
            if current is None and len(prompt["aliases"]) >= MAX_ALIASES:
                raise _error("prompt_alias_limit_reached", 409, "too many aliases")
            record = {
                "prompt_id": prompt_id,
                "alias": alias,
                "version": version,
                "content_digest": target["content_digest"],
                "generation": generation + 1,
                "updated_at": self._now(),
            }
            self._state["prompts"][prompt_id]["aliases"][alias] = record
            self._save()
            return dict(record)

    def _release(self, release_id: str, agent_id: Optional[str] = None) -> dict[str, Any]:
        release = self._state["releases"].get(release_id)
        if release is None or (agent_id is not None and release["agent_id"] != agent_id):
            raise _error("not_found", 404, f"release {release_id} not found")
        return dict(release)

    def get_release(self, release_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._release(release_id))

    def create_release(
        self,
        agent_id: str,
        slots: Mapping[str, str],
        *,
        children: Optional[Mapping[str, str]] = None,
        status: str = "approved",
        name: Optional[str] = None,
    ) -> str:
        if not is_uuid(agent_id):
            raise _error("validation_error", 400, "agent_id must be a lowercase uuid")
        if status not in RELEASE_STATUSES:
            raise _error("validation_error", 400, f"unknown release status {status}")
        with self._lock:
            manifest_slots: dict[str, Any] = {}
            for slot_path in sorted_keys(slots):
                if len(slot_path) > 128 or _SLOT.fullmatch(slot_path) is None:
                    raise _error(
                        "agent_prompt_slot_invalid", 400, "invalid slot path", slot_path=slot_path
                    )
                ref = parse_execution_ref(slots[slot_path], workspace_id=self.workspace_id)
                if isinstance(ref, PromptAliasRef):
                    raise _error("validation_error", 400, "a manifest pins versions, never aliases")
                pinned = (
                    ref
                    if isinstance(ref, PromptVersionRef)
                    else ref.to_version_ref(self.workspace_id)
                )
                version = self.get_version(pinned.prompt_id, pinned.version)
                if version.kind == "fragment":
                    raise _error(
                        "prompt_kind_mismatch",
                        400,
                        "a fragment cannot fill a slot",
                        slot_path=slot_path,
                    )
                manifest_slots[slot_path] = {
                    "prompt_id": pinned.prompt_id,
                    "version": pinned.version,
                    "content_digest": version.content_digest,
                }
            manifest_children: dict[str, Any] = {}
            for child_id, child_release_id in (children or {}).items():
                child = self._release(child_release_id, child_id)
                if child_id == agent_id:
                    raise _error("validation_error", 400, "an agent cannot be its own child")
                manifest_children[child_id] = {
                    "release_id": child_release_id,
                    "genome_version": child["genome_version"],
                }
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "agent_id": agent_id,
                "slots": manifest_slots,
                "children": manifest_children,
            }
            taken = {
                r["name"] for r in self._state["releases"].values() if r["agent_id"] == agent_id
            }
            if name is None:
                number = 1
                while f"av_{number:04d}" in taken:
                    number += 1
                name = f"av_{number:04d}"
            elif name in taken:
                raise _error("candidate_name_taken", 409, f"release name {name} is taken")
            release_id = str(uuid.uuid4())
            runtime = blake3.blake3(b"agenomic.local_runtime/v1\0" + agent_id.encode("ascii"))
            self._state["releases"][release_id] = {
                "release_id": release_id,
                "agent_id": agent_id,
                "name": name,
                "status": status,
                "genome_version": None,
                "manifest": manifest,
                "prompt_manifest_digest": manifest_digest(manifest),
                "bundle_id": str(uuid.uuid5(_RUNTIME_NAMESPACE, agent_id)),
                "bundle_hash": "blake3:" + runtime.hexdigest(),
                "created_at": self._now(),
            }
            self._save()
            return release_id

    def set_release_status(self, release_id: str, status: str) -> None:
        if status not in RELEASE_STATUSES:
            raise _error("validation_error", 400, f"unknown release status {status}")
        with self._lock:
            self._release(release_id)
            self._state["releases"][release_id]["status"] = status
            self._save()

    def get_channel(self, agent_id: str, name: str) -> dict[str, Any]:
        stored = self._state["channels"].get(agent_id, {}).get(name)
        if stored is None:
            if name != "production":
                raise _error("channel_not_found", 404, f"channel {name} not found")
            stored = {"release_id": None, "generation": 0, "protected": True, "history": []}
        return {
            "agent_id": agent_id,
            "name": name,
            "release_id": stored["release_id"],
            "generation": stored["generation"],
            "protected": stored["protected"],
            "materialized": stored["generation"] > 0,
            "history": copy.deepcopy(stored["history"]),
        }

    def move_channel(
        self, agent_id: str, name: str, release_id: str, *, expected_generation: int
    ) -> dict[str, Any]:
        if _CHANNEL.fullmatch(name) is None:
            raise _error("channel_name_invalid", 400, "invalid channel name")
        with self._lock:
            release = self._release(release_id, agent_id)
            agent_channels = self._state["channels"].setdefault(agent_id, {})
            current = agent_channels.get(name) or {
                "release_id": None,
                "generation": 0,
                "protected": name == "production",
                "history": [],
            }
            if expected_generation != current["generation"]:
                raise _error(
                    "channel_conflict",
                    409,
                    "the channel generation changed",
                    current=current["generation"],
                )
            allowed = PROTECTED_TARGETS if current["protected"] else UNPROTECTED_TARGETS
            if release["status"] not in allowed:
                raise _error(
                    "release_not_promotable",
                    409,
                    f"a {release['status']} release cannot be a target of {name}",
                )
            if name == "production":
                for other in self._state["releases"].values():
                    if other["agent_id"] == agent_id and other["status"] == "production":
                        other["status"] = "approved"
                self._state["releases"][release_id]["status"] = "production"
            generation = current["generation"] + 1
            current["history"].append(
                {
                    "generation": generation,
                    "action": "promote",
                    "from_release_id": current["release_id"],
                    "to_release_id": release_id,
                    "created_at": self._now(),
                }
            )
            current["release_id"] = release_id
            current["generation"] = generation
            agent_channels[name] = current
            self._save()
            return self.get_channel(agent_id, name)

    def _children(self, release: Mapping[str, Any]) -> dict[str, Any]:
        children: dict[str, Any] = {}
        pending = list(release["manifest"]["children"].items())
        while pending:
            child_id, pin = pending.pop()
            if child_id in children:
                continue
            child = self._release(pin["release_id"], child_id)
            children[child_id] = child
            pending.extend(child["manifest"]["children"].items())
        return children

    def _prompts(self, manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        prompts: dict[str, Any] = {}
        pending = [pin for manifest in manifests for pin in manifest["slots"].values()]
        while pending:
            pin = pending.pop()
            key = f"{pin['prompt_id']}:{pin['version']}"
            if key in prompts:
                continue
            record = self._record(pin["prompt_id"], pin["version"])
            if record is None:
                raise _error("artifact_integrity_error", 500, f"pinned prompt {key} is missing")
            prompts[key] = {
                "prompt_id": record.prompt_id,
                "version": record.version,
                "prompt_kind": record.prompt_kind,
                "content_digest": record.content_digest,
                "content": copy.deepcopy(record.content),
            }
            pending.extend(record.content["fragments"].values())
        return prompts

    @staticmethod
    def _require_selector(channel: Optional[str], release_id: Optional[str]) -> None:
        if (channel is None) == (release_id is None):
            raise _error(
                "agent_selector_required", 400, "name exactly one of channel and release_id"
            )

    def _target(
        self, agent_id: str, channel: Optional[str], release_id: Optional[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_selector(channel, release_id)
        if channel is not None:
            state = self.get_channel(agent_id, channel)
            if state["release_id"] is None:
                raise _error("channel_unassigned", 409, f"channel {channel} has no target")
            return self._release(state["release_id"], agent_id), {
                "channel": channel,
                "generation": state["generation"],
            }
        return self._release(str(release_id), agent_id), {"release_id": release_id}

    def _bindable(self, release: Mapping[str, Any], children: Mapping[str, Any]) -> None:
        if release["status"] in UNBINDABLE_STATUSES:
            raise _error("release_not_bindable", 409, f"the release is {release['status']}")
        for child_id in sorted_keys(children):
            if children[child_id]["status"] in UNBINDABLE_STATUSES:
                raise _error(
                    "release_not_bindable",
                    409,
                    "a pinned child release is not bindable",
                    child_agent_id=child_id,
                )

    def _artifacts(
        self, release: Mapping[str, Any], resolved_from: Mapping[str, Any]
    ) -> dict[str, Any]:
        children = self._children(release)
        manifests = [release["manifest"], *(child["manifest"] for child in children.values())]
        source: dict[str, Any] = (
            {"channel": resolved_from["channel"], "channel_generation": resolved_from["generation"]}
            if "channel" in resolved_from
            else {"release_id": resolved_from["release_id"]}
        )
        document: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "workspace_id": self.workspace_id,
            "agent_id": release["agent_id"],
            "source": source,
            "release": {
                "release_id": release["release_id"],
                "release_name": release["name"],
                "genome_version": release["genome_version"],
                "bundle_id": release["bundle_id"],
                "bundle_hash": release["bundle_hash"],
                "legacy": False,
            },
            "prompt_manifest_digest": release["prompt_manifest_digest"],
            "manifest": copy.deepcopy(release["manifest"]),
            "children": {
                child_id: {
                    "release_id": child["release_id"],
                    "genome_version": child["genome_version"],
                    "prompt_manifest_digest": child["prompt_manifest_digest"],
                    "manifest": copy.deepcopy(child["manifest"]),
                }
                for child_id, child in children.items()
            },
            "prompts": self._prompts(manifests),
            "exported_at": self._now(),
            "expires_at": None,
        }
        document["prompt_bundle_digest"] = artifact_set_digest(document)
        return document

    def resolve(
        self, agent_id: str, *, channel: Optional[str] = None, release_id: Optional[str] = None
    ) -> dict[str, Any]:
        release, resolved_from = self._target(agent_id, channel, release_id)
        self._bindable(release, self._children(release))
        return self._artifacts(release, resolved_from)

    def create_binding(
        self,
        agent_id: str,
        *,
        thread_key: str,
        scope: Literal["thread", "execution"],
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        expect_manifest_digest: Optional[str] = None,
        runtime_client: Optional[Mapping[str, Optional[str]]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        reason = _thread_key_reason(thread_key)
        if reason is not None:
            raise _error("thread_key_invalid", 400, "invalid thread key", reason=reason)
        if scope not in ("thread", "execution"):
            raise _error("validation_error", 400, "scope must be thread or execution")
        self._require_selector(channel, release_id)
        with self._lock:
            existing = self._state["bindings"].get(agent_id, {}).get(thread_key)
            if existing is not None:
                selector = (
                    {"channel": channel} if channel is not None else {"release_id": release_id}
                )
                bound = existing["resolved_from"]
                same = (
                    existing["scope"] == scope
                    and all(bound.get(key) == value for key, value in selector.items())
                    and expect_manifest_digest in (None, existing["prompt_manifest_digest"])
                )
                if not same:
                    raise _error(
                        "execution_binding_conflict",
                        409,
                        "the thread is already bound to another selector, scope or manifest",
                        binding_id=existing["binding_id"],
                        release_id=existing["release_id"],
                        resolved_from=copy.deepcopy(bound),
                        prompt_manifest_digest=existing["prompt_manifest_digest"],
                    )
                return copy.deepcopy(existing), self._binding_artifacts(existing), False
            release, resolved_from = self._target(agent_id, channel, release_id)
            children = self._children(release)
            self._bindable(release, children)
            if expect_manifest_digest not in (None, release["prompt_manifest_digest"]):
                raise _error(
                    "execution_binding_conflict",
                    409,
                    "the release pins another prompt manifest",
                )
            binding = {
                "schema": "agenomic.execution_binding/v1",
                "binding_id": "bnd_" + ulid.new().str.lower(),
                "workspace_id": self.workspace_id,
                "agent_id": agent_id,
                "thread_key": thread_key,
                "scope": scope,
                "release_id": release["release_id"],
                "release_name": release["name"],
                "genome_version": release["genome_version"],
                "prompt_manifest_digest": release["prompt_manifest_digest"],
                "runtime": {
                    "bundle_id": release["bundle_id"],
                    "bundle_hash": release["bundle_hash"],
                },
                "resolved_from": resolved_from,
                "children": {
                    child_id: {
                        "release_id": child["release_id"],
                        "genome_version": child["genome_version"],
                        "prompt_manifest_digest": child["prompt_manifest_digest"],
                        "source": "manifest",
                        "channel": None,
                        "generation": None,
                    }
                    for child_id, child in children.items()
                },
                "parent_binding_id": None,
                "experiment": None,
                "runtime_client": dict(
                    runtime_client
                    or {"sdk": None, "sdk_version": None, "adapter": None, "adapter_version": None}
                ),
                "created_at": self._now(),
                "created_by": {"user_id": None, "api_key_id": None},
            }
            self._state["bindings"].setdefault(agent_id, {})[thread_key] = binding
            self._save()
            return copy.deepcopy(binding), self._artifacts(release, resolved_from), True

    def _binding_artifacts(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        return self._artifacts(self._release(binding["release_id"]), binding["resolved_from"])

    def get_binding(self, agent_id: str, binding_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        for binding in self._state["bindings"].get(agent_id, {}).values():
            if binding["binding_id"] == binding_id:
                return copy.deepcopy(binding), self._binding_artifacts(binding)
        raise _error("execution_binding_not_found", 404, f"binding {binding_id} not found")

    def export_bundle(
        self,
        agent_id: str,
        *,
        signer: SigningKey,
        channel: Optional[str] = None,
        release_id: Optional[str] = None,
        expires_in_days: int = 30,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        if isinstance(expires_in_days, bool) or not 1 <= expires_in_days <= 365:
            raise _error("validation_error", 400, "expires_in_days must be between 1 and 365")
        release, resolved_from = self._target(agent_id, channel, release_id)
        children = self._children(release)
        self._bindable(release, children)
        document = self._artifacts(release, resolved_from)
        for ref in sorted_keys(document["prompts"]):
            findings = content_secrets(document["prompts"][ref]["content"])
            if findings:
                raise _error(
                    "prompt_bundle_export_blocked",
                    409,
                    "a pinned prompt matches a secret pattern",
                    ref=ref,
                    pattern=findings[0].pattern,
                )
        moment = now or datetime.now(timezone.utc)
        protected = False
        if channel is not None:
            protected = bool(self.get_channel(agent_id, channel)["protected"])
        statuses = [release["status"], *(child["status"] for child in children.values())]
        document["governance"] = {
            "release_status": release["status"],
            "channel": channel,
            "channel_protected": protected,
            "approved": all(status in GOVERNED_STATUSES for status in statuses),
        }
        document["exported_at"] = _timestamp(moment)
        document["expires_at"] = _timestamp(moment + timedelta(days=expires_in_days))
        return build_bundle(document, signer=signer)
