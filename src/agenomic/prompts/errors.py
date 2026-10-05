from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from agenomic.exceptions import ApiError


class AjsError(ValueError):
    def __init__(self, reason: str, value_path: str) -> None:
        super().__init__(f"{reason} at {value_path or '/'}")
        self.reason = reason
        self.value_path = value_path


class PromptRefError(ApiError):
    pass


class PromptTemplateError(ApiError):
    pass


class PromptRenderError(ApiError):
    pass


class PromptIntegrityError(ApiError):
    pass


class PromptImportError(ApiError):
    pass


class PromptBindingError(ApiError):
    pass


class RegistryUnavailableError(ApiError):
    pass


class PromptConflictError(ApiError):
    pass


_CODE_CLASSES: dict[str, type[ApiError]] = {
    "prompt_ref_invalid": PromptRefError,
    "prompt_ref_unversioned": PromptRefError,
    "prompt_ref_cross_workspace": PromptRefError,
    "workspace_mismatch": PromptRefError,
    "alias_in_managed_run": PromptRefError,
    "prompt_template_invalid": PromptTemplateError,
    "prompt_secret_detected": PromptTemplateError,
    "prompt_content_too_large": PromptTemplateError,
    "prompt_kind_mismatch": PromptTemplateError,
    "prompt_fragment_cycle": PromptTemplateError,
    "prompt_fragment_depth_exceeded": PromptTemplateError,
    "prompt_render_error": PromptRenderError,
    "prompt_digest_mismatch": PromptIntegrityError,
    "manifest_digest_mismatch": PromptIntegrityError,
    "bundle_signature_invalid": PromptIntegrityError,
    "bundle_untrusted_key": PromptIntegrityError,
    "bundle_incomplete": PromptIntegrityError,
    "bundle_expired": PromptIntegrityError,
    "bundle_scope_mismatch": PromptIntegrityError,
    "bundle_ungoverned": PromptIntegrityError,
    "cache_conflict": PromptIntegrityError,
    "artifact_integrity_error": PromptIntegrityError,
    "prompt_import_invalid": PromptImportError,
    "prompt_import_plan_stale": PromptImportError,
    "prompt_import_expired": PromptImportError,
    "prompt_import_already_applied": PromptImportError,
    "prompt_import_item_blocked": PromptImportError,
    "yaml_support_not_installed": PromptImportError,
    "binding_missing": PromptBindingError,
    "binding_mismatch": PromptBindingError,
    "binding_target_mismatch": PromptBindingError,
    "binding_checkpoint_mismatch": PromptBindingError,
    "child_agent_not_pinned": PromptBindingError,
    "slot_not_in_manifest": PromptBindingError,
    "thread_id_required": PromptBindingError,
    "execution_key_required": PromptBindingError,
    "execution_binding_unrecoverable": PromptBindingError,
    "agenomic_reserved_key": PromptBindingError,
    "prompt_set_unavailable": PromptBindingError,
    "prompt_set_not_serializable": PromptBindingError,
    "factory_topology_mismatch": PromptBindingError,
    "nested_bind_unsupported": PromptBindingError,
    "privileged_credential": PromptBindingError,
    "binding_store_corrupt": PromptBindingError,
    "execution_binding_conflict": PromptBindingError,
    "child_agent_conflict": PromptBindingError,
    "release_not_bindable": PromptBindingError,
    "session_required": PromptBindingError,
    "registry_unavailable": RegistryUnavailableError,
    "prompt_draft_conflict": PromptConflictError,
    "prompt_version_conflict": PromptConflictError,
    "prompt_alias_conflict": PromptConflictError,
    "prompt_metadata_conflict": PromptConflictError,
    "agent_prompt_slots_conflict": PromptConflictError,
    "channel_conflict": PromptConflictError,
}


def error_class_for(code: str) -> type[ApiError]:
    return _CODE_CLASSES.get(code, ApiError)


def api_error(
    code: str,
    status: int,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> ApiError:
    return error_class_for(code)(code, status, message, details)


def item_details(item: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {"reason": item["code"], "errors": [dict(item)]}
    for key, value in item.items():
        if key != "code":
            details[key] = value
    return details


def render_error(item: Mapping[str, Any]) -> PromptRenderError:
    return PromptRenderError(
        "prompt_render_error",
        0,
        f"render failed: {item['code']}",
        item_details(item),
    )


def integrity_error(code: str, message: str, **details: Any) -> PromptIntegrityError:
    return PromptIntegrityError(code, 0, message, details)


def binding_error(code: str, message: str, **details: Any) -> PromptBindingError:
    return PromptBindingError(code, 0, message, details)
