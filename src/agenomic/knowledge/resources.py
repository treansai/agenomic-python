from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union
from urllib.parse import quote

from pydantic import BaseModel, ValidationError

from agenomic._transport import ApiResponse, segment
from agenomic.exceptions import ApiError
from agenomic.knowledge.models import (
    UNSET,
    AgentBindingResult,
    AgentKnowledge,
    AgentSearchResponse,
    AnswerResponse,
    BacklinkList,
    DocumentDetail,
    DocumentList,
    DocumentText,
    DocumentWrite,
    KnowledgeBaseDetail,
    KnowledgeBaseRecord,
    KnowledgeCollection,
    KnowledgeDiff,
    KnowledgeDocument,
    KnowledgeJob,
    KnowledgeSection,
    KnowledgeVersion,
    Publication,
    QueryResponse,
    RetrievalEvent,
    RetrievalSnapshot,
    RevisionList,
    SearchResponse,
    SectionDetail,
    SectionTree,
    Unset,
    VersionDecision,
    VersionDetail,
    VersionLike,
    VersionVerification,
    VersionWrite,
    normalize_version,
    version_number,
)
from agenomic.prompts.resources import Call, Flow, Page, arun_flow, run_flow

if TYPE_CHECKING:
    from agenomic._client import Client

__all__ = [
    "AgentKnowledgeResource",
    "KnowledgeBase",
    "KnowledgeCollectionsResource",
    "KnowledgeDocumentsResource",
    "KnowledgeJobsResource",
    "KnowledgeResource",
    "KnowledgeRetrievalsResource",
    "KnowledgeVersionsResource",
]

_M = TypeVar("_M", bound=BaseModel)
_R = TypeVar("_R")

SearchMode = Literal["keyword", "semantic", "hybrid", "section", "exact"]
Expand = Literal["none", "parent", "section"]
UploadSource = Union[bytes, bytearray, memoryview, str, "os.PathLike[str]"]

_KB_ID = re.compile(r"kb_[a-z0-9]+(?:[_-][a-z0-9]+)*", re.ASCII)
_SECTION_ID = re.compile(r"sec_[0-9a-f]{16}", re.ASCII)
_HEADER_TEXT = re.compile(r"[\x20-\x7e]*", re.ASCII)
_MODES = frozenset({"keyword", "semantic", "hybrid", "section", "exact"})
_EXPANDS = frozenset({"none", "parent", "section"})
_SECTION_INCLUDES = frozenset({"children", "descendants"})
_FILTER_LISTS = ("collections", "document_ids", "tags")
_FILTER_TEXTS = ("path_prefix", "classification_max")
_EXECUTION_KEYS = frozenset(
    {"binding_id", "execution_id", "run_id", "trace_id", "session_id", "parent_span_id"}
)
_OPERATIONS = frozenset(
    {"get_section", "list_children", "sections_tagged", "search", "get_document"}
)
_OCTET_STREAM = "application/octet-stream"

_sleep = time.sleep
_asleep = asyncio.sleep
_monotonic = time.monotonic


def _cloud_required(operation: str) -> ApiError:
    return ApiError(
        "cloud_required",
        0,
        f"{operation} needs Agenomic Cloud; knowledge bases are served by the gateway only",
    )


def _require_cloud(client: Client, operation: str) -> None:
    if not client.is_cloud:
        raise _cloud_required(operation)


def _invalid(response: ApiResponse, what: str) -> ApiError:
    return ApiError("invalid_response", response.status, f"the response carries no valid {what}")


def _model(model: type[_M], value: Any, response: ApiResponse, what: str) -> _M:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise _invalid(response, what) from error


def _wrapped(model: type[_M], response: ApiResponse, key: str) -> _M:
    if not isinstance(response.body.get(key), dict):
        raise _invalid(response, key)
    return _model(model, response.body[key], response, key)


def _bare(model: type[_M], response: ApiResponse, what: str) -> _M:
    return _model(model, response.body, response, what)


def _page(model: type[_M], response: ApiResponse, key: str) -> Page[_M]:
    items = response.body.get(key)
    if not isinstance(items, list):
        raise _invalid(response, key)
    cursor = response.body.get("next_cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise _invalid(response, "next_cursor")
    return Page([_model(model, item, response, key) for item in items], cursor)


def _kb_id(kb_id: str) -> str:
    if not isinstance(kb_id, str) or len(kb_id) > 64 or not _KB_ID.fullmatch(kb_id):
        raise ValueError("knowledge_base must match kb_[a-z0-9]+(?:[_-][a-z0-9]+)* (64 chars max)")
    return kb_id


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _kb_path(kb_id: str, *rest: str) -> str:
    return "/".join(("/v1/knowledge-bases", segment(_kb_id(kb_id)), *rest))


def _document_path(kb_id: str, document_id: str, *rest: str) -> str:
    return _kb_path(kb_id, "documents", segment(_id(document_id, "document_id")), *rest)


def _agent_path(agent_id: str, *rest: str) -> str:
    return "/".join(("/v1/agents", segment(_id(agent_id, "agent_id")), "knowledge", *rest))


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _ranged(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strings(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must be a list of strings")
    return list(values)


def _labels(values: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(values, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in values.items()
    ):
        raise ValueError(f"{name} must map strings to strings")
    return dict(values)


def _page_params(cursor: Optional[str], limit: Optional[int]) -> dict[str, str]:
    params: dict[str, str] = {}
    if limit is not None:
        params["limit"] = str(_ranged(limit, "limit", 1, 200))
    if cursor is not None:
        params["cursor"] = _id(cursor, "cursor")
    return params


def _set(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _filters(filters: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if filters is None:
        return None
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a mapping")
    unknown = set(filters) - {*_FILTER_LISTS, *_FILTER_TEXTS, "metadata"}
    if unknown:
        raise ValueError(f"unknown filters {sorted(unknown)}")
    body: dict[str, Any] = {}
    for key in _FILTER_LISTS:
        if filters.get(key) is not None:
            body[key] = _strings(filters[key], f"filters.{key}")
    for key in _FILTER_TEXTS:
        if filters.get(key) is not None:
            body[key] = _text(filters[key], f"filters.{key}")
    if filters.get("metadata") is not None:
        body["metadata"] = _labels(filters["metadata"], "filters.metadata")
    return body or None


def _execution(execution: Optional[Mapping[str, str]]) -> Optional[dict[str, str]]:
    if execution is None:
        return None
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping")
    unknown = set(execution) - _EXECUTION_KEYS
    if unknown:
        raise ValueError(f"unknown execution members {sorted(unknown)}")
    body = {
        key: _text(value, f"execution.{key}")
        for key, value in execution.items()
        if value is not None
    }
    return body or None


def _mode(mode: Optional[str]) -> Optional[str]:
    if mode is not None and mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    return mode


def _version_body(version: Optional[VersionLike]) -> Union[int, str, None]:
    return None if version is None else normalize_version(version)


def _version_param(version: Optional[VersionLike]) -> Optional[str]:
    return None if version is None else str(normalize_version(version))


def _search_body(
    query: str,
    *,
    version: Optional[VersionLike] = None,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    filters: Optional[Mapping[str, Any]] = None,
    rerank: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
    include_context: bool = False,
    expand: Optional[str] = None,
    debug: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": _text(query, "query")}
    _set(body, "version", _version_body(version))
    _set(body, "mode", _mode(mode))
    _set(body, "top_k", None if top_k is None else _ranged(top_k, "top_k", 1, 50))
    _set(body, "filters", _filters(filters))
    _set(body, "rerank", None if rerank is None else _text(rerank, "rerank"))
    if max_context_tokens is not None:
        body["max_context_tokens"] = _positive(max_context_tokens, "max_context_tokens")
    if expand is not None:
        if expand not in _EXPANDS:
            raise ValueError(f"expand must be one of {sorted(_EXPANDS)}")
        body["expand"] = expand
    if include_context:
        body["include_context"] = True
    if debug:
        body["debug"] = True
    return body


def _header(value: str, name: str) -> str:
    if not isinstance(value, str) or not _HEADER_TEXT.fullmatch(value):
        raise ValueError(f"{name} must be printable ASCII text")
    return value


def _encoded(value: str) -> str:
    return quote(value, safe="")


def _upload_headers(
    path: Optional[str],
    collection: Optional[str],
    tags: Optional[Sequence[str]],
    classification: Optional[str],
    change_message: Optional[str],
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if path is not None:
        headers["x-agenomic-document-path"] = quote(_text(path, "path"), safe="/")
    if collection is not None:
        headers["x-agenomic-collection"] = _encoded(_text(collection, "collection"))
    if tags is not None:
        values = _strings(tags, "tags")
        if any("," in tag or not tag.strip() for tag in values):
            raise ValueError("tags must be non-empty and hold no comma")
        headers["x-agenomic-tags"] = ",".join(_encoded(tag) for tag in values)
    if classification is not None:
        headers["x-agenomic-classification"] = _encoded(_text(classification, "classification"))
    if change_message is not None:
        if not isinstance(change_message, str):
            raise ValueError("change_message must be a string")
        headers["x-agenomic-change-message"] = _encoded(change_message)
    return headers


def _read_file(file: Path) -> bytes:
    return file.read_bytes()


def _source(source: UploadSource, path: Optional[str]) -> Flow[tuple[bytes, Optional[str]]]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), path
    if isinstance(source, (str, os.PathLike)):
        file = Path(source)
        data = yield partial(_read_file, file)
        if not isinstance(data, bytes):
            raise TypeError("the upload source did not read as bytes")
        return data, path if path is not None else file.name
    raise TypeError("source must be bytes or a file path")


def _content_type(content_type: Optional[str]) -> str:
    if content_type is None:
        return _OCTET_STREAM
    return _header(_text(content_type, "content_type"), "content_type")


def _agent_knowledge(response: ApiResponse) -> AgentKnowledge:
    if isinstance(response.body.get("knowledge"), dict):
        return _wrapped(AgentKnowledge, response, "knowledge")
    return _bare(AgentKnowledge, response, "agent knowledge")


def _binding_input(
    knowledge_base: str,
    version: VersionLike,
    access: Optional[str],
    collections: Optional[Sequence[str]],
    max_classification: Optional[str],
    enabled: Optional[bool],
) -> dict[str, Any]:
    selector = normalize_version(version)
    if selector == "draft":
        raise ValueError("agent bindings name a version number or 'published', never 'draft'")
    binding: dict[str, Any] = {"knowledge_base": _kb_id(knowledge_base), "version": selector}
    _set(binding, "access", None if access is None else _text(access, "access"))
    if collections is not None:
        binding["collections"] = _strings(collections, "collections")
    _set(
        binding,
        "max_classification",
        None if max_classification is None else _text(max_classification, "max_classification"),
    )
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        binding["enabled"] = enabled
    return binding


def _bindings(bindings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(bindings, (str, bytes, Mapping)):
        raise ValueError("bindings must be a list of mappings")
    result: list[dict[str, Any]] = []
    for item in bindings:
        if not isinstance(item, Mapping):
            raise ValueError("bindings must be a list of mappings")
        unknown = set(item) - {
            "knowledge_base",
            "version",
            "access",
            "collections",
            "max_classification",
            "enabled",
        }
        if unknown or "knowledge_base" not in item or "version" not in item:
            raise ValueError("each binding names knowledge_base and version and nothing unknown")
        result.append(
            _binding_input(
                item["knowledge_base"],
                item["version"],
                item.get("access"),
                item.get("collections"),
                item.get("max_classification"),
                item.get("enabled"),
            )
        )
    return result


def _tri_state(body: dict[str, Any], key: str, value: Union[str, None, Unset]) -> None:
    if value is UNSET:
        return
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string or None")
    body[key] = value


def _section_match(response: QueryResponse, document_id: Optional[str]) -> KnowledgeSection:
    sections = {section.section_id: section for section in response.sections}
    for match in response.matches:
        if document_id is not None and match.document_id != document_id:
            continue
        section = sections.get(match.section_id)
        if section is not None:
            return section
    raise ApiError(
        "knowledge_section_not_found",
        0,
        "no section of the requested document matches that heading",
        {"document_id": document_id} if document_id is not None else {},
    )


class _Base:
    def __init__(self, client: Client) -> None:
        self._client = client

    def _run(self, flow: Flow[_R]) -> _R:
        return run_flow(self._client, flow)

    async def _arun(self, flow: Flow[_R]) -> _R:
        return await arun_flow(self._client, flow)


def _get_base_flow(client: Client, kb_id: str) -> Flow[KnowledgeBaseDetail]:
    _require_cloud(client, "knowledge.get")
    response = yield Call("GET", _kb_path(kb_id), retry=True)
    detail = _bare(KnowledgeBaseDetail, response, "knowledge base")
    if detail.knowledge_base.kb_id != kb_id:
        raise _invalid(response, "knowledge base for the requested kb_id")
    return detail


def _search_flow(client: Client, kb_id: str, body: Mapping[str, Any]) -> Flow[SearchResponse]:
    _require_cloud(client, "knowledge.search")
    response = yield Call("POST", _kb_path(kb_id, "search"), dict(body), retry=True)
    return _bare(SearchResponse, response, "search response")


def _query_body(
    query: Optional[str],
    operation: Optional[Mapping[str, Any]],
    version: Optional[VersionLike],
) -> dict[str, Any]:
    if (query is None) == (operation is None):
        raise ValueError("pass exactly one of query (text form) and operation")
    body: dict[str, Any] = {}
    if query is not None:
        body["query"] = _text(query, "query")
    else:
        if not isinstance(operation, Mapping) or operation.get("op") not in _OPERATIONS:
            raise ValueError(
                f"operation must be a mapping whose op is one of {sorted(_OPERATIONS)}"
            )
        body["operation"] = dict(operation)
    _set(body, "version", _version_body(version))
    return body


def _query_flow(client: Client, kb_id: str, body: Mapping[str, Any]) -> Flow[QueryResponse]:
    _require_cloud(client, "knowledge.query")
    response = yield Call("POST", _kb_path(kb_id, "query"), dict(body), retry=True)
    return _bare(QueryResponse, response, "query response")


def _answer_flow(client: Client, kb_id: str, body: Mapping[str, Any]) -> Flow[AnswerResponse]:
    _require_cloud(client, "knowledge.answer")
    response = yield Call("POST", _kb_path(kb_id, "answer"), dict(body), retry=True)
    return _bare(AnswerResponse, response, "answer response")


def _answer_body(
    query: str,
    version: Optional[VersionLike],
    top_k: Optional[int],
    filters: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": _text(query, "query")}
    _set(body, "version", _version_body(version))
    _set(body, "top_k", None if top_k is None else _ranged(top_k, "top_k", 1, 50))
    _set(body, "filters", _filters(filters))
    return body


def _document_flow(client: Client, kb_id: str, document_id: str) -> Flow[DocumentDetail]:
    _require_cloud(client, "knowledge.documents.get")
    response = yield Call("GET", _document_path(kb_id, document_id), retry=True)
    detail = _bare(DocumentDetail, response, "document")
    if detail.document.document_id != document_id:
        raise _invalid(response, "document for the requested document_id")
    return detail


def _section_flow(
    client: Client,
    kb_id: str,
    document_id: str,
    section_id: str,
    revision: Optional[int],
    version: Optional[VersionLike],
    include: Optional[Sequence[str]],
) -> Flow[SectionDetail]:
    _require_cloud(client, "knowledge.documents.section")
    params: dict[str, str] = {}
    if revision is not None and version is not None:
        raise ValueError("pass at most one of revision and version")
    if revision is not None:
        params["revision"] = str(_positive(revision, "revision"))
    _set(params, "version", _version_param(version))
    if include is not None:
        values = _strings(include, "include")
        if set(values) - _SECTION_INCLUDES:
            raise ValueError(f"include takes {sorted(_SECTION_INCLUDES)}")
        if values:
            params["include"] = ",".join(values)
    path = _document_path(kb_id, document_id, "sections", segment(_id(section_id, "section_id")))
    response = yield Call("GET", path, params=params or None, retry=True)
    detail = _bare(SectionDetail, response, "section")
    if detail.section.section_id != section_id or detail.document_id != document_id:
        raise _invalid(response, "section for the requested section_id")
    return detail


def _versions_flow(
    client: Client, kb_id: str, cursor: Optional[str], limit: Optional[int]
) -> Flow[Page[KnowledgeVersion]]:
    _require_cloud(client, "knowledge.versions.list")
    params = _page_params(cursor, limit)
    response = yield Call("GET", _kb_path(kb_id, "versions"), params=params or None, retry=True)
    return _page(KnowledgeVersion, response, "versions")


def _publish_flow(
    client: Client, kb_id: str, version: VersionLike, if_match: int, reason: Optional[str]
) -> Flow[Publication]:
    _require_cloud(client, "knowledge.publish")
    body: dict[str, Any] = {"version": version_number(version)}
    _set(body, "reason", None if reason is None else _text(reason, "reason"))
    response = yield Call(
        "POST",
        _kb_path(kb_id, "publish"),
        body,
        if_match=_non_negative(if_match, "if_match"),
    )
    return _bare(Publication, response, "publication")


def _rollback_flow(
    client: Client,
    kb_id: str,
    reason: str,
    to_version: Optional[VersionLike],
    if_match: int,
) -> Flow[Publication]:
    _require_cloud(client, "knowledge.rollback")
    body: dict[str, Any] = {"reason": _text(reason, "reason")}
    _set(body, "to_version", None if to_version is None else version_number(to_version))
    response = yield Call(
        "POST",
        _kb_path(kb_id, "rollback"),
        body,
        if_match=_non_negative(if_match, "if_match"),
    )
    return _bare(Publication, response, "publication")


def _get_section_flow(
    client: Client,
    kb_id: str,
    section: str,
    document_id: Optional[str],
    document: Optional[str],
    version: Optional[VersionLike],
) -> Flow[KnowledgeSection]:
    _require_cloud(client, "knowledge.get_section")
    _text(section, "section")
    if document_id is None and document is None:
        raise ValueError("pass document_id or document")
    if _SECTION_ID.fullmatch(section):
        if document_id is None:
            raise ValueError("a section id needs the document_id it belongs to")
        detail = yield from _section_flow(client, kb_id, document_id, section, None, version, None)
        return detail.section
    name = document if document is not None else _id(str(document_id), "document_id")
    body = _query_body(
        None,
        {"op": "get_section", "document": _text(name, "document"), "section": section},
        version,
    )
    answer = yield from _query_flow(client, kb_id, body)
    return _section_match(answer, document_id)


class KnowledgeBase:
    def __init__(
        self, client: Client, kb_id: str, detail: Optional[KnowledgeBaseDetail] = None
    ) -> None:
        self._client = client
        self.kb_id = _kb_id(kb_id)
        self.detail = detail

    def __repr__(self) -> str:
        return f"KnowledgeBase({self.kb_id!r})"

    @property
    def record(self) -> Optional[KnowledgeBaseRecord]:
        return None if self.detail is None else self.detail.knowledge_base

    @property
    def published_version(self) -> Optional[int]:
        record = self.record
        return None if record is None else record.published_version

    def _search(
        self,
        query: str,
        version: Optional[VersionLike],
        top_k: Optional[int],
        mode: Optional[str],
        filters: Optional[Mapping[str, Any]],
        include_context: bool,
        options: Mapping[str, Any],
    ) -> Flow[SearchResponse]:
        body = _search_body(
            query,
            version=version,
            top_k=top_k,
            mode=mode,
            filters=filters,
            include_context=include_context,
            **options,
        )
        return (yield from _search_flow(self._client, self.kb_id, body))

    def search(
        self,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        top_k: Optional[int] = 5,
        mode: Optional[SearchMode] = None,
        filters: Optional[Mapping[str, Any]] = None,
        include_context: bool = False,
        rerank: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        expand: Optional[Expand] = None,
        debug: bool = False,
    ) -> SearchResponse:
        options = {
            "rerank": rerank,
            "max_context_tokens": max_context_tokens,
            "expand": expand,
            "debug": debug,
        }
        flow = self._search(query, version, top_k, mode, filters, include_context, options)
        return run_flow(self._client, flow)

    async def asearch(
        self,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        top_k: Optional[int] = 5,
        mode: Optional[SearchMode] = None,
        filters: Optional[Mapping[str, Any]] = None,
        include_context: bool = False,
        rerank: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        expand: Optional[Expand] = None,
        debug: bool = False,
    ) -> SearchResponse:
        options = {
            "rerank": rerank,
            "max_context_tokens": max_context_tokens,
            "expand": expand,
            "debug": debug,
        }
        flow = self._search(query, version, top_k, mode, filters, include_context, options)
        return await arun_flow(self._client, flow)

    def query(
        self,
        query: Optional[str] = None,
        *,
        operation: Optional[Mapping[str, Any]] = None,
        version: Optional[VersionLike] = None,
    ) -> QueryResponse:
        return run_flow(self._client, self._query(query, operation, version))

    async def aquery(
        self,
        query: Optional[str] = None,
        *,
        operation: Optional[Mapping[str, Any]] = None,
        version: Optional[VersionLike] = None,
    ) -> QueryResponse:
        return await arun_flow(self._client, self._query(query, operation, version))

    def _query(
        self,
        query: Optional[str],
        operation: Optional[Mapping[str, Any]],
        version: Optional[VersionLike],
    ) -> Flow[QueryResponse]:
        body = _query_body(query, operation, version)
        return (yield from _query_flow(self._client, self.kb_id, body))

    def get_document(self, document_id: str) -> DocumentDetail:
        return run_flow(self._client, _document_flow(self._client, self.kb_id, document_id))

    async def aget_document(self, document_id: str) -> DocumentDetail:
        return await arun_flow(self._client, _document_flow(self._client, self.kb_id, document_id))

    def get_section(
        self,
        *,
        section: str,
        document_id: Optional[str] = None,
        document: Optional[str] = None,
        version: Optional[VersionLike] = None,
    ) -> KnowledgeSection:
        flow = _get_section_flow(self._client, self.kb_id, section, document_id, document, version)
        return run_flow(self._client, flow)

    async def aget_section(
        self,
        *,
        section: str,
        document_id: Optional[str] = None,
        document: Optional[str] = None,
        version: Optional[VersionLike] = None,
    ) -> KnowledgeSection:
        flow = _get_section_flow(self._client, self.kb_id, section, document_id, document, version)
        return await arun_flow(self._client, flow)

    def versions(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeVersion]:
        return run_flow(self._client, _versions_flow(self._client, self.kb_id, cursor, limit))

    async def aversions(
        self, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeVersion]:
        flow = _versions_flow(self._client, self.kb_id, cursor, limit)
        return await arun_flow(self._client, flow)

    def publish(
        self, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        flow = _publish_flow(self._client, self.kb_id, version, if_match, reason)
        return run_flow(self._client, flow)

    async def apublish(
        self, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        flow = _publish_flow(self._client, self.kb_id, version, if_match, reason)
        return await arun_flow(self._client, flow)

    def refresh(self) -> KnowledgeBase:
        self.detail = run_flow(self._client, _get_base_flow(self._client, self.kb_id))
        return self

    async def arefresh(self) -> KnowledgeBase:
        self.detail = await arun_flow(self._client, _get_base_flow(self._client, self.kb_id))
        return self


class KnowledgeCollectionsResource(_Base):
    def _list(
        self, kb_id: str, cursor: Optional[str], limit: Optional[int]
    ) -> Flow[Page[KnowledgeCollection]]:
        _require_cloud(self._client, "knowledge.collections.list")
        params = _page_params(cursor, limit)
        response = yield Call(
            "GET", _kb_path(kb_id, "collections"), params=params or None, retry=True
        )
        return _page(KnowledgeCollection, response, "collections")

    def _create(self, kb_id: str, body: dict[str, Any]) -> Flow[KnowledgeCollection]:
        _require_cloud(self._client, "knowledge.collections.create")
        response = yield Call("POST", _kb_path(kb_id, "collections"), body)
        return _wrapped(KnowledgeCollection, response, "collection")

    def _update(
        self, kb_id: str, collection_id: str, body: dict[str, Any], if_match: Optional[int]
    ) -> Flow[KnowledgeCollection]:
        _require_cloud(self._client, "knowledge.collections.update")
        path = _kb_path(kb_id, "collections", segment(_id(collection_id, "collection_id")))
        response = yield Call(
            "PATCH",
            path,
            body,
            if_match=None if if_match is None else _non_negative(if_match, "if_match"),
        )
        return _wrapped(KnowledgeCollection, response, "collection")

    def _delete(self, kb_id: str, collection_id: str) -> Flow[None]:
        _require_cloud(self._client, "knowledge.collections.delete")
        path = _kb_path(kb_id, "collections", segment(_id(collection_id, "collection_id")))
        yield Call("DELETE", path)

    @staticmethod
    def _create_body(
        collection_id: str,
        name: str,
        description: Optional[str],
        classification: Optional[str],
        restricted_roles: Optional[Sequence[str]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "collection_id": _text(collection_id, "collection_id"),
            "name": _text(name, "name"),
        }
        _set(body, "description", description)
        _set(body, "classification", classification)
        if restricted_roles is not None:
            body["restricted_roles"] = _strings(restricted_roles, "restricted_roles")
        return body

    @staticmethod
    def _update_body(
        name: Optional[str],
        description: Union[str, None, Unset],
        classification: Union[str, None, Unset],
        restricted_roles: Optional[Sequence[str]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        _set(body, "name", None if name is None else _text(name, "name"))
        _tri_state(body, "description", description)
        _tri_state(body, "classification", classification)
        if restricted_roles is not None:
            body["restricted_roles"] = _strings(restricted_roles, "restricted_roles")
        return body

    def list(
        self, kb_id: str, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeCollection]:
        return self._run(self._list(kb_id, cursor, limit))

    async def alist(
        self, kb_id: str, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeCollection]:
        return await self._arun(self._list(kb_id, cursor, limit))

    def create(
        self,
        kb_id: str,
        collection_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        classification: Optional[str] = None,
        restricted_roles: Optional[Sequence[str]] = None,
    ) -> KnowledgeCollection:
        body = self._create_body(collection_id, name, description, classification, restricted_roles)
        return self._run(self._create(kb_id, body))

    async def acreate(
        self,
        kb_id: str,
        collection_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        classification: Optional[str] = None,
        restricted_roles: Optional[Sequence[str]] = None,
    ) -> KnowledgeCollection:
        body = self._create_body(collection_id, name, description, classification, restricted_roles)
        return await self._arun(self._create(kb_id, body))

    def update(
        self,
        kb_id: str,
        collection_id: str,
        *,
        if_match: Optional[int] = None,
        name: Optional[str] = None,
        description: Union[str, None, Unset] = UNSET,
        classification: Union[str, None, Unset] = UNSET,
        restricted_roles: Optional[Sequence[str]] = None,
    ) -> KnowledgeCollection:
        body = self._update_body(name, description, classification, restricted_roles)
        return self._run(self._update(kb_id, collection_id, body, if_match))

    async def aupdate(
        self,
        kb_id: str,
        collection_id: str,
        *,
        if_match: Optional[int] = None,
        name: Optional[str] = None,
        description: Union[str, None, Unset] = UNSET,
        classification: Union[str, None, Unset] = UNSET,
        restricted_roles: Optional[Sequence[str]] = None,
    ) -> KnowledgeCollection:
        body = self._update_body(name, description, classification, restricted_roles)
        return await self._arun(self._update(kb_id, collection_id, body, if_match))

    def delete(self, kb_id: str, collection_id: str) -> None:
        self._run(self._delete(kb_id, collection_id))

    async def adelete(self, kb_id: str, collection_id: str) -> None:
        await self._arun(self._delete(kb_id, collection_id))


class KnowledgeDocumentsResource(_Base):
    def _list(
        self,
        kb_id: str,
        filters: Mapping[str, Optional[str]],
        tree: bool,
        cursor: Optional[str],
        limit: Optional[int],
    ) -> Flow[DocumentList]:
        _require_cloud(self._client, "knowledge.documents.list")
        params = _page_params(cursor, limit)
        for key, value in filters.items():
            if value is not None:
                params[key] = _text(value, key)
        if tree:
            params["view"] = "tree"
        response = yield Call(
            "GET", _kb_path(kb_id, "documents"), params=params or None, retry=True
        )
        if not isinstance(response.body.get("documents"), list):
            raise _invalid(response, "documents")
        return _bare(DocumentList, response, "document list")

    def _create(self, kb_id: str, body: dict[str, Any]) -> Flow[DocumentWrite]:
        _require_cloud(self._client, "knowledge.documents.create")
        response = yield Call("POST", _kb_path(kb_id, "documents"), body)
        return _bare(DocumentWrite, response, "document write")

    def _upload(
        self,
        kb_id: str,
        source: UploadSource,
        path: Optional[str],
        content_type: Optional[str],
        headers: Mapping[str, Optional[Any]],
    ) -> Flow[DocumentWrite]:
        _require_cloud(self._client, "knowledge.documents.upload")
        target = _kb_path(kb_id, "documents", "upload")
        kind = _content_type(content_type)
        data, document_path = yield from _source(source, path)
        if document_path is None:
            raise ValueError("path is required when uploading bytes")
        extra = _upload_headers(document_path, **headers)
        response = yield Call(
            "POST", target, content=data, content_type=kind, headers=extra, retry=True
        )
        return _bare(DocumentWrite, response, "document write")

    def _upload_revision(
        self,
        kb_id: str,
        document_id: str,
        source: UploadSource,
        if_match: int,
        content_type: Optional[str],
        change_message: Optional[str],
    ) -> Flow[DocumentWrite]:
        _require_cloud(self._client, "knowledge.documents.upload_revision")
        target = _document_path(kb_id, document_id, "upload")
        kind = _content_type(content_type)
        revision = _positive(if_match, "if_match")
        data, _ = yield from _source(source, document_id)
        extra = _upload_headers(None, None, None, None, change_message)
        response = yield Call(
            "POST",
            target,
            content=data,
            content_type=kind,
            headers=extra,
            if_match=revision,
        )
        return _bare(DocumentWrite, response, "document write")

    def _put_content(
        self, kb_id: str, document_id: str, body: dict[str, Any], if_match: int
    ) -> Flow[DocumentWrite]:
        _require_cloud(self._client, "knowledge.documents.put_content")
        response = yield Call(
            "PUT",
            _document_path(kb_id, document_id, "content"),
            body,
            if_match=_positive(if_match, "if_match"),
        )
        return _bare(DocumentWrite, response, "document write")

    def _update(
        self, kb_id: str, document_id: str, body: dict[str, Any], if_match: int
    ) -> Flow[KnowledgeDocument]:
        _require_cloud(self._client, "knowledge.documents.update")
        response = yield Call(
            "PATCH",
            _document_path(kb_id, document_id),
            body,
            if_match=_non_negative(if_match, "if_match"),
        )
        return _wrapped(KnowledgeDocument, response, "document")

    def _duplicate(self, kb_id: str, document_id: str, path: str) -> Flow[DocumentWrite]:
        _require_cloud(self._client, "knowledge.documents.duplicate")
        response = yield Call(
            "POST",
            _document_path(kb_id, document_id, "duplicate"),
            {"path": _text(path, "path")},
        )
        return _bare(DocumentWrite, response, "document write")

    def _delete(self, kb_id: str, document_id: str) -> Flow[None]:
        _require_cloud(self._client, "knowledge.documents.delete")
        yield Call("DELETE", _document_path(kb_id, document_id))

    def _restore(self, kb_id: str, document_id: str) -> Flow[KnowledgeDocument]:
        _require_cloud(self._client, "knowledge.documents.restore")
        response = yield Call("POST", _document_path(kb_id, document_id, "restore"), {})
        return _wrapped(KnowledgeDocument, response, "document")

    def _revisions(
        self, kb_id: str, document_id: str, cursor: Optional[str], limit: Optional[int]
    ) -> Flow[RevisionList]:
        _require_cloud(self._client, "knowledge.documents.revisions")
        params = _page_params(cursor, limit)
        response = yield Call(
            "GET",
            _document_path(kb_id, document_id, "revisions"),
            params=params or None,
            retry=True,
        )
        return _bare(RevisionList, response, "revision list")

    def _text(self, kb_id: str, document_id: str, revision: Optional[int]) -> Flow[DocumentText]:
        _require_cloud(self._client, "knowledge.documents.text")
        params = {"format": "text"}
        if revision is not None:
            params["revision"] = str(_positive(revision, "revision"))
        response = yield Call(
            "GET", _document_path(kb_id, document_id, "content"), params=params, retry=True
        )
        return _bare(DocumentText, response, "document text")

    def _sections(
        self,
        kb_id: str,
        document_id: str,
        revision: Optional[int],
        version: Optional[VersionLike],
        include_content: bool,
    ) -> Flow[SectionTree]:
        _require_cloud(self._client, "knowledge.documents.sections")
        if revision is not None and version is not None:
            raise ValueError("pass at most one of revision and version")
        params: dict[str, str] = {}
        if revision is not None:
            params["revision"] = str(_positive(revision, "revision"))
        _set(params, "version", _version_param(version))
        if include_content:
            params["include"] = "content"
        response = yield Call(
            "GET",
            _document_path(kb_id, document_id, "sections"),
            params=params or None,
            retry=True,
        )
        return _bare(SectionTree, response, "section tree")

    def _backlinks(
        self, kb_id: str, document_id: str, version: Optional[VersionLike]
    ) -> Flow[BacklinkList]:
        _require_cloud(self._client, "knowledge.documents.backlinks")
        params: dict[str, str] = {}
        _set(params, "version", _version_param(version))
        response = yield Call(
            "GET",
            _document_path(kb_id, document_id, "backlinks"),
            params=params or None,
            retry=True,
        )
        return _bare(BacklinkList, response, "backlinks")

    @staticmethod
    def _create_body(
        path: str,
        content: str,
        media_type: Optional[str],
        title: Optional[str],
        collection: Optional[str],
        tags: Optional[Sequence[str]],
        metadata: Optional[Mapping[str, str]],
        classification: Optional[str],
        change_message: Optional[str],
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("content must be a string; upload bytes with documents.upload")
        body: dict[str, Any] = {"path": _text(path, "path"), "content": content}
        _set(body, "media_type", media_type)
        _set(body, "title", title)
        _set(body, "collection", collection)
        if tags is not None:
            body["tags"] = _strings(tags, "tags")
        if metadata is not None:
            body["metadata"] = _labels(metadata, "metadata")
        _set(body, "classification", classification)
        _set(body, "change_message", change_message)
        return body

    @staticmethod
    def _update_body(
        path: Optional[str],
        title: Union[str, None, Unset],
        collection: Union[str, None, Unset],
        tags: Optional[Sequence[str]],
        metadata: Optional[Mapping[str, str]],
        classification: Optional[str],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        _set(body, "path", None if path is None else _text(path, "path"))
        _tri_state(body, "title", title)
        _tri_state(body, "collection", collection)
        if tags is not None:
            body["tags"] = _strings(tags, "tags")
        if metadata is not None:
            body["metadata"] = _labels(metadata, "metadata")
        _set(body, "classification", classification)
        return body

    @staticmethod
    def _content_body(
        content: str, media_type: Optional[str], change_message: Optional[str]
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        body: dict[str, Any] = {"content": content}
        _set(body, "media_type", media_type)
        _set(body, "change_message", change_message)
        return body

    def list(
        self,
        kb_id: str,
        *,
        path_prefix: Optional[str] = None,
        collection: Optional[str] = None,
        tag: Optional[str] = None,
        q: Optional[str] = None,
        status: Optional[str] = None,
        tree: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> DocumentList:
        filters = {
            "path_prefix": path_prefix,
            "collection": collection,
            "tag": tag,
            "q": q,
            "status": status,
        }
        return self._run(self._list(kb_id, filters, tree, cursor, limit))

    async def alist(
        self,
        kb_id: str,
        *,
        path_prefix: Optional[str] = None,
        collection: Optional[str] = None,
        tag: Optional[str] = None,
        q: Optional[str] = None,
        status: Optional[str] = None,
        tree: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> DocumentList:
        filters = {
            "path_prefix": path_prefix,
            "collection": collection,
            "tag": tag,
            "q": q,
            "status": status,
        }
        return await self._arun(self._list(kb_id, filters, tree, cursor, limit))

    def get(self, kb_id: str, document_id: str) -> DocumentDetail:
        return self._run(_document_flow(self._client, kb_id, document_id))

    async def aget(self, kb_id: str, document_id: str) -> DocumentDetail:
        return await self._arun(_document_flow(self._client, kb_id, document_id))

    def create(
        self,
        kb_id: str,
        path: str,
        content: str,
        *,
        media_type: Optional[str] = None,
        title: Optional[str] = None,
        collection: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
        classification: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        body = self._create_body(
            path,
            content,
            media_type,
            title,
            collection,
            tags,
            metadata,
            classification,
            change_message,
        )
        return self._run(self._create(kb_id, body))

    async def acreate(
        self,
        kb_id: str,
        path: str,
        content: str,
        *,
        media_type: Optional[str] = None,
        title: Optional[str] = None,
        collection: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
        classification: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        body = self._create_body(
            path,
            content,
            media_type,
            title,
            collection,
            tags,
            metadata,
            classification,
            change_message,
        )
        return await self._arun(self._create(kb_id, body))

    def upload(
        self,
        kb_id: str,
        source: UploadSource,
        *,
        path: Optional[str] = None,
        content_type: Optional[str] = None,
        collection: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        classification: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        headers = {
            "collection": collection,
            "tags": tags,
            "classification": classification,
            "change_message": change_message,
        }
        return self._run(self._upload(kb_id, source, path, content_type, headers))

    async def aupload(
        self,
        kb_id: str,
        source: UploadSource,
        *,
        path: Optional[str] = None,
        content_type: Optional[str] = None,
        collection: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        classification: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        headers = {
            "collection": collection,
            "tags": tags,
            "classification": classification,
            "change_message": change_message,
        }
        return await self._arun(self._upload(kb_id, source, path, content_type, headers))

    def upload_revision(
        self,
        kb_id: str,
        document_id: str,
        source: UploadSource,
        *,
        if_match: int,
        content_type: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        flow = self._upload_revision(
            kb_id, document_id, source, if_match, content_type, change_message
        )
        return self._run(flow)

    async def aupload_revision(
        self,
        kb_id: str,
        document_id: str,
        source: UploadSource,
        *,
        if_match: int,
        content_type: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        flow = self._upload_revision(
            kb_id, document_id, source, if_match, content_type, change_message
        )
        return await self._arun(flow)

    def put_content(
        self,
        kb_id: str,
        document_id: str,
        content: str,
        *,
        if_match: int,
        media_type: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        body = self._content_body(content, media_type, change_message)
        return self._run(self._put_content(kb_id, document_id, body, if_match))

    async def aput_content(
        self,
        kb_id: str,
        document_id: str,
        content: str,
        *,
        if_match: int,
        media_type: Optional[str] = None,
        change_message: Optional[str] = None,
    ) -> DocumentWrite:
        body = self._content_body(content, media_type, change_message)
        return await self._arun(self._put_content(kb_id, document_id, body, if_match))

    def update(
        self,
        kb_id: str,
        document_id: str,
        *,
        if_match: int,
        path: Optional[str] = None,
        title: Union[str, None, Unset] = UNSET,
        collection: Union[str, None, Unset] = UNSET,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
        classification: Optional[str] = None,
    ) -> KnowledgeDocument:
        body = self._update_body(path, title, collection, tags, metadata, classification)
        return self._run(self._update(kb_id, document_id, body, if_match))

    async def aupdate(
        self,
        kb_id: str,
        document_id: str,
        *,
        if_match: int,
        path: Optional[str] = None,
        title: Union[str, None, Unset] = UNSET,
        collection: Union[str, None, Unset] = UNSET,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
        classification: Optional[str] = None,
    ) -> KnowledgeDocument:
        body = self._update_body(path, title, collection, tags, metadata, classification)
        return await self._arun(self._update(kb_id, document_id, body, if_match))

    def duplicate(self, kb_id: str, document_id: str, *, path: str) -> DocumentWrite:
        return self._run(self._duplicate(kb_id, document_id, path))

    async def aduplicate(self, kb_id: str, document_id: str, *, path: str) -> DocumentWrite:
        return await self._arun(self._duplicate(kb_id, document_id, path))

    def delete(self, kb_id: str, document_id: str) -> None:
        self._run(self._delete(kb_id, document_id))

    async def adelete(self, kb_id: str, document_id: str) -> None:
        await self._arun(self._delete(kb_id, document_id))

    def restore(self, kb_id: str, document_id: str) -> KnowledgeDocument:
        return self._run(self._restore(kb_id, document_id))

    async def arestore(self, kb_id: str, document_id: str) -> KnowledgeDocument:
        return await self._arun(self._restore(kb_id, document_id))

    def revisions(
        self,
        kb_id: str,
        document_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> RevisionList:
        return self._run(self._revisions(kb_id, document_id, cursor, limit))

    async def arevisions(
        self,
        kb_id: str,
        document_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> RevisionList:
        return await self._arun(self._revisions(kb_id, document_id, cursor, limit))

    def text(self, kb_id: str, document_id: str, *, revision: Optional[int] = None) -> DocumentText:
        return self._run(self._text(kb_id, document_id, revision))

    async def atext(
        self, kb_id: str, document_id: str, *, revision: Optional[int] = None
    ) -> DocumentText:
        return await self._arun(self._text(kb_id, document_id, revision))

    def sections(
        self,
        kb_id: str,
        document_id: str,
        *,
        revision: Optional[int] = None,
        version: Optional[VersionLike] = None,
        include_content: bool = False,
    ) -> SectionTree:
        return self._run(self._sections(kb_id, document_id, revision, version, include_content))

    async def asections(
        self,
        kb_id: str,
        document_id: str,
        *,
        revision: Optional[int] = None,
        version: Optional[VersionLike] = None,
        include_content: bool = False,
    ) -> SectionTree:
        flow = self._sections(kb_id, document_id, revision, version, include_content)
        return await self._arun(flow)

    def section(
        self,
        kb_id: str,
        document_id: str,
        section_id: str,
        *,
        revision: Optional[int] = None,
        version: Optional[VersionLike] = None,
        include: Optional[Sequence[str]] = None,
    ) -> SectionDetail:
        flow = _section_flow(
            self._client, kb_id, document_id, section_id, revision, version, include
        )
        return self._run(flow)

    async def asection(
        self,
        kb_id: str,
        document_id: str,
        section_id: str,
        *,
        revision: Optional[int] = None,
        version: Optional[VersionLike] = None,
        include: Optional[Sequence[str]] = None,
    ) -> SectionDetail:
        flow = _section_flow(
            self._client, kb_id, document_id, section_id, revision, version, include
        )
        return await self._arun(flow)

    def backlinks(
        self, kb_id: str, document_id: str, *, version: Optional[VersionLike] = None
    ) -> BacklinkList:
        return self._run(self._backlinks(kb_id, document_id, version))

    async def abacklinks(
        self, kb_id: str, document_id: str, *, version: Optional[VersionLike] = None
    ) -> BacklinkList:
        return await self._arun(self._backlinks(kb_id, document_id, version))


class KnowledgeVersionsResource(_Base):
    def _get(self, kb_id: str, version: VersionLike) -> Flow[VersionDetail]:
        _require_cloud(self._client, "knowledge.versions.get")
        number = version_number(version)
        response = yield Call("GET", _kb_path(kb_id, "versions", str(number)), retry=True)
        detail = _bare(VersionDetail, response, "version")
        if detail.version.version != number or detail.version.kb_id != kb_id:
            raise _invalid(response, "version for the requested number")
        return detail

    def _create(self, kb_id: str, body: dict[str, Any]) -> Flow[VersionWrite]:
        _require_cloud(self._client, "knowledge.versions.create")
        retry = "idempotency_key" in body
        response = yield Call("POST", _kb_path(kb_id, "versions"), body, retry=retry)
        return _bare(VersionWrite, response, "version write")

    def _diff(
        self, kb_id: str, version: VersionLike, against: Optional[VersionLike]
    ) -> Flow[KnowledgeDiff]:
        _require_cloud(self._client, "knowledge.versions.diff")
        path = _kb_path(kb_id, "versions", str(version_number(version)), "diff")
        params = None if against is None else {"against": str(version_number(against))}
        response = yield Call("GET", path, params=params, retry=True)
        return _bare(KnowledgeDiff, response, "diff")

    def _verify(self, kb_id: str, version: VersionLike) -> Flow[VersionVerification]:
        _require_cloud(self._client, "knowledge.versions.verify")
        path = _kb_path(kb_id, "versions", str(version_number(version)), "verify")
        response = yield Call("GET", path, retry=True)
        return _bare(VersionVerification, response, "verification")

    def _decide(
        self, kb_id: str, version: VersionLike, action: str, reason: Optional[str]
    ) -> Flow[VersionDecision]:
        _require_cloud(self._client, f"knowledge.versions.{action}")
        path = _kb_path(kb_id, "versions", str(version_number(version)), action)
        body: dict[str, Any] = {}
        _set(body, "reason", None if reason is None else _text(reason, "reason"))
        response = yield Call("POST", path, body)
        return _bare(VersionDecision, response, "version decision")

    @staticmethod
    def _create_body(
        change_message: Optional[str],
        expected_draft_revision: Optional[int],
        idempotency_key: Optional[str],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        _set(body, "change_message", change_message)
        if expected_draft_revision is not None:
            body["expected_draft_revision"] = _non_negative(
                expected_draft_revision, "expected_draft_revision"
            )
        _set(
            body,
            "idempotency_key",
            None if idempotency_key is None else _text(idempotency_key, "idempotency_key"),
        )
        return body

    def list(
        self, kb_id: str, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeVersion]:
        return self._run(_versions_flow(self._client, kb_id, cursor, limit))

    async def alist(
        self, kb_id: str, *, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Page[KnowledgeVersion]:
        return await self._arun(_versions_flow(self._client, kb_id, cursor, limit))

    def get(self, kb_id: str, version: VersionLike) -> VersionDetail:
        return self._run(self._get(kb_id, version))

    async def aget(self, kb_id: str, version: VersionLike) -> VersionDetail:
        return await self._arun(self._get(kb_id, version))

    def create(
        self,
        kb_id: str,
        *,
        change_message: Optional[str] = None,
        expected_draft_revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> VersionWrite:
        body = self._create_body(change_message, expected_draft_revision, idempotency_key)
        return self._run(self._create(kb_id, body))

    async def acreate(
        self,
        kb_id: str,
        *,
        change_message: Optional[str] = None,
        expected_draft_revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> VersionWrite:
        body = self._create_body(change_message, expected_draft_revision, idempotency_key)
        return await self._arun(self._create(kb_id, body))

    def diff(
        self, kb_id: str, version: VersionLike, *, against: Optional[VersionLike] = None
    ) -> KnowledgeDiff:
        return self._run(self._diff(kb_id, version, against))

    async def adiff(
        self, kb_id: str, version: VersionLike, *, against: Optional[VersionLike] = None
    ) -> KnowledgeDiff:
        return await self._arun(self._diff(kb_id, version, against))

    def verify(self, kb_id: str, version: VersionLike) -> VersionVerification:
        return self._run(self._verify(kb_id, version))

    async def averify(self, kb_id: str, version: VersionLike) -> VersionVerification:
        return await self._arun(self._verify(kb_id, version))

    def approve(
        self, kb_id: str, version: VersionLike, *, reason: Optional[str] = None
    ) -> VersionDecision:
        return self._run(self._decide(kb_id, version, "approve", reason))

    async def aapprove(
        self, kb_id: str, version: VersionLike, *, reason: Optional[str] = None
    ) -> VersionDecision:
        return await self._arun(self._decide(kb_id, version, "approve", reason))

    def reject(
        self, kb_id: str, version: VersionLike, *, reason: Optional[str] = None
    ) -> VersionDecision:
        return self._run(self._decide(kb_id, version, "reject", reason))

    async def areject(
        self, kb_id: str, version: VersionLike, *, reason: Optional[str] = None
    ) -> VersionDecision:
        return await self._arun(self._decide(kb_id, version, "reject", reason))

    def publish(
        self, kb_id: str, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        return self._run(_publish_flow(self._client, kb_id, version, if_match, reason))

    async def apublish(
        self, kb_id: str, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        return await self._arun(_publish_flow(self._client, kb_id, version, if_match, reason))

    def rollback(
        self,
        kb_id: str,
        *,
        reason: str,
        if_match: int,
        to_version: Optional[VersionLike] = None,
    ) -> Publication:
        return self._run(_rollback_flow(self._client, kb_id, reason, to_version, if_match))

    async def arollback(
        self,
        kb_id: str,
        *,
        reason: str,
        if_match: int,
        to_version: Optional[VersionLike] = None,
    ) -> Publication:
        return await self._arun(_rollback_flow(self._client, kb_id, reason, to_version, if_match))


class KnowledgeJobsResource(_Base):
    def _get(self, job_id: str) -> Flow[KnowledgeJob]:
        _require_cloud(self._client, "knowledge.jobs.get")
        path = "/".join(("/v1/knowledge-jobs", segment(_id(job_id, "job_id"))))
        response = yield Call("GET", path, retry=True)
        job = _wrapped(KnowledgeJob, response, "job")
        if job.job_id != job_id:
            raise _invalid(response, "job for the requested job_id")
        return job

    @staticmethod
    def _timing(timeout: float, poll_interval: float) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("timeout must be a non-negative number of seconds")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be a positive number of seconds")

    @staticmethod
    def _timed_out(job: KnowledgeJob, timeout: float) -> ApiError:
        return ApiError(
            "knowledge_job_timeout",
            0,
            f"job {job.job_id} is still {job.status} after {timeout} seconds",
            {"job_id": job.job_id, "status": job.status},
        )

    def get(self, job_id: str) -> KnowledgeJob:
        return self._run(self._get(job_id))

    async def aget(self, job_id: str) -> KnowledgeJob:
        return await self._arun(self._get(job_id))

    def wait_for(
        self, job_id: str, *, timeout: float = 60.0, poll_interval: float = 1.0
    ) -> KnowledgeJob:
        self._timing(timeout, poll_interval)
        deadline = _monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.terminal:
                return job
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise self._timed_out(job, timeout)
            _sleep(min(poll_interval, remaining))

    async def await_for(
        self, job_id: str, *, timeout: float = 60.0, poll_interval: float = 1.0
    ) -> KnowledgeJob:
        self._timing(timeout, poll_interval)
        deadline = _monotonic() + timeout
        while True:
            job = await self.aget(job_id)
            if job.terminal:
                return job
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise self._timed_out(job, timeout)
            await _asleep(min(poll_interval, remaining))


class KnowledgeRetrievalsResource(_Base):
    def _list(
        self, filters: Mapping[str, Optional[str]], cursor: Optional[str], limit: Optional[int]
    ) -> Flow[Page[RetrievalEvent]]:
        _require_cloud(self._client, "knowledge.retrievals.list")
        params = _page_params(cursor, limit)
        for key, value in filters.items():
            if value is not None:
                params[key] = _text(value, key)
        response = yield Call("GET", "/v1/knowledge/retrievals", params=params or None, retry=True)
        return _page(RetrievalEvent, response, "events")

    def _get(self, event_id: str) -> Flow[RetrievalEvent]:
        _require_cloud(self._client, "knowledge.retrievals.get")
        path = "/".join(("/v1/knowledge/retrievals", segment(_id(event_id, "event_id"))))
        response = yield Call("GET", path, retry=True)
        event = _wrapped(RetrievalEvent, response, "event")
        if event.event_id != event_id:
            raise _invalid(response, "event for the requested event_id")
        return event

    def _snapshot(self, event_id: str) -> Flow[RetrievalSnapshot]:
        _require_cloud(self._client, "knowledge.retrievals.snapshot")
        path = "/".join(
            ("/v1/knowledge/retrievals", segment(_id(event_id, "event_id")), "snapshot")
        )
        response = yield Call("GET", path, retry=True)
        snapshot = _bare(RetrievalSnapshot, response, "snapshot")
        if snapshot.event.event_id != event_id:
            raise _invalid(response, "snapshot for the requested event_id")
        return snapshot

    def list(
        self,
        *,
        kb_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        binding_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Page[RetrievalEvent]:
        filters = {
            "kb_id": kb_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "binding_id": binding_id,
        }
        return self._run(self._list(filters, cursor, limit))

    async def alist(
        self,
        *,
        kb_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        binding_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Page[RetrievalEvent]:
        filters = {
            "kb_id": kb_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "binding_id": binding_id,
        }
        return await self._arun(self._list(filters, cursor, limit))

    def get(self, event_id: str) -> RetrievalEvent:
        return self._run(self._get(event_id))

    async def aget(self, event_id: str) -> RetrievalEvent:
        return await self._arun(self._get(event_id))

    def snapshot(self, event_id: str) -> RetrievalSnapshot:
        return self._run(self._snapshot(event_id))

    async def asnapshot(self, event_id: str) -> RetrievalSnapshot:
        return await self._arun(self._snapshot(event_id))


class AgentKnowledgeResource(_Base):
    def _get(self, agent_id: str) -> Flow[AgentKnowledge]:
        _require_cloud(self._client, "knowledge.agents.get")
        response = yield Call("GET", _agent_path(agent_id), retry=True)
        knowledge = _agent_knowledge(response)
        if knowledge.agent_id != agent_id:
            raise _invalid(response, "agent knowledge for the requested agent_id")
        return knowledge

    def _put(self, agent_id: str, body: dict[str, Any], if_match: int) -> Flow[AgentKnowledge]:
        _require_cloud(self._client, "knowledge.agents.put")
        response = yield Call(
            "PUT", _agent_path(agent_id), body, if_match=_non_negative(if_match, "if_match")
        )
        return _agent_knowledge(response)

    def _attach(self, agent_id: str, body: dict[str, Any]) -> Flow[AgentBindingResult]:
        _require_cloud(self._client, "knowledge.agents.attach")
        response = yield Call("POST", _agent_path(agent_id, "bindings"), body)
        return _bare(AgentBindingResult, response, "agent binding")

    def _detach(self, agent_id: str, binding_id: str) -> Flow[None]:
        _require_cloud(self._client, "knowledge.agents.detach")
        path = _agent_path(agent_id, "bindings", segment(_id(binding_id, "binding_id")))
        yield Call("DELETE", path)

    @staticmethod
    def _put_body(
        enabled: bool,
        bindings: Sequence[Mapping[str, Any]],
        retrieval: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        body: dict[str, Any] = {"enabled": enabled, "bindings": _bindings(bindings)}
        if retrieval is not None:
            if not isinstance(retrieval, Mapping):
                raise ValueError("retrieval must be a mapping")
            body["retrieval"] = dict(retrieval)
        return body

    def get(self, agent_id: str) -> AgentKnowledge:
        return self._run(self._get(agent_id))

    async def aget(self, agent_id: str) -> AgentKnowledge:
        return await self._arun(self._get(agent_id))

    def put(
        self,
        agent_id: str,
        *,
        if_match: int,
        enabled: bool,
        bindings: Sequence[Mapping[str, Any]],
        retrieval: Optional[Mapping[str, Any]] = None,
    ) -> AgentKnowledge:
        body = self._put_body(enabled, bindings, retrieval)
        return self._run(self._put(agent_id, body, if_match))

    async def aput(
        self,
        agent_id: str,
        *,
        if_match: int,
        enabled: bool,
        bindings: Sequence[Mapping[str, Any]],
        retrieval: Optional[Mapping[str, Any]] = None,
    ) -> AgentKnowledge:
        body = self._put_body(enabled, bindings, retrieval)
        return await self._arun(self._put(agent_id, body, if_match))

    def attach(
        self,
        agent_id: str,
        knowledge_base: str,
        *,
        version: VersionLike,
        access: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        max_classification: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> AgentBindingResult:
        body = _binding_input(
            knowledge_base, version, access, collections, max_classification, enabled
        )
        return self._run(self._attach(agent_id, body))

    async def aattach(
        self,
        agent_id: str,
        knowledge_base: str,
        *,
        version: VersionLike,
        access: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        max_classification: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> AgentBindingResult:
        body = _binding_input(
            knowledge_base, version, access, collections, max_classification, enabled
        )
        return await self._arun(self._attach(agent_id, body))

    def detach(self, agent_id: str, binding_id: str) -> None:
        self._run(self._detach(agent_id, binding_id))

    async def adetach(self, agent_id: str, binding_id: str) -> None:
        await self._arun(self._detach(agent_id, binding_id))


def _agent_search_body(
    query: str,
    knowledge_base: Optional[str],
    top_k: Optional[int],
    mode: Optional[str],
    filters: Optional[Mapping[str, Any]],
    include_context: bool,
    execution: Optional[Mapping[str, str]],
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": _text(query, "query")}
    _set(body, "knowledge_base", None if knowledge_base is None else _kb_id(knowledge_base))
    _set(body, "top_k", None if top_k is None else _ranged(top_k, "top_k", 1, 50))
    _set(body, "mode", _mode(mode))
    _set(body, "filters", _filters(filters))
    if include_context:
        body["include_context"] = True
    _set(body, "execution", _execution(execution))
    return body


def _agent_search_flow(
    client: Client, agent_id: str, body: Mapping[str, Any]
) -> Flow[AgentSearchResponse]:
    _require_cloud(client, "knowledge.agent_search")
    response = yield Call("POST", _agent_path(agent_id, "search"), dict(body), retry=True)
    return _bare(AgentSearchResponse, response, "agent search response")


class KnowledgeResource(_Base):
    def __init__(self, client: Client) -> None:
        super().__init__(client)
        self.collections = KnowledgeCollectionsResource(client)
        self.documents = KnowledgeDocumentsResource(client)
        self.versions = KnowledgeVersionsResource(client)
        self.jobs = KnowledgeJobsResource(client)
        self.retrievals = KnowledgeRetrievalsResource(client)
        self.agents = AgentKnowledgeResource(client)

    def _list(
        self, filters: Mapping[str, Optional[str]], cursor: Optional[str], limit: Optional[int]
    ) -> Flow[Page[KnowledgeBaseRecord]]:
        _require_cloud(self._client, "knowledge.list")
        params = _page_params(cursor, limit)
        for key, value in filters.items():
            if value is not None:
                params[key] = _text(value, key)
        response = yield Call("GET", "/v1/knowledge-bases", params=params or None, retry=True)
        return _page(KnowledgeBaseRecord, response, "knowledge_bases")

    def _get(self, kb_id: str) -> Flow[KnowledgeBase]:
        detail = yield from _get_base_flow(self._client, kb_id)
        return KnowledgeBase(self._client, kb_id, detail)

    def _create(self, body: dict[str, Any]) -> Flow[KnowledgeBaseRecord]:
        _require_cloud(self._client, "knowledge.create")
        response = yield Call("POST", "/v1/knowledge-bases", body)
        return _wrapped(KnowledgeBaseRecord, response, "knowledge_base")

    def _update(self, kb_id: str, body: dict[str, Any], if_match: int) -> Flow[KnowledgeBaseRecord]:
        _require_cloud(self._client, "knowledge.update")
        response = yield Call(
            "PATCH", _kb_path(kb_id), body, if_match=_non_negative(if_match, "if_match")
        )
        return _wrapped(KnowledgeBaseRecord, response, "knowledge_base")

    def _transition(
        self, kb_id: str, action: str, reason: Optional[str]
    ) -> Flow[KnowledgeBaseRecord]:
        _require_cloud(self._client, f"knowledge.{action}")
        body: dict[str, Any] = {}
        _set(body, "reason", None if reason is None else _text(reason, "reason"))
        response = yield Call("POST", _kb_path(kb_id, action), body)
        return _wrapped(KnowledgeBaseRecord, response, "knowledge_base")

    def _delete(self, kb_id: str) -> Flow[None]:
        _require_cloud(self._client, "knowledge.delete")
        yield Call("DELETE", _kb_path(kb_id))

    @staticmethod
    def _create_body(
        kb_id: str,
        name: str,
        description: Optional[str],
        owner: Optional[str],
        tags: Optional[Sequence[str]],
        labels: Optional[Mapping[str, str]],
        settings: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"kb_id": _kb_id(kb_id), "name": _text(name, "name")}
        _set(body, "description", description)
        _set(body, "owner", owner)
        if tags is not None:
            body["tags"] = _strings(tags, "tags")
        if labels is not None:
            body["labels"] = _labels(labels, "labels")
        if settings is not None:
            body["settings"] = dict(settings)
        return body

    @staticmethod
    def _update_body(
        name: Optional[str],
        description: Union[str, None, Unset],
        owner: Union[str, None, Unset],
        tags: Optional[Sequence[str]],
        labels: Optional[Mapping[str, str]],
        settings: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        _set(body, "name", None if name is None else _text(name, "name"))
        _tri_state(body, "description", description)
        _tri_state(body, "owner", owner)
        if tags is not None:
            body["tags"] = _strings(tags, "tags")
        if labels is not None:
            body["labels"] = _labels(labels, "labels")
        if settings is not None:
            body["settings"] = dict(settings)
        return body

    def base(self, kb_id: str) -> KnowledgeBase:
        return KnowledgeBase(self._client, kb_id)

    def list(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        sort: Optional[Literal["updated", "name"]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Page[KnowledgeBaseRecord]:
        filters = {"q": q, "status": status, "tag": tag, "sort": sort}
        return self._run(self._list(filters, cursor, limit))

    async def alist(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        sort: Optional[Literal["updated", "name"]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Page[KnowledgeBaseRecord]:
        filters = {"q": q, "status": status, "tag": tag, "sort": sort}
        return await self._arun(self._list(filters, cursor, limit))

    def get(self, kb_id: str) -> KnowledgeBase:
        return self._run(self._get(kb_id))

    async def aget(self, kb_id: str) -> KnowledgeBase:
        return await self._arun(self._get(kb_id))

    def create(
        self,
        kb_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        labels: Optional[Mapping[str, str]] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        body = self._create_body(kb_id, name, description, owner, tags, labels, settings)
        return self._run(self._create(body))

    async def acreate(
        self,
        kb_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        labels: Optional[Mapping[str, str]] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        body = self._create_body(kb_id, name, description, owner, tags, labels, settings)
        return await self._arun(self._create(body))

    def update(
        self,
        kb_id: str,
        *,
        if_match: int,
        name: Optional[str] = None,
        description: Union[str, None, Unset] = UNSET,
        owner: Union[str, None, Unset] = UNSET,
        tags: Optional[Sequence[str]] = None,
        labels: Optional[Mapping[str, str]] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        body = self._update_body(name, description, owner, tags, labels, settings)
        return self._run(self._update(kb_id, body, if_match))

    async def aupdate(
        self,
        kb_id: str,
        *,
        if_match: int,
        name: Optional[str] = None,
        description: Union[str, None, Unset] = UNSET,
        owner: Union[str, None, Unset] = UNSET,
        tags: Optional[Sequence[str]] = None,
        labels: Optional[Mapping[str, str]] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        body = self._update_body(name, description, owner, tags, labels, settings)
        return await self._arun(self._update(kb_id, body, if_match))

    def archive(self, kb_id: str, *, reason: Optional[str] = None) -> KnowledgeBaseRecord:
        return self._run(self._transition(kb_id, "archive", reason))

    async def aarchive(self, kb_id: str, *, reason: Optional[str] = None) -> KnowledgeBaseRecord:
        return await self._arun(self._transition(kb_id, "archive", reason))

    def restore(self, kb_id: str, *, reason: Optional[str] = None) -> KnowledgeBaseRecord:
        return self._run(self._transition(kb_id, "restore", reason))

    async def arestore(self, kb_id: str, *, reason: Optional[str] = None) -> KnowledgeBaseRecord:
        return await self._arun(self._transition(kb_id, "restore", reason))

    def delete(self, kb_id: str) -> None:
        self._run(self._delete(kb_id))

    async def adelete(self, kb_id: str) -> None:
        await self._arun(self._delete(kb_id))

    def search(
        self,
        kb_id: str,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        mode: Optional[SearchMode] = None,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        rerank: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        include_context: bool = False,
        expand: Optional[Expand] = None,
        debug: bool = False,
    ) -> SearchResponse:
        body = _search_body(
            query,
            version=version,
            mode=mode,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
            max_context_tokens=max_context_tokens,
            include_context=include_context,
            expand=expand,
            debug=debug,
        )
        return self._run(_search_flow(self._client, kb_id, body))

    async def asearch(
        self,
        kb_id: str,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        mode: Optional[SearchMode] = None,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
        rerank: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        include_context: bool = False,
        expand: Optional[Expand] = None,
        debug: bool = False,
    ) -> SearchResponse:
        body = _search_body(
            query,
            version=version,
            mode=mode,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
            max_context_tokens=max_context_tokens,
            include_context=include_context,
            expand=expand,
            debug=debug,
        )
        return await self._arun(_search_flow(self._client, kb_id, body))

    def query(
        self,
        kb_id: str,
        query: Optional[str] = None,
        *,
        operation: Optional[Mapping[str, Any]] = None,
        version: Optional[VersionLike] = None,
    ) -> QueryResponse:
        body = _query_body(query, operation, version)
        return self._run(_query_flow(self._client, kb_id, body))

    async def aquery(
        self,
        kb_id: str,
        query: Optional[str] = None,
        *,
        operation: Optional[Mapping[str, Any]] = None,
        version: Optional[VersionLike] = None,
    ) -> QueryResponse:
        body = _query_body(query, operation, version)
        return await self._arun(_query_flow(self._client, kb_id, body))

    def answer(
        self,
        kb_id: str,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> AnswerResponse:
        body = _answer_body(query, version, top_k, filters)
        return self._run(_answer_flow(self._client, kb_id, body))

    async def aanswer(
        self,
        kb_id: str,
        query: str,
        *,
        version: Optional[VersionLike] = None,
        top_k: Optional[int] = None,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> AnswerResponse:
        body = _answer_body(query, version, top_k, filters)
        return await self._arun(_answer_flow(self._client, kb_id, body))

    def publish(
        self, kb_id: str, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        return self._run(_publish_flow(self._client, kb_id, version, if_match, reason))

    async def apublish(
        self, kb_id: str, version: VersionLike, *, if_match: int, reason: Optional[str] = None
    ) -> Publication:
        return await self._arun(_publish_flow(self._client, kb_id, version, if_match, reason))

    def rollback(
        self,
        kb_id: str,
        *,
        reason: str,
        if_match: int,
        to_version: Optional[VersionLike] = None,
    ) -> Publication:
        return self._run(_rollback_flow(self._client, kb_id, reason, to_version, if_match))

    async def arollback(
        self,
        kb_id: str,
        *,
        reason: str,
        if_match: int,
        to_version: Optional[VersionLike] = None,
    ) -> Publication:
        return await self._arun(_rollback_flow(self._client, kb_id, reason, to_version, if_match))

    def agent_search(
        self,
        agent_id: str,
        query: str,
        *,
        knowledge_base: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        filters: Optional[Mapping[str, Any]] = None,
        include_context: bool = False,
        execution: Optional[Mapping[str, str]] = None,
    ) -> AgentSearchResponse:
        body = _agent_search_body(
            query, knowledge_base, top_k, mode, filters, include_context, execution
        )
        return self._run(_agent_search_flow(self._client, agent_id, body))

    async def aagent_search(
        self,
        agent_id: str,
        query: str,
        *,
        knowledge_base: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        filters: Optional[Mapping[str, Any]] = None,
        include_context: bool = False,
        execution: Optional[Mapping[str, str]] = None,
    ) -> AgentSearchResponse:
        body = _agent_search_body(
            query, knowledge_base, top_k, mode, filters, include_context, execution
        )
        return await self._arun(_agent_search_flow(self._client, agent_id, body))
