from __future__ import annotations

import enum
import re
from typing import Any, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "UNSET",
    "ActorRef",
    "AgentBindingResult",
    "AgentKnowledge",
    "AgentKnowledgeBinding",
    "AgentKnowledgeConfig",
    "AgentKnowledgeManifest",
    "AgentKnowledgeRelease",
    "AgentSearchResponse",
    "AnswerResponse",
    "Backlink",
    "BacklinkList",
    "CandidateCounts",
    "Citation",
    "ConfigurationDiff",
    "ContentDiff",
    "DiffSummary",
    "DiffVersionRef",
    "DigestMismatch",
    "DocumentChange",
    "DocumentChanges",
    "DocumentDetail",
    "DocumentList",
    "DocumentRevision",
    "DocumentSource",
    "DocumentText",
    "DocumentWrite",
    "EvidenceConflict",
    "ExecutionPin",
    "FolderNode",
    "KnowledgeBaseDetail",
    "KnowledgeBaseHealth",
    "KnowledgeBaseRecord",
    "KnowledgeBaseStats",
    "KnowledgeCollection",
    "KnowledgeDiff",
    "KnowledgeDocument",
    "KnowledgeJob",
    "KnowledgeRefs",
    "KnowledgeSection",
    "KnowledgeVersion",
    "MetadataChange",
    "Publication",
    "PublicationEvent",
    "QueryResponse",
    "RetrievalEvent",
    "RetrievalEventResult",
    "RetrievalInfo",
    "RetrievalPrincipal",
    "RetrievalSnapshot",
    "RevisionList",
    "RiskAssessment",
    "SearchResponse",
    "SearchResult",
    "SearchScores",
    "SectionChange",
    "SectionDetail",
    "SectionMatch",
    "SectionTree",
    "SignatureVerification",
    "Unset",
    "VersionApproval",
    "VersionCounts",
    "VersionDecision",
    "VersionDetail",
    "VersionLike",
    "VersionSelector",
    "VersionSignature",
    "VersionSummary",
    "VersionVerification",
    "VersionWrite",
    "normalize_version",
    "version_number",
]

_VERSION_TEXT = re.compile(r"v?([1-9][0-9]{0,9})", re.ASCII)
_MAX_VERSION = 2147483647
_RETRIEVAL_REF_KEYS = (
    "event_id",
    "kb_id",
    "version",
    "version_manifest_digest",
    "index_config_digest",
    "mode",
)
_CITATION_KEYS = (
    "uri",
    "kb_id",
    "version",
    "document_id",
    "document_revision",
    "section_id",
    "chunk_id",
    "content_digest",
)

VersionSelector = Union[int, Literal["published", "draft"]]
VersionLike = Union[int, str]


class Unset(enum.Enum):
    UNSET = "UNSET"


UNSET = Unset.UNSET


def normalize_version(value: VersionLike) -> VersionSelector:
    if isinstance(value, bool):
        raise ValueError("version must be a number, 'v<n>', 'published' or 'draft'")
    if isinstance(value, int):
        if not 1 <= value <= _MAX_VERSION:
            raise ValueError("version must be between 1 and 2147483647")
        return value
    if not isinstance(value, str):
        raise ValueError("version must be a number, 'v<n>', 'published' or 'draft'")
    if value == "published":
        return "published"
    if value == "draft":
        return "draft"
    match = _VERSION_TEXT.fullmatch(value)
    if match is None:
        raise ValueError("version must be a number, 'v<n>', 'published' or 'draft'")
    number = int(match.group(1))
    if number > _MAX_VERSION:
        raise ValueError("version must be between 1 and 2147483647")
    return number


def version_number(value: VersionLike) -> int:
    normalized = normalize_version(value)
    if not isinstance(normalized, int):
        raise ValueError("this call takes a version number, not 'published' or 'draft'")
    return normalized


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class KnowledgeRefs(TypedDict):
    retrievals: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    agent_manifest_digest: Optional[str]


class ActorRef(_Model):
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None


class KnowledgeBaseRecord(_Model):
    kb_id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: list[str] = []
    labels: dict[str, str] = {}
    status: str
    uri: Optional[str] = None
    agenomic_uri: Optional[str] = None
    metadata_revision: int
    draft_revision: int = 0
    latest_version: Optional[int] = None
    published_version: Optional[int] = None
    publication_generation: int
    document_count: int = 0
    storage_bytes: int = 0
    agent_count: int = 0
    last_sync_at: Optional[str] = None
    health: Optional[str] = None
    settings: dict[str, Any] = {}
    created_at: Optional[str] = None
    created_by: Optional[ActorRef] = None
    updated_at: Optional[str] = None
    archived_at: Optional[str] = None
    archive_reason: Optional[str] = None


class KnowledgeBaseStats(_Model):
    collection_count: int = 0
    deleted_document_count: int = 0
    section_count: int = 0
    token_count: int = 0
    version_count: int = 0
    source_count: int = 0
    documents_by_parse_status: dict[str, int] = {}
    documents_by_classification: dict[str, int] = {}
    documents_with_secret_findings: int = 0
    pending_jobs: int = 0
    failed_jobs: int = 0
    retrievals_24h: int = 0
    last_retrieval_at: Optional[str] = None


class KnowledgeBaseHealth(_Model):
    status: str
    reasons: list[str] = []


class VersionSummary(_Model):
    version: int
    status: str
    manifest_digest: str
    change_message: Optional[str] = None
    document_count: int = 0
    created_at: Optional[str] = None
    ready_at: Optional[str] = None


class KnowledgeBaseDetail(_Model):
    knowledge_base: KnowledgeBaseRecord
    stats: Optional[KnowledgeBaseStats] = None
    health: Optional[KnowledgeBaseHealth] = None
    published: Optional[VersionSummary] = None


class KnowledgeCollection(_Model):
    collection_id: str
    name: str
    description: Optional[str] = None
    classification: Optional[str] = None
    restricted_roles: list[str] = []
    document_count: int = 0
    revision: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentSource(_Model):
    kind: str
    source_id: str
    external_ref: Optional[str] = None


class KnowledgeDocument(_Model):
    document_id: str
    kb_id: str
    uri: Optional[str] = None
    path: str
    title: str
    collection: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, str] = {}
    classification: str
    status: str
    current_revision: int
    metadata_revision: int
    media_type: Optional[str] = None
    format: Optional[str] = None
    byte_size: int = 0
    token_count: int = 0
    section_count: int = 0
    parse_status: Optional[str] = None
    content_digest: Optional[str] = None
    source: Optional[DocumentSource] = None
    created_at: Optional[str] = None
    created_by: Optional[ActorRef] = None
    updated_at: Optional[str] = None
    updated_by: Optional[ActorRef] = None
    deleted_at: Optional[str] = None


class DocumentRevision(_Model):
    revision: int
    media_type: Optional[str] = None
    format: Optional[str] = None
    byte_size: int = 0
    blob_digest: Optional[str] = None
    content_digest: Optional[str] = None
    parser_version: Optional[str] = None
    parse_status: Optional[str] = None
    parse_error_code: Optional[str] = None
    section_count: int = 0
    token_count: int = 0
    secret_findings: int = 0
    change_message: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    created_by: Optional[ActorRef] = None
    parsed_at: Optional[str] = None


class DocumentDetail(_Model):
    document: KnowledgeDocument
    revision: Optional[DocumentRevision] = None


class FolderNode(_Model):
    path: str
    name: str
    document_count: int = 0
    children: list[FolderNode] = []


class DocumentList(_Model):
    documents: list[KnowledgeDocument]
    next_cursor: Optional[str] = None
    tree: Optional[list[FolderNode]] = None


class KnowledgeJob(_Model):
    job_id: str
    kb_id: Optional[str] = None
    kind: str
    status: str
    attempts: int = 0
    max_attempts: int = 0
    progress: dict[str, Any] = {}
    error_code: Optional[str] = None
    error: Optional[str] = None
    subject: dict[str, Any] = {}
    requested_at: Optional[str] = None
    requested_by: Optional[ActorRef] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_at: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")


class DocumentWrite(_Model):
    document: KnowledgeDocument
    revision: Optional[DocumentRevision] = None
    job: Optional[KnowledgeJob] = None
    created: bool = True


class RevisionList(_Model):
    document_id: str
    revisions: list[DocumentRevision]
    next_cursor: Optional[str] = None


class DocumentText(_Model):
    document_id: str
    revision: int
    media_type: Optional[str] = None
    content_digest: Optional[str] = None
    text: str


class KnowledgeSection(_Model):
    section_id: str
    document_id: str
    parent_section_id: Optional[str] = None
    ordinal: int = 0
    depth: int = 0
    kind: Optional[str] = None
    heading: str
    heading_path: list[str] = []
    anchor: Optional[str] = None
    tags: list[str] = []
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    token_count: int = 0
    content_digest: str
    section_version: Optional[int] = None
    content: Optional[str] = None
    children: Optional[list[KnowledgeSection]] = None


class SectionTree(_Model):
    document_id: str
    revision: int
    version: Optional[int] = None
    sections: list[KnowledgeSection]


class SectionDetail(_Model):
    document_id: str
    revision: int
    version: Optional[int] = None
    section: KnowledgeSection
    children: Optional[list[KnowledgeSection]] = None
    descendants: Optional[list[KnowledgeSection]] = None


class Backlink(_Model):
    document_id: str
    path: str
    title: Optional[str] = None
    section_id: str
    heading_path: list[str] = []
    target_section_id: Optional[str] = None
    link_text: Optional[str] = None


class BacklinkList(_Model):
    document_id: str
    version: Optional[int] = None
    backlinks: list[Backlink]


class SearchScores(_Model):
    keyword: Optional[float] = None
    semantic: Optional[float] = None
    rerank: Optional[float] = None


class RiskAssessment(_Model):
    score: int = 0
    level: str
    flags: list[str] = []


class Citation(_Model):
    uri: str
    kb_id: str
    version: Optional[int] = None
    document_id: str
    document_revision: int
    section_id: str
    chunk_id: Optional[str] = None
    content_digest: str


class SearchResult(_Model):
    rank: int
    evidence_id: str
    chunk_id: Optional[str] = None
    kb_id: str
    version: Optional[int] = None
    document_id: str
    document_revision: int
    path: str
    title: Optional[str] = None
    section_id: str
    heading_path: list[str] = []
    page: Optional[int] = None
    text: str
    token_count: int = 0
    score: float
    scores: SearchScores = SearchScores()
    risk: RiskAssessment
    citation: Citation


class CandidateCounts(_Model):
    keyword: int = 0
    semantic: int = 0


class RetrievalInfo(_Model):
    event_id: str
    kb_id: str
    version: Optional[int] = None
    version_manifest_digest: Optional[str] = None
    index_config_digest: Optional[str] = None
    mode: str
    vector_backend: Optional[str] = None
    latency_ms: int = 0
    candidates: CandidateCounts = CandidateCounts()
    keyword_truncated: bool = False
    filtered_count: int = 0
    excluded_for_risk: int = 0
    tokens_returned: int = 0


def _refs(
    retrievals: list[RetrievalInfo],
    results: list[SearchResult],
    agent_manifest_digest: Optional[str],
) -> KnowledgeRefs:
    return {
        "retrievals": [
            {key: getattr(item, key) for key in _RETRIEVAL_REF_KEYS} for item in retrievals
        ],
        "citations": [
            {key: getattr(item.citation, key) for key in _CITATION_KEYS} for item in results
        ],
        "agent_manifest_digest": agent_manifest_digest,
    }


class SearchResponse(_Model):
    results: list[SearchResult]
    context: Optional[str] = None
    retrieval: RetrievalInfo
    debug: Optional[dict[str, Any]] = None

    def knowledge_refs(self) -> KnowledgeRefs:
        return _refs([self.retrieval], self.results, None)


class SectionMatch(_Model):
    section_id: str
    document_id: str
    document_revision: int
    path: str
    heading_path: list[str] = []
    match_kind: str
    score: float


class QueryResponse(_Model):
    operation: dict[str, Any]
    matches: list[SectionMatch]
    sections: list[KnowledgeSection]
    documents: list[KnowledgeDocument] = []
    retrieval: RetrievalInfo


class EvidenceConflict(_Model):
    kind: str
    evidence_ids: list[str] = []
    detail: str
    heuristic: bool = True


class AnswerResponse(_Model):
    answer: Optional[str] = None
    abstained: bool
    reason: Optional[str] = None
    citations: list[Citation] = []
    evidence: list[SearchResult] = []
    conflicts: list[EvidenceConflict] = []
    invalid_citations: list[str] = []
    model: Optional[str] = None
    retrieval: RetrievalInfo

    def knowledge_refs(self) -> KnowledgeRefs:
        return _refs([self.retrieval], self.evidence, None)


class ExecutionPin(_Model):
    execution_key: str
    manifest_digest: str
    resolved_via: str


class AgentSearchResponse(_Model):
    results: list[SearchResult]
    context: Optional[str] = None
    retrievals: list[RetrievalInfo]
    execution: ExecutionPin
    debug: Optional[dict[str, Any]] = None

    def knowledge_refs(self) -> KnowledgeRefs:
        return _refs(self.retrievals, self.results, self.execution.manifest_digest)


class VersionCounts(_Model):
    documents: int = 0
    sections: int = 0
    chunks: int = 0
    tokens: int = 0
    bytes: int = 0


class VersionSignature(_Model):
    key_id: str
    signed_at: Optional[str] = None
    algorithm: Optional[str] = None


class KnowledgeVersion(_Model):
    kb_id: str
    version: int
    uri: Optional[str] = None
    status: str
    manifest_digest: str
    index_config_digest: Optional[str] = None
    parent_version: Optional[int] = None
    change_message: Optional[str] = None
    counts: VersionCounts = VersionCounts()
    signature: Optional[VersionSignature] = None
    created_at: Optional[str] = None
    created_by: Optional[ActorRef] = None
    ready_at: Optional[str] = None
    decided_at: Optional[str] = None
    decided_by: Optional[ActorRef] = None
    decision_reason: Optional[str] = None
    publishable: bool = False
    published: bool = False


class VersionApproval(_Model):
    approver_user_id: str
    decision: str
    reason: Optional[str] = None
    manifest_digest: str
    created_at: Optional[str] = None


class VersionDetail(_Model):
    version: KnowledgeVersion
    manifest: dict[str, Any]
    approvals: list[VersionApproval] = []


class VersionWrite(_Model):
    version: KnowledgeVersion
    job: Optional[KnowledgeJob] = None
    created: bool = True


class PublicationEvent(_Model):
    generation: int
    action: str
    from_version: Optional[int] = None
    to_version: Optional[int] = None
    reason: Optional[str] = None
    actor: Optional[ActorRef] = None
    created_at: Optional[str] = None


class VersionDecision(_Model):
    version: KnowledgeVersion
    approval: Optional[VersionApproval] = None
    event: Optional[PublicationEvent] = None


class Publication(_Model):
    knowledge_base: KnowledgeBaseRecord
    event: Optional[PublicationEvent] = None


class DiffVersionRef(_Model):
    version: int
    manifest_digest: str
    index_config_digest: Optional[str] = None


class DocumentChange(_Model):
    document_id: str
    path: str
    from_path: Optional[str] = None
    from_revision: Optional[int] = None
    to_revision: Optional[int] = None


class DocumentChanges(_Model):
    added: list[DocumentChange] = []
    modified: list[DocumentChange] = []
    deleted: list[DocumentChange] = []
    moved: list[DocumentChange] = []


class SectionChange(_Model):
    document_id: str
    path: str
    section_id: str
    heading_path: list[str] = []
    change: str


class ContentDiff(_Model):
    documents: DocumentChanges
    sections: list[SectionChange] = []


class ConfigurationDiff(_Model):
    index_config_changed: bool = False
    chunking_changed: bool = False
    embedding_changed: bool = False
    text_search_changed: bool = False


class MetadataChange(_Model):
    document_id: str
    path: str
    field: str
    before: Any = None
    after: Any = None


class DiffSummary(_Model):
    documents_added: int = 0
    documents_modified: int = 0
    documents_deleted: int = 0
    documents_moved: int = 0
    sections_added: int = 0
    sections_modified: int = 0
    sections_deleted: int = 0
    metadata_changes: int = 0
    configuration_changed: bool = False
    affected_agents: int = 0


class AffectedAgent(_Model):
    agent_id: str
    agent_name: Optional[str] = None
    binding_id: str
    selector: str
    pinned_version: Optional[int] = None


class KnowledgeDiff(_Model):
    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    kb_id: str
    from_version: Optional[DiffVersionRef] = Field(default=None, alias="from")
    to_version: DiffVersionRef = Field(alias="to")
    identical: bool
    content: ContentDiff
    configuration: ConfigurationDiff
    metadata: list[MetadataChange] = []
    summary: DiffSummary
    affected_agents: list[AffectedAgent] = []


class SignatureVerification(_Model):
    present: bool
    valid: bool
    key_id: Optional[str] = None


class DigestMismatch(_Model):
    document_id: str
    expected: str
    actual: Optional[str] = None


class VersionVerification(_Model):
    kb_id: str
    version: int
    manifest_digest: str
    manifest_digest_valid: bool
    signature: SignatureVerification
    documents_checked: int = 0
    digest_errors: list[DigestMismatch] = []

    @property
    def valid(self) -> bool:
        signature_ok = self.signature.valid or not self.signature.present
        return self.manifest_digest_valid and signature_ok and not self.digest_errors


class AgentKnowledgeConfig(_Model):
    enabled: bool
    retrieval: dict[str, Any] = {}
    revision: int


class AgentKnowledgeBinding(_Model):
    binding_id: str
    knowledge_base: str
    knowledge_base_name: Optional[str] = None
    version: Union[int, str]
    resolved_version: Optional[int] = None
    access: str = "read"
    collections: list[str] = []
    max_classification: Optional[str] = None
    enabled: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AgentKnowledgeManifest(_Model):
    digest: str
    document: dict[str, Any]


class AgentKnowledgeRelease(_Model):
    release_id: str
    release_name: Optional[str] = None
    genome_version: Optional[str] = None
    knowledge_manifest_digest: Optional[str] = None


class AgentKnowledge(_Model):
    agent_id: str
    config: AgentKnowledgeConfig
    bindings: list[AgentKnowledgeBinding] = []
    manifest: AgentKnowledgeManifest
    release: Optional[AgentKnowledgeRelease] = None
    drift: Optional[str] = None


class AgentBindingResult(_Model):
    binding: AgentKnowledgeBinding
    knowledge: AgentKnowledge


class RetrievalPrincipal(_Model):
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    auth_method: Optional[str] = None


class RetrievalEventResult(_Model):
    rank: int
    chunk_id: Optional[str] = None
    document_id: str
    document_revision: int
    section_id: str
    content_digest: str
    score: float


class RetrievalEvent(_Model):
    event_id: str
    kb_id: str
    version: Optional[int] = None
    version_manifest_digest: Optional[str] = None
    index_config_digest: Optional[str] = None
    agent_id: Optional[str] = None
    agent_manifest_digest: Optional[str] = None
    execution_key: Optional[str] = None
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    principal: Optional[RetrievalPrincipal] = None
    surface: str
    operation: str
    query_text: Optional[str] = None
    query_digest: str
    strategy: Optional[str] = None
    params: dict[str, Any] = {}
    results: list[RetrievalEventResult] = []
    decision: str
    reason_codes: list[str] = []
    filtered_count: int = 0
    latency_ms: int = 0
    tokens_returned: int = 0
    embedding_tokens: int = 0
    cost_microusd: int = 0
    created_at: Optional[str] = None


class RetrievalSnapshot(_Model):
    event: RetrievalEvent
    evidence: list[SearchResult]
    verified: bool
    missing: list[str] = []
