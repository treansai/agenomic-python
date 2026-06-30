"""Model-provider connectors.

Provider integrations live here. They are intentionally light: the SDK core
depends only on ``httpx`` (already a hard dependency), so importing this
package never pulls in a vendor SDK. Provider-specific extras (e.g.
``huggingface-hub``) are optional and only used by helper utilities, never
required by the connector itself.
"""

from agenomic.providers.huggingface import (
    HuggingFaceAuthError,
    HuggingFaceClient,
    HuggingFaceConfig,
    HuggingFaceError,
    ModelMetadata,
    build_lockfile_model,
    is_huggingface,
    normalize_provider,
)

__all__ = [
    "HuggingFaceAuthError",
    "HuggingFaceClient",
    "HuggingFaceConfig",
    "HuggingFaceError",
    "ModelMetadata",
    "build_lockfile_model",
    "is_huggingface",
    "normalize_provider",
]
