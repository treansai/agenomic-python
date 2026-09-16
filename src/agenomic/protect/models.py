"""Typed mirrors of the Protect wire types used by the SDK."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ProtectOverlay(BaseModel):
    """Instruction overlay computed by the gateway for one Protect run.

    Example:
        >>> ProtectOverlay(version="1", digest="blake3:x", text="Only call allowed tools.").truncated
        False
    """

    model_config = ConfigDict(extra="allow")

    version: str
    digest: str
    text: str
    policies: list[str] = Field(default_factory=list)
    truncated: bool = False


Overlay = Optional[Union[str, ProtectOverlay]]


def overlay_text(overlay: Overlay) -> Optional[str]:
    """Text of an overlay argument (``None`` when nothing is to be injected).

    Example:
        >>> overlay_text(ProtectOverlay(version="1", digest="d", text="Be safe."))
        'Be safe.'
        >>> overlay_text("") is None
        True
    """
    if overlay is None:
        return None
    text = overlay if isinstance(overlay, str) else overlay.text
    return text or None


__all__ = ["Overlay", "ProtectOverlay", "overlay_text"]
