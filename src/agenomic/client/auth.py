"""Authentication helpers for Agenomic Cloud."""

from __future__ import annotations


def bearer_header(api_key: str) -> dict[str, str]:
    """Return the Authorization header for `api_key`.

    Example:
        >>> bearer_header("k")
        {'Authorization': 'Bearer k'}
    """
    return {"Authorization": f"Bearer {api_key}"}
