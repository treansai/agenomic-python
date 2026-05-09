"""Retry policy used by the cloud client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy.

    The cloud client retries network errors and 429/502/503/504 responses
    up to ``max_retries`` times, sleeping ``base_delay * 2**attempt`` seconds
    between attempts. ``Retry-After`` headers override the exponential delay.
    """

    max_retries: int = 3
    base_delay: float = 0.2

    def delay_for(self, attempt: int) -> float:
        return float(self.base_delay * (2**attempt))
