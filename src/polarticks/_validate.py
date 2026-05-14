"""Shared validation helpers for polarticks indicators."""

from __future__ import annotations


def _validate_period(period: int, name: str, min_period: int = 1) -> None:
    """Raise ``ValueError`` when *period* is below *min_period*.

    Args:
        period: The period value supplied by the caller.
        name: Indicator name used in the error message.
        min_period: Minimum acceptable value (default 1).

    Raises:
        ValueError: If ``period < min_period``.
    """
    if period < min_period:
        raise ValueError(f"{name} period must be at least {min_period}, got {period}.")
