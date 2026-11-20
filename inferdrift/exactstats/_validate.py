# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Closed-domain input validation for the exact statistics.

Every public entry point validates its inputs here before any computation.
Each rejection raises :class:`StatsInputError` carrying a reason code from the
closed set :data:`REASON_CODES`, the offending field name and a detail.
Nothing is coerced: booleans are never numbers, floats are never counts (even
when integral), strings are never numbers, NaN and infinities never satisfy a
range, and absence is never a default.

Declared domains: counts are Python integers (never bool) in 0..the function's
cap; alpha is a finite real in the open interval (0, 1), every such value
accepted including subnormals such as 5e-324 (derived per-tail levels that
would underflow are carried as logarithms, see ``_numeric``); probabilities lie
in [0, 1]; the nuisance fraction lies in (0, 1).
"""

from __future__ import annotations

import math
from typing import Any

MAX_EXACT_INT = 2**53 - 1

REASON_CODES = frozenset({
    "WRONG_TYPE", "BOOL_NOT_NUMERIC", "NONFINITE", "NEGATIVE_COUNT", "COUNT_OUT_OF_RANGE", "ALPHA_OUT_OF_RANGE",
    "PROBABILITY_OUT_OF_RANGE", "INCONSISTENT_COUNTS", "UNKNOWN_ENUM", "VALUE_OUT_OF_BOUNDS",
})


class StatsInputError(ValueError):
    """A rejected input.  ``reason`` is one of :data:`REASON_CODES`."""

    def __init__(self, reason: str, field: str, detail: str) -> None:
        if reason not in REASON_CODES:  # pragma: no cover - programming error guard
            raise AssertionError(f"unknown reason code {reason!r}")
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(f"{reason}: {field}: {detail}")


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def check_count(value: Any, field: str, *, maximum: int = MAX_EXACT_INT) -> int:
    """A nonnegative integer count.  Rejects bool, float (even integral), str, None."""
    if is_bool(value):
        raise StatsInputError("BOOL_NOT_NUMERIC", field, "booleans are not counts")
    if isinstance(value, int):
        v = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise StatsInputError("NONFINITE", field, f"count is {value!r}")
        raise StatsInputError("WRONG_TYPE", field, "counts must be integers, not floats")
    else:
        raise StatsInputError("WRONG_TYPE", field, f"expected an integer count, got {type(value).__name__}")
    if v < 0:
        raise StatsInputError("NEGATIVE_COUNT", field, f"count {v} is negative")
    if v > maximum:
        raise StatsInputError("COUNT_OUT_OF_RANGE", field, f"count {v} exceeds the supported maximum {maximum}")
    return v


def check_real(value: Any, field: str) -> float:
    """A finite real number given as int (not bool) or float.  Returns float."""
    if is_bool(value):
        raise StatsInputError("BOOL_NOT_NUMERIC", field, "booleans are not numbers")
    if isinstance(value, int):
        if abs(value) > MAX_EXACT_INT:
            raise StatsInputError("VALUE_OUT_OF_BOUNDS", field, "integer outside the exact float64 range")
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StatsInputError("NONFINITE", field, f"value is {value!r}")
        return value
    raise StatsInputError("WRONG_TYPE", field, f"expected a number, got {type(value).__name__}")


def check_alpha(value: Any, field: str = "alpha") -> float:
    """An error rate strictly inside (0, 1)."""
    v = check_real(value, field)
    if not (0.0 < v < 1.0):
        raise StatsInputError("ALPHA_OUT_OF_RANGE", field, f"alpha {v!r} is not in (0, 1)")
    return v


def check_probability(value: Any, field: str, *, open_interval: bool = False) -> float:
    """A probability in [0, 1] (or (0, 1) when ``open_interval``)."""
    v = check_real(value, field)
    ok = (0.0 < v < 1.0) if open_interval else (0.0 <= v <= 1.0)
    if not ok:
        rng = "(0, 1)" if open_interval else "[0, 1]"
        raise StatsInputError("PROBABILITY_OUT_OF_RANGE", field, f"{v!r} is not in {rng}")
    return v


def check_enum(value: Any, field: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StatsInputError("UNKNOWN_ENUM", field, f"{value!r} is not one of {', '.join(allowed)}")
    return str(value)
