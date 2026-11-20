# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Result types of the exact statistics.  Immutable and JSON-serialisable."""

from __future__ import annotations

from dataclasses import asdict, dataclass

SIDES = ("lower", "upper", "two-sided")
ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass(frozen=True)
class Interval:
    """A confidence bound or interval for a real parameter.

    ``side='lower'`` is a one-sided lower confidence bound (``upper`` is the
    parameter-space maximum), ``side='upper'`` a one-sided upper bound, and
    ``side='two-sided'`` an equal-tailed interval with alpha/2 in each tail.
    Coverage is at least ``1 - alpha``."""

    lower: float
    upper: float
    alpha: float
    side: str
    method: str

    def to_dict(self) -> dict:
        return asdict(self)
