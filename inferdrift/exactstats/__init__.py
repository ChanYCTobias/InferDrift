# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact binomial and paired bounds (standard library only).

* ``clopper_pearson`` and ``binomial_pvalue``: exact one-sided and equal-tailed
  bounds and tests for a binomial proportion;
* ``paired_mean_bound``: an exact bound on the mean of {-1, 0, +1} paired
  residuals (net change), from a two-sided bound on the discordance rate and a
  conditional bound on the direction of the discordant items, combined by the
  union bound;
* ``harmful_flip_rate_bound``, ``flip_direction_bound`` and
  ``mcnemar_exact_pvalue``: the related exact bounds and test.

Every bound is valid for every parameter value and every sample size (no
asymptotics) and conservative.  Each module documents its derivation.
"""

from __future__ import annotations

from ._types import Interval
from ._validate import REASON_CODES, StatsInputError
from .binomial import MAX_TRIALS, binomial_pvalue, clopper_pearson
from .paired import (DEFAULT_NUISANCE_FRACTION, flip_direction_bound, harmful_flip_rate_bound, mcnemar_exact_pvalue,
                     paired_mean_bound, residual_counts)

__all__ = ["DEFAULT_NUISANCE_FRACTION", "Interval", "MAX_TRIALS", "REASON_CODES", "StatsInputError",
           "binomial_pvalue", "clopper_pearson", "flip_direction_bound", "harmful_flip_rate_bound",
           "mcnemar_exact_pvalue", "paired_mean_bound", "residual_counts"]
