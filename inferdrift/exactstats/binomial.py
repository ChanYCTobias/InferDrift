# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact binomial tests and Clopper-Pearson bounds.

Derivation
----------
For X ~ Binomial(n, p) the upper tail P_p(X >= k) is continuous and strictly
increasing in p for 1 <= k <= n (it equals the regularised incomplete beta
I_p(k, n - k + 1)).  Inverting the one-sided exact test that rejects
H0: p <= p0 when P_p0(X >= k_obs) <= alpha gives the Clopper-Pearson lower bound
L(k) = inf{p : P_p(X >= k) > alpha}, the root of P_L(X >= k) = alpha, with
L(0) = 0.  Because P_p(P_p(X >= X) <= alpha) <= alpha for every p, the bound
has coverage at least 1 - alpha for every p (Clopper and Pearson, Biometrika 26,
1934; Casella and Berger, Statistical Inference, 2nd ed., Theorem 9.2.12).  The
upper bound is the mirror image with P_p(X <= k).  The two-sided interval is the
intersection of the two one-sided bounds at alpha/2 each (equal tails), so its
coverage is at least 1 - alpha by the union bound.

Conventions
-----------
* Bounds: ``side`` is 'lower', 'upper' or 'two-sided' (alpha/2 per tail).
* Tests: ``alternative`` is 'greater' (H1: p > p0, p-value P(X >= k)), 'less'
  (H1: p < p0, p-value P(X <= k)) or 'two-sided', here the *central* (doubled
  one-sided) p-value min(1, 2 min(P(X >= k), P(X <= k))), the convention dual to
  the two-sided Clopper-Pearson interval.
* n = 0 carries no information: every p-value is 1 and every bound is [0, 1].

Domains: counts are integers in 0..MAX_TRIALS with successes <= trials; p0 in
[0, 1]; alpha in (0, 1), every value accepted (the two-sided per-tail level
alpha/2 enters through its logarithm, so it never underflows to zero).
"""

from __future__ import annotations

import math

from . import _numeric as nm
from ._types import ALTERNATIVES, SIDES, Interval
from ._validate import StatsInputError, check_alpha, check_count, check_enum, check_probability

MAX_TRIALS = 1_000_000  # computational cap


def _check_kn(successes: object, trials: object) -> tuple[int, int]:
    n = check_count(trials, "trials", maximum=MAX_TRIALS)
    k = check_count(successes, "successes", maximum=MAX_TRIALS)
    if k > n:
        raise StatsInputError("INCONSISTENT_COUNTS", "successes", f"successes {k} > trials {n}")
    return k, n


def binomial_pvalue(successes: int, trials: int, p0: float, *, alternative: str) -> float:
    """Exact binomial test p-value for H0: p = p0 (one-sided or central two-sided)."""
    k, n = _check_kn(successes, trials)
    p0 = check_probability(p0, "p0")
    alternative = check_enum(alternative, "alternative", ALTERNATIVES)
    if n == 0:
        return 1.0
    sf = math.exp(nm.log_binom_sf(k, n, p0))
    cdf = math.exp(nm.log_binom_cdf(k, n, p0))
    if alternative == "greater":
        return sf
    if alternative == "less":
        return cdf
    return min(1.0, 2.0 * min(sf, cdf))


def clopper_pearson(successes: int, trials: int, *, alpha: float, side: str) -> Interval:
    """Exact Clopper-Pearson confidence bound(s) for a binomial proportion."""
    k, n = _check_kn(successes, trials)
    alpha = check_alpha(alpha)
    side = check_enum(side, "side", SIDES)
    if side == "lower":
        return Interval(nm.cp_lower(k, n, alpha), 1.0, alpha, side, "clopper_pearson")
    if side == "upper":
        return Interval(0.0, nm.cp_upper(k, n, alpha), alpha, side, "clopper_pearson")
    half = alpha / 2.0
    log_h = nm.log_half(alpha)  # finite even when alpha / 2 underflows (alpha subnormal)
    return Interval(nm.cp_lower(k, n, half, log_h), nm.cp_upper(k, n, half, log_h), alpha, side, "clopper_pearson")
