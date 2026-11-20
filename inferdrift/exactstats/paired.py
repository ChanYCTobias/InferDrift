# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired bounds for {-1, 0, +1} residuals and harmful-flip counts.

Each evaluation item i is graded under two runs.  The paired residual d_i is +1,
-1 or 0.  Write p+ = P(d = +1), p- = P(d = -1), pi = p+ + p- (discordance rate),
r = p+/pi (the +1 share of the discordant items) and delta = p+ - p- =
pi (2r - 1).  Items are independent and identically distributed draws.

Methods and derivations
-----------------------
* ``harmful_flip_rate_bound``: n- ~ Binomial(n, p-) exactly, so the
  Clopper-Pearson bound on p- is exact (see ``binomial``).
* ``flip_direction_bound`` / ``mcnemar_exact_pvalue``: conditional on the number
  of discordant items D = n+ + n-, n- ~ Binomial(D, 1 - r) (McNemar,
  Psychometrika 12, 1947; Agresti, Categorical Data Analysis, 3rd ed., section
  11.1).  Clopper-Pearson applied conditionally has conditional coverage
  >= 1 - alpha for every D, hence unconditional coverage >= 1 - alpha.  D = 0
  gives the vacuous interval [0, 1].
* ``paired_mean_bound`` (lower bound on delta; the upper bound is the mirror
  image obtained by exchanging the roles of +1 and -1):
    - [pi_L, pi_U] is the two-sided Clopper-Pearson interval for pi from
      D ~ Binomial(n, pi) at level gamma (gamma/2 per tail), so
      P(pi outside [pi_L, pi_U]) <= gamma;
    - r_L is the one-sided Clopper-Pearson lower bound for r from n+ out of D at
      level alpha - gamma, conditional on D, so P(r < r_L) <= alpha - gamma
      (r_L = 0 when D = 0);
    - on the intersection of the two events, delta = pi (2r - 1) >= pi (2 r_L - 1)
      >= (2 r_L - 1) * (pi_L if 2 r_L - 1 >= 0 else pi_U).
  By the union bound the event fails with probability at most
  gamma + (alpha - gamma) = alpha, so the bound has coverage >= 1 - alpha for
  every (p+, p-).  gamma = nuisance_fraction * alpha is fixed before the data
  are seen (default 0.1).  The method is exact (valid for every parameter value
  and every n, no asymptotics) and conservative.

Domains: counts are integers in 0..MAX_TRIALS (their sum too); alpha in (0, 1)
and nuisance_fraction in (0, 1), every value accepted.  The derived levels
alpha/2, gamma/2 and per_side - gamma are used as float64 numbers while they
are normal; when one would underflow they enter the Clopper-Pearson kernels
through their logarithms, log(gamma/2) = log(nuisance_fraction) +
log(per_side) - log 2 and log(per_side - gamma) = log(per_side) +
log1p(-nuisance_fraction), which are always finite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import _numeric as nm
from ._types import ALTERNATIVES, SIDES, Interval
from ._validate import StatsInputError, check_alpha, check_count, check_enum, check_real, is_bool
from .binomial import MAX_TRIALS, binomial_pvalue

DEFAULT_NUISANCE_FRACTION = 0.1


def residual_counts(residuals: Sequence[int]) -> tuple[int, int, int]:
    """Return (n_pos, n_zero, n_neg) from a list or tuple of integer residuals in {-1, 0, 1}.

    Booleans, floats (even integral ones), strings and values outside {-1, 0, 1}
    are rejected."""
    if not isinstance(residuals, (list, tuple)):
        raise StatsInputError("WRONG_TYPE", "residuals", f"got {type(residuals).__name__}")
    if len(residuals) > MAX_TRIALS:
        raise StatsInputError("COUNT_OUT_OF_RANGE", "residuals", f"more than {MAX_TRIALS} residuals")
    values = []
    for i, v in enumerate(residuals):
        if is_bool(v):
            raise StatsInputError("BOOL_NOT_NUMERIC", f"residuals[{i}]", "booleans are not residuals")
        if isinstance(v, float):
            check_real(v, f"residuals[{i}]")  # NaN/inf -> NONFINITE
            raise StatsInputError("WRONG_TYPE", f"residuals[{i}]", "residuals must be integers")
        if not isinstance(v, int):
            raise StatsInputError("WRONG_TYPE", f"residuals[{i}]", f"got {type(v).__name__}")
        if v not in (-1, 0, 1):
            raise StatsInputError("VALUE_OUT_OF_BOUNDS", f"residuals[{i}]", f"{v} is not in {{-1, 0, 1}}")
        values.append(v)
    return values.count(1), values.count(0), values.count(-1)


def _check_triple(n_pos: object, n_zero: object, n_neg: object) -> tuple[int, int, int]:
    a = check_count(n_pos, "n_pos", maximum=MAX_TRIALS)
    c = check_count(n_zero, "n_zero", maximum=MAX_TRIALS)
    b = check_count(n_neg, "n_neg", maximum=MAX_TRIALS)
    if a + b + c > MAX_TRIALS:
        raise StatsInputError("COUNT_OUT_OF_RANGE", "n_pos+n_zero+n_neg", f"total exceeds {MAX_TRIALS}")
    return a, c, b


def harmful_flip_rate_bound(harmful: int, pairs: int, *, alpha: float, side: str) -> Interval:
    """Exact Clopper-Pearson bound on the harmful-flip rate p- = P(d = -1)."""
    n = check_count(pairs, "pairs", maximum=MAX_TRIALS)
    h = check_count(harmful, "harmful", maximum=MAX_TRIALS)
    if h > n:
        raise StatsInputError("INCONSISTENT_COUNTS", "harmful", f"harmful {h} > pairs {n}")
    alpha = check_alpha(alpha)
    side = check_enum(side, "side", SIDES)
    if side == "lower":
        return Interval(nm.cp_lower(h, n, alpha), 1.0, alpha, side, "harmful_flip_rate_clopper_pearson")
    if side == "upper":
        return Interval(0.0, nm.cp_upper(h, n, alpha), alpha, side, "harmful_flip_rate_clopper_pearson")
    half = alpha / 2.0
    log_h = nm.log_half(alpha)  # finite even when alpha / 2 underflows
    return Interval(nm.cp_lower(h, n, half, log_h), nm.cp_upper(h, n, half, log_h), alpha, side,
                    "harmful_flip_rate_clopper_pearson")


def flip_direction_bound(harmful: int, helpful: int, *, alpha: float, side: str) -> Interval:
    """Exact conditional (McNemar) Clopper-Pearson bound on q = P(harmful | flip)."""
    h = check_count(harmful, "harmful", maximum=MAX_TRIALS)
    g = check_count(helpful, "helpful", maximum=MAX_TRIALS)
    if h + g > MAX_TRIALS:
        raise StatsInputError("COUNT_OUT_OF_RANGE", "harmful+helpful", f"total exceeds {MAX_TRIALS}")
    alpha = check_alpha(alpha)
    side = check_enum(side, "side", SIDES)
    D = h + g
    if side == "lower":
        return Interval(nm.cp_lower(h, D, alpha), 1.0, alpha, side, "flip_direction_conditional_clopper_pearson")
    if side == "upper":
        return Interval(0.0, nm.cp_upper(h, D, alpha), alpha, side, "flip_direction_conditional_clopper_pearson")
    half = alpha / 2.0
    log_h = nm.log_half(alpha)
    return Interval(nm.cp_lower(h, D, half, log_h), nm.cp_upper(h, D, half, log_h), alpha, side,
                    "flip_direction_conditional_clopper_pearson")


def mcnemar_exact_pvalue(harmful: int, helpful: int, *, alternative: str) -> float:
    """Exact conditional McNemar test of H0: P(harmful | flip) = 1/2.

    'greater' tests H1: harmful flips dominate (p-value P(Bin(D, 1/2) >= harmful));
    'less' tests H1: helpful flips dominate; 'two-sided' is the central p-value."""
    h = check_count(harmful, "harmful", maximum=MAX_TRIALS)
    g = check_count(helpful, "helpful", maximum=MAX_TRIALS)
    if h + g > MAX_TRIALS:
        raise StatsInputError("COUNT_OUT_OF_RANGE", "harmful+helpful", f"total exceeds {MAX_TRIALS}")
    alternative = check_enum(alternative, "alternative", ALTERNATIVES)
    return binomial_pvalue(h, h + g, 0.5, alternative=alternative)


def levels(per_side: float, log_per_side: float, frac: float) -> tuple[float, float, float, float]:
    """(gamma/2, its log, per_side - gamma, its log) with gamma = frac * per_side.

    While both float64 levels are normal numbers they are used exactly as
    computed; the union bound then holds for the float values themselves,
    2 (gamma/2) + (per_side - gamma) = per_side.  When either would underflow,
    both come from logarithms of their factors (module doc), which are finite
    for every alpha and frac in (0, 1)."""
    gamma = frac * per_side
    g_half = gamma / 2.0
    rest = per_side - gamma
    if g_half >= nm.DBL_MIN and rest >= nm.DBL_MIN:
        return g_half, math.log(g_half), rest, math.log(rest)
    log_g = math.log(frac) + log_per_side - nm.LOG2
    log_r = log_per_side + math.log1p(-frac)
    return math.exp(log_g), log_g, math.exp(log_r), log_r


def delta_lower(a: int, b: int, n: int, lv: tuple[float, float, float, float]) -> float:
    """Lower confidence bound on delta = p+ - p- (a = n+, b = n-), see the module doc."""
    g_half, log_g, rest, log_r = lv
    D = a + b
    r_lo = nm.cp_lower(a, D, rest, log_r)
    slope = 2.0 * r_lo - 1.0
    if slope >= 0.0:  # only pi_L is used (the library computes both; the value is the same)
        return max(-1.0, slope * nm.cp_lower(D, n, g_half, log_g))
    return max(-1.0, slope * nm.cp_upper(D, n, g_half, log_g))


def paired_mean_bound(n_pos: int, n_zero: int, n_neg: int, *, alpha: float, side: str,
                      nuisance_fraction: float = DEFAULT_NUISANCE_FRACTION) -> Interval:
    """Exact bound on the mean paired residual delta = P(+1) - P(-1)."""
    a, c, b = _check_triple(n_pos, n_zero, n_neg)
    alpha = check_alpha(alpha)
    side = check_enum(side, "side", SIDES)
    frac = check_real(nuisance_fraction, "nuisance_fraction")
    if not (0.0 < frac < 1.0):
        raise StatsInputError("PROBABILITY_OUT_OF_RANGE", "nuisance_fraction", f"{frac!r} is not in (0, 1)")
    n = a + b + c
    if side == "two-sided":
        per_side, log_per_side = alpha / 2.0, nm.log_half(alpha)
    else:
        per_side, log_per_side = alpha, math.log(alpha)
    lv = levels(per_side, log_per_side, frac)
    lower, upper = -1.0, 1.0
    if side in ("lower", "two-sided"):
        lower = delta_lower(a, b, n, lv)
    if side in ("upper", "two-sided"):
        upper = -delta_lower(b, a, n, lv)
    return Interval(lower, upper, alpha, side, "paired_mean_conditional_product")
