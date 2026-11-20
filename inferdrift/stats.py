# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired one-sided bounds: harmful-flip rate and accuracy change.

Model.  Items are independent draws from the task distribution the bank
represents.  For item i both stacks are graded; d_i = (candidate correct) -
(reference correct) is +1 (helpful flip), -1 (harmful flip) or 0.  With
p- = P(d = -1) and p+ = P(d = +1):

* the harmful-flip count h over n items is Binomial(n, p-), so the
  Clopper-Pearson bounds on p- are exact (coverage >= 1 - alpha for every p-);
* the accuracy change is delta = p+ - p-.

The bounds are computed here, with the standard library only:

* the Clopper-Pearson bounds are found by bisection on exactly summed log
  tails and rounded outward (the closed forms for k = 0 and k = n are used
  directly; the tests check both against them);
* for delta the union bound LB(p+, alpha/2) - UB(p-, alpha/2) (and its
  mirror) is used, which is valid and conservative.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

_TINY_LOG = -745.0


@lru_cache(maxsize=64)
def _logfact(n: int) -> tuple[float, ...]:
    out = [0.0] * (n + 1)
    for i in range(2, n + 1):
        out[i] = out[i - 1] + math.log(i)
    return tuple(out)


def _log_pmf_terms(n: int, p: float, lo: int, hi: int) -> list[float]:
    lf = _logfact(n)
    if p <= 0.0:
        return [0.0 if j == 0 else -math.inf for j in range(lo, hi + 1)]
    if p >= 1.0:
        return [0.0 if j == n else -math.inf for j in range(lo, hi + 1)]
    lp, lq = math.log(p), math.log1p(-p)
    return [lf[n] - lf[j] - lf[n - j] + j * lp + (n - j) * lq for j in range(lo, hi + 1)]


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def log_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k), X ~ Binomial(n, p), summed term by term."""
    if k >= n:
        return 0.0
    if k < 0:
        return -math.inf
    return _logsumexp(_log_pmf_terms(n, p, 0, k))


def log_sf(k: int, n: int, p: float) -> float:
    """log P(X >= k)."""
    if k <= 0:
        return 0.0
    if k > n:
        return -math.inf
    return _logsumexp(_log_pmf_terms(n, p, k, n))


def _bisect(f, increasing: bool, target: float) -> tuple[float, float]:
    """Return (lo, hi) bracketing the root of f(p) = target on [0, 1]."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        v = f(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return lo, hi


def cp_upper(k: int, n: int, alpha: float) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound for a binomial proportion."""
    _check(k, n, alpha)
    if n == 0 or k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    target = math.log(alpha)
    # P_p(X <= k) decreases in p; the bound is the root of log_cdf = log(alpha).
    lo, hi = _bisect(lambda p: log_cdf(k, n, p), increasing=False, target=target)
    return hi  # round outward (conservative)


def cp_lower(k: int, n: int, alpha: float) -> float:
    """One-sided (1 - alpha) Clopper-Pearson lower bound."""
    _check(k, n, alpha)
    if n == 0 or k == 0:
        return 0.0
    if k == n:
        return math.exp(math.log(alpha) / n)
    target = math.log(alpha)
    lo, hi = _bisect(lambda p: log_sf(k, n, p), increasing=True, target=target)
    return lo  # round outward (conservative)


def _check(k: int, n: int, alpha: float) -> None:
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("counts must be integers")
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts k={k} n={n}")
    if not (isinstance(alpha, float) or isinstance(alpha, int)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


@dataclass(frozen=True)
class Bound:
    point: float | None
    lower: float
    upper: float
    alpha: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


class StatsBackend:
    """The built-in exact binomial.  ``mode`` is ``'auto'`` or ``'builtin'``; both select it."""

    def __init__(self, mode: str = "auto") -> None:
        if mode not in ("auto", "builtin"):
            raise ValueError(f"unknown stats mode {mode!r}")
        self.name = "builtin exact binomial"

    # -- harmful (or any) flip rate --------------------------------------------------
    def rate_upper(self, k: int, n: int, alpha: float) -> float:
        return cp_upper(k, n, alpha)

    def rate_lower(self, k: int, n: int, alpha: float) -> float:
        return cp_lower(k, n, alpha)

    def rate_bound(self, k: int, n: int, alpha: float) -> Bound:
        """Two one-sided (1 - alpha) bounds (each separately valid, not a joint interval)."""
        return Bound(
            point=(k / n) if n else None,
            lower=self.rate_lower(k, n, alpha),
            upper=self.rate_upper(k, n, alpha),
            alpha=alpha,
            method="builtin Clopper-Pearson",
        )

    # -- accuracy change ---------------------------------------------------------------
    def accuracy_change(self, n_pos: int, n_zero: int, n_neg: int, alpha: float) -> Bound:
        """One-sided (1 - alpha) lower and upper bounds on delta = p+ - p-."""
        n = n_pos + n_zero + n_neg
        point = ((n_pos - n_neg) / n) if n else None
        if n == 0:
            return Bound(point, -1.0, 1.0, alpha, "empty")
        half = alpha / 2.0
        lo = cp_lower(n_pos, n, half) - cp_upper(n_neg, n, half)
        hi = cp_upper(n_pos, n, half) - cp_lower(n_neg, n, half)
        return Bound(point, max(-1.0, lo), min(1.0, hi), alpha, "builtin union of Clopper-Pearson bounds")

    # -- independent sampling: excess harmful-flip rate over a noise floor ------------
    def excess_rate(self, h_rc: int, n_rc: int, h_rr: int, n_rr: int, alpha: float) -> Bound:
        """Bounds on p_rc - p_rr (cross-stack harmful-flip rate minus the reference's
        own replicate harmful-flip rate), each side at alpha by the union of two
        alpha/2 Clopper-Pearson bounds.  Exact and conservative."""
        half = alpha / 2.0
        lo = self.rate_lower(h_rc, n_rc, half) - self.rate_upper(h_rr, n_rr, half)
        hi = self.rate_upper(h_rc, n_rc, half) - self.rate_lower(h_rr, n_rr, half)
        point = (h_rc / n_rc - h_rr / n_rr) if (n_rc and n_rr) else None
        return Bound(point, max(-1.0, lo), min(1.0, hi), alpha, "union of Clopper-Pearson bounds (independent sampling)")
