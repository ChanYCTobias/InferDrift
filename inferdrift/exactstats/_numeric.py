# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Numerical kernels for the exact bounds (standard library only).

* Binomial tails are summed in log space from exact log-factorials
  (``math.lgamma``), so a small tail keeps its relative precision and no
  complement ``1 - x`` is ever formed.  The sum starts at the largest term of
  the requested range and walks outward; it stops when a geometric bound on
  everything not yet added is below 2**-60 of the running sum, far below the
  rounding error of a float64 sum.
* Roots of monotone functions on [0, 1] are returned as two adjacent floats
  (a, b) with h(a) < 0 <= h(b): the same postcondition as a bisection on the
  IEEE-754 bit pattern.  A safeguarded Newton iteration (falling back to
  bit-pattern bisection whenever a step leaves the bracket or stalls) reaches
  that pair in far fewer evaluations than plain bisection.

Derived tail levels
-------------------
Every alpha in (0, 1) is accepted, including the smallest subnormal 5e-324.
A derived per-tail level such as alpha/2 or nuisance_fraction * alpha can
underflow to a subnormal or to 0.0 in float64 although the level itself is a
positive number.  The kernels therefore take the level as its float64 value
*and* its logarithm: :func:`log_level` returns ``math.log(value)`` whenever the
float value is a normal number and otherwise the logarithm computed from the
level's factors, which is always finite.
"""

from __future__ import annotations

import math
import struct
import sys
from array import array
from collections.abc import Callable

DBL_MIN = sys.float_info.min  # smallest positive normal float64, 2**-1022
LOG2 = math.log(2.0)
_STOP = 2.0 ** -60


def log_level(value: float, log_fallback: float) -> float:
    """log of a tail level whose float64 value is ``value``: ``math.log(value)``
    when ``value`` is a normal float64, otherwise ``log_fallback`` (the logarithm
    of the exact level, computed by the caller from the level's factors)."""
    return math.log(value) if value >= DBL_MIN else log_fallback


def log_half(alpha: float) -> float:
    """log(alpha / 2) for alpha in (0, 1), finite even when alpha / 2 underflows.

    When alpha / 2 is exactly representable it is the log of that exact value,
    so a two-sided level 2a halves to exactly the one-sided level a."""
    half = alpha / 2.0
    if half > 0.0 and half * 2.0 == alpha:  # halving was exact
        return math.log(half)
    return log_level(half, math.log(alpha) - LOG2)


# --------------------------------------------------------------------------
# log-factorial table
# --------------------------------------------------------------------------
_LOGFACT = array("d", [0.0, 0.0])  # _LOGFACT[i] = log(i!) = lgamma(i + 1)


def logfact_table(n: int) -> array:
    """Return a table whose entries 0..n are log(i!), each from ``math.lgamma``."""
    global _LOGFACT
    have = len(_LOGFACT)
    if have <= n:
        new_len = max(n + 1, min(2 * have, 1_000_001))
        lg = math.lgamma
        ext = array("d", (lg(i + 1.0) for i in range(have, new_len)))
        table = array("d", _LOGFACT)
        table.extend(ext)
        _LOGFACT = table
    return _LOGFACT


# --------------------------------------------------------------------------
# binomial tails
# --------------------------------------------------------------------------
def log_binom_pmf(j: int, n: int, p: float) -> float:
    """log P(X = j), X ~ Binomial(n, p), 0 < p < 1, 0 <= j <= n."""
    lf = logfact_table(n)
    return lf[n] - lf[j] - lf[n - j] + j * math.log(p) + (n - j) * math.log1p(-p)


def log_tail_sum(n: int, p: float, lo: int, hi: int) -> float:
    """log of sum_{j=lo..hi} P(X = j), X ~ Binomial(n, p), for 0 < p < 1 and
    0 <= lo <= hi <= n.

    The pmf is unimodal, so the largest term of [lo, hi] is at the mode clipped
    into the range.  Terms are added from there outward; going down from term
    t_{i+1} the ratio t_i / t_{i+1} = (i + 1)(1 - p) / ((n - i) p) decreases as
    i decreases, so once it is below 1 everything not yet added is at most
    t * r / (1 - r) (a geometric series).  A side stops when that remainder is
    below 2**-60 of the running sum (likewise going up)."""
    lf = logfact_table(n)
    lp = math.log(p)
    lq = math.log1p(-p)
    q = 1.0 - p
    m = int((n + 1) * p)
    if m < lo:
        m = lo
    elif m > hi:
        m = hi
    lfn = lf[n]
    base = lfn - lf[m] - lf[n - m] + m * lp + (n - m) * lq
    exp = math.exp
    s = 1.0
    # downward: terms m-1, m-2, ..., lo
    j = m
    while j > lo:
        i = j - 1
        t = exp(lfn - lf[i] - lf[n - i] + i * lp + (n - i) * lq - base)
        s += t
        j = i
        if j > lo:
            r = (i * q) / ((n - i + 1) * p)  # t_{i-1} / t_i
            if r < 1.0 and t * r <= _STOP * s * (1.0 - r):
                break
    # upward: terms m+1, ..., hi
    j = m
    while j < hi:
        i = j + 1
        t = exp(lfn - lf[i] - lf[n - i] + i * lp + (n - i) * lq - base)
        s += t
        j = i
        if j < hi:
            r = ((n - i) * p) / ((i + 1) * q)  # t_{i+1} / t_i
            if r < 1.0 and t * r <= _STOP * s * (1.0 - r):
                break
    return base + math.log(s)


def log_binom_sf(k: int, n: int, p: float) -> float:
    """log P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 0.0
    if k > n:
        return -math.inf
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return 0.0
    return min(0.0, log_tail_sum(n, p, k, n))


def log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p)."""
    if k >= n:
        return 0.0
    if k < 0:
        return -math.inf
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    return min(0.0, log_tail_sum(n, p, 0, k))


# --------------------------------------------------------------------------
# root finding on [0, 1]
# --------------------------------------------------------------------------
def _f2i(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _i2f(i: int) -> float:
    return struct.unpack("<d", struct.pack("<q", i))[0]


def _mid(a: float, b: float) -> float:
    return _i2f((_f2i(a) + _f2i(b)) // 2)


def solve_increasing(h: Callable[[float], float], slope: Callable[[float, float], float | None],
                     lo: float, hi: float, x0: float | None = None) -> tuple[float, float]:
    """For nondecreasing ``h`` with h(lo) < 0 <= h(hi) on 0 <= lo < hi <= 1, return
    adjacent floats (a, b) with h(a) < 0 <= h(b).

    ``slope(x, h(x))`` returns h'(x) (> 0) or None.  A Newton step is taken when it
    lands strictly inside the current bracket and is at most half the previous
    Newton step (so it converges); otherwise the step is a bit-pattern bisection.
    When Newton has converged to within an ulp, the neighbouring float on the
    other side is probed, which pins the adjacent pair."""
    a, b = lo, hi
    ia, ib = _f2i(a), _f2i(b)
    x = x0 if (x0 is not None and a < x0 < b) else _mid(a, b)
    prev_step = math.inf
    newton_budget = 60
    for _ in range(600):
        hx = h(x)
        if hx < 0.0:
            a, ia = x, _f2i(x)
        else:
            b, ib = x, _f2i(x)
        if ib - ia <= 1:
            return a, b
        nxt = None
        if newton_budget > 0 and math.isfinite(hx):
            d = slope(x, hx)
            if d is not None and d > 0.0 and math.isfinite(d):
                step = hx / d
                cand = x - step
                if cand == x:  # converged to within an ulp: probe the neighbour on the other side
                    cand = math.nextafter(x, a if hx >= 0.0 else b)
                if a < cand < b and abs(step) <= 0.5 * prev_step:
                    nxt = cand
                    prev_step = abs(step)
                    newton_budget -= 1
        if nxt is None:
            nxt = _mid(a, b)
            prev_step = math.inf
        x = nxt
    while ib - ia > 1:  # pragma: no cover - the step rules above end long before this
        m = (ia + ib) // 2
        if h(_i2f(m)) < 0.0:
            ia = m
        else:
            ib = m
    return _i2f(ia), _i2f(ib)


def bisect_increasing(h: Callable[[float], float], lo: float, hi: float) -> tuple[float, float]:
    """Plain bit-pattern bisection: for nondecreasing ``h`` with h(lo) < 0 <= h(hi),
    return adjacent floats (a, b) with h(a) < 0 <= h(b).  (Reference path for tests.)"""
    a, b = _f2i(lo), _f2i(hi)
    while b - a > 1:
        m = (a + b) // 2
        if h(_i2f(m)) < 0.0:
            a = m
        else:
            b = m
    return _i2f(a), _i2f(b)


# --------------------------------------------------------------------------
# Clopper-Pearson bounds
# --------------------------------------------------------------------------
def _scaled(n: int, log_value: float) -> float | None:
    """n * exp(log_value), or None when it would overflow (the solver then bisects)."""
    if not log_value < 700.0:
        return None
    return n * math.exp(log_value)


def _z_guess(log_alpha: float) -> float:
    return math.sqrt(max(0.0, -2.0 * log_alpha))


def _cp_lower_parts(k: int, n: int, alpha: float, log_alpha: float | None):
    """(h, slope) defining the CP lower bound as the root of h, on the smaller tail."""
    lf = logfact_table(n)
    if alpha <= 0.5:
        la = math.log(alpha) if log_alpha is None else log_alpha

        def h(p: float) -> float:
            return log_binom_sf(k, n, p) - la

        def slope(p: float, hp: float):
            # d/dp P(X >= k) = n * pmf_{n-1}(k - 1; p);  h = log sf - la, so sf = exp(hp + la)
            if p <= 0.0 or p >= 1.0:
                return None
            lpmf = lf[n - 1] - lf[k - 1] - lf[n - k] + (k - 1) * math.log(p) + (n - k) * math.log1p(-p)
            return _scaled(n, lpmf - (hp + la))
        return h, slope, la
    lq = math.log(1.0 - alpha)

    def h2(p: float) -> float:
        return lq - log_binom_cdf(k - 1, n, p)

    def slope2(p: float, hp: float):
        # d/dp P(X <= k - 1) = -n * pmf_{n-1}(k - 1; p);  log cdf = lq - hp
        if p <= 0.0 or p >= 1.0:
            return None
        lpmf = lf[n - 1] - lf[k - 1] - lf[n - k] + (k - 1) * math.log(p) + (n - k) * math.log1p(-p)
        return _scaled(n, lpmf - (lq - hp))
    return h2, slope2, None


def _cp_upper_parts(k: int, n: int, alpha: float, log_alpha: float | None):
    lf = logfact_table(n)
    if alpha <= 0.5:
        la = math.log(alpha) if log_alpha is None else log_alpha

        def h(p: float) -> float:
            return la - log_binom_cdf(k, n, p)

        def slope(p: float, hp: float):
            # d/dp P(X <= k) = -n * pmf_{n-1}(k; p);  log cdf = la - hp
            if p <= 0.0 or p >= 1.0:
                return None
            lpmf = lf[n - 1] - lf[k] - lf[n - 1 - k] + k * math.log(p) + (n - 1 - k) * math.log1p(-p)
            return _scaled(n, lpmf - (la - hp))
        return h, slope, la
    lq = math.log(1.0 - alpha)

    def h2(p: float) -> float:
        return log_binom_sf(k + 1, n, p) - lq

    def slope2(p: float, hp: float):
        # d/dp P(X >= k + 1) = n * pmf_{n-1}(k; p);  log sf = hp + lq
        if p <= 0.0 or p >= 1.0:
            return None
        lpmf = lf[n - 1] - lf[k] - lf[n - 1 - k] + k * math.log(p) + (n - 1 - k) * math.log1p(-p)
        return _scaled(n, lpmf - (hp + lq))
    return h2, slope2, None


def cp_lower(k: int, n: int, alpha: float, log_alpha: float | None = None) -> float:
    """One-sided Clopper-Pearson lower bound: the p solving P_p(X >= k) = alpha.

    The equation is always solved on the *smaller* tail so the root keeps full
    precision: for alpha <= 1/2 on P_p(X >= k) = alpha, otherwise on the
    equivalent P_p(X <= k - 1) = 1 - alpha (1 - alpha is exact in floating point
    for alpha >= 1/2 by Sterbenz's lemma).  The result is the largest float at
    which the upper tail is still below alpha (rounded outward, i.e. down).

    ``log_alpha`` is log(alpha) supplied by a caller whose level may have
    underflowed in float64 (see :func:`log_level`); it is used on the
    alpha <= 1/2 branch, where only the logarithm of the level enters."""
    if n == 0 or k == 0:
        return 0.0
    h, slope, la = _cp_lower_parts(k, n, alpha, log_alpha)
    phat = k / n
    z = _z_guess(la if la is not None else math.log(1.0 - alpha))
    x0 = phat - (z + 1.0) * math.sqrt(k * (1.0 - phat)) / n - 1.0 / n
    a, _b = solve_increasing(h, slope, 0.0, 1.0, x0 if x0 > 0.0 else None)
    return a


def cp_upper(k: int, n: int, alpha: float, log_alpha: float | None = None) -> float:
    """One-sided Clopper-Pearson upper bound: the p solving P_p(X <= k) = alpha,
    solved on the smaller tail as in :func:`cp_lower` and rounded outward (up).
    ``log_alpha`` as in :func:`cp_lower`."""
    if n == 0 or k == n:
        return 1.0
    h, slope, la = _cp_upper_parts(k, n, alpha, log_alpha)
    phat = k / n
    z = _z_guess(la if la is not None else math.log(1.0 - alpha))
    x0 = phat + (z + 1.0) * math.sqrt(max(k, 1) * (1.0 - phat)) / n + 1.0 / n
    _a, b = solve_increasing(h, slope, 0.0, 1.0, x0 if x0 < 1.0 else None)
    return b


def cp_lower_reference(k: int, n: int, alpha: float, log_alpha: float | None = None) -> float:
    """``cp_lower`` by plain bit-pattern bisection (reference path for tests)."""
    if n == 0 or k == 0:
        return 0.0
    h, _slope, _la = _cp_lower_parts(k, n, alpha, log_alpha)
    return bisect_increasing(h, 0.0, 1.0)[0]


def cp_upper_reference(k: int, n: int, alpha: float, log_alpha: float | None = None) -> float:
    """``cp_upper`` by plain bit-pattern bisection (reference path for tests)."""
    if n == 0 or k == n:
        return 1.0
    h, _slope, _la = _cp_upper_parts(k, n, alpha, log_alpha)
    return bisect_increasing(h, 0.0, 1.0)[1]


def cp_lower_predicate(k: int, n: int, alpha: float, log_alpha: float | None, p: float) -> bool:
    """True iff ``cp_lower(k, n, alpha, log_alpha) >= p`` for a monotone tail, i.e.
    iff the defining function is still below 0 at p (the bisection's own test)."""
    if n == 0 or k == 0:
        return p <= 0.0
    h, _slope, _la = _cp_lower_parts(k, n, alpha, log_alpha)
    return h(p) < 0.0
