# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The exact statistics package: closed forms, defining equations in exact rational arithmetic,
coverage by enumeration, boundary cases and input refusals."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from inferdrift import stats as stats010
from inferdrift.exactstats import (StatsInputError, binomial_pvalue, clopper_pearson, flip_direction_bound,
                                   harmful_flip_rate_bound, mcnemar_exact_pvalue, paired_mean_bound, residual_counts)
from inferdrift.exactstats import _numeric as nm


def exact_cdf(k: int, n: int, p: float) -> Fraction:
    q = Fraction(p)
    return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k + 1))


def exact_sf(k: int, n: int, p: float) -> Fraction:
    q = Fraction(p)
    return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k, n + 1))


@pytest.mark.parametrize("n", [1, 2, 7, 32, 100, 500])
@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.0025])
def test_closed_forms_at_the_edges(n, alpha):
    assert clopper_pearson(0, n, alpha=alpha, side="upper").upper == pytest.approx(1.0 - alpha ** (1.0 / n), rel=1e-12)
    assert clopper_pearson(n, n, alpha=alpha, side="lower").lower == pytest.approx(alpha ** (1.0 / n), rel=1e-12)
    assert clopper_pearson(0, n, alpha=alpha, side="lower").lower == 0.0
    assert clopper_pearson(n, n, alpha=alpha, side="upper").upper == 1.0
    assert clopper_pearson(0, 0, alpha=alpha, side="two-sided").to_dict()["lower"] == 0.0


def test_defining_equations_hold_exactly_and_are_tight():
    rng = random.Random(7)
    for _ in range(40):
        n = rng.randint(2, 60)
        k = rng.randint(1, n - 1)
        alpha = rng.choice([0.05, 0.01, 0.1, 0.0025])
        ub = clopper_pearson(k, n, alpha=alpha, side="upper").upper
        lb = clopper_pearson(k, n, alpha=alpha, side="lower").lower
        assert 0.0 < lb < k / n < ub < 1.0
        assert float(exact_cdf(k, n, ub)) <= alpha * (1 + 1e-9)
        assert float(exact_sf(k, n, lb)) <= alpha * (1 + 1e-9)
        assert float(exact_cdf(k, n, ub - 1e-7)) > alpha
        assert float(exact_sf(k, n, lb + 1e-7)) > alpha


def test_fast_root_finder_matches_plain_bisection_and_the_010_bounds():
    rng = random.Random(3)
    worst_ref = worst_010 = 0.0
    for _ in range(300):
        n = rng.choice([1, 3, 10, 64, 300, 2000])
        k = rng.randint(0, n)
        a = rng.choice([0.05, 0.0025, 0.3, 0.6, 0.9, 1e-12])
        lo, up = nm.cp_lower(k, n, a), nm.cp_upper(k, n, a)
        worst_ref = max(worst_ref, abs(lo - nm.cp_lower_reference(k, n, a)), abs(up - nm.cp_upper_reference(k, n, a)))
        if n <= 300:
            worst_010 = max(worst_010, abs(lo - stats010.cp_lower(k, n, a)), abs(up - stats010.cp_upper(k, n, a)))
    assert worst_ref <= 1e-13
    assert worst_010 <= 1e-9


def test_truncated_tail_sums_agree_with_full_sums():
    rng = random.Random(5)
    for _ in range(200):
        n = rng.randint(1, 3000)
        k = rng.randint(0, n)
        p = rng.choice([1e-6, 0.001, 0.03, 0.3, 0.5, 0.9, 0.999])
        if k >= n:
            continue
        logs = [nm.log_binom_pmf(j, n, p) for j in range(k + 1)]
        top = max(logs)
        full_cdf = top + math.log(math.fsum(math.exp(x - top) for x in logs))
        assert nm.log_binom_cdf(k, n, p) == pytest.approx(min(0.0, full_cdf), abs=1e-12, rel=1e-12)


def test_paired_bound_boundary_cases():
    r = paired_mean_bound(0, 0, 0, alpha=0.05, side="two-sided")
    assert (r.lower, r.upper) == (-1.0, 1.0)  # n = 0 is vacuous
    r = paired_mean_bound(0, 50, 0, alpha=0.05, side="two-sided")
    assert r.lower == pytest.approx(-r.upper) and r.lower < 0 < r.upper
    lo = paired_mean_bound(30, 0, 0, alpha=0.05, side="lower").lower
    up = paired_mean_bound(0, 0, 30, alpha=0.05, side="upper").upper
    assert lo == pytest.approx(-up) and lo > 0  # mirror symmetry


@pytest.mark.parametrize("counts", [(7, 80, 3), (0, 10, 0), (12, 5, 9), (1, 0, 1), (40, 900, 25)])
@pytest.mark.parametrize("alpha", [0.05, 0.2, 1e-300, 5e-324 * 2])
def test_two_sided_at_2alpha_equals_the_one_sided_halves(counts, alpha):
    a, c, b = counts
    two = paired_mean_bound(a, c, b, alpha=2 * alpha, side="two-sided")
    assert two.lower == paired_mean_bound(a, c, b, alpha=alpha, side="lower").lower
    assert two.upper == paired_mean_bound(a, c, b, alpha=alpha, side="upper").upper


def test_paired_bound_is_monotone_in_the_direction_of_the_discordant_items():
    """At a fixed number D of discordant items, turning a -1 into a +1 never lowers the lower bound.
    (Turning a 0 into a +1 can: it also widens the bound on the discordance.)"""
    for n in (40, 200):
        for D in (0, 1, 7, 20):
            lbs = [paired_mean_bound(a, n - D, D - a, alpha=0.05, side="lower").lower for a in range(D + 1)]
            ubs = [paired_mean_bound(a, n - D, D - a, alpha=0.05, side="upper").upper for a in range(D + 1)]
            assert all(x <= y for x, y in zip(lbs, lbs[1:]))
            assert all(x <= y for x, y in zip(ubs, ubs[1:]))


_BOUNDS: dict = {}


def _bound(a: int, c: int, b: int, alpha: float, side: str):
    key = (a, c, b, alpha, side)
    if key not in _BOUNDS:
        _BOUNDS[key] = paired_mean_bound(a, c, b, alpha=alpha, side=side)
    return _BOUNDS[key]


def _trinomial_coverage(n: int, p_plus: float, p_minus: float, alpha: float, side: str) -> float:
    theta = p_plus - p_minus
    p0 = 1.0 - p_plus - p_minus
    total = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            w = math.comb(n, a) * math.comb(n - a, b) * p_plus ** a * p_minus ** b * p0 ** c
            r = _bound(a, c, b, alpha, side)
            if (r.lower <= theta) if side == "lower" else (r.upper >= theta):
                total.append(w)
    return math.fsum(total)


@pytest.mark.parametrize("n", [5, 10, 15])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_exact_coverage_by_enumeration(n, alpha):
    for p_plus in (0.0, 0.05, 0.3, 0.6):
        for p_minus in (0.0, 0.1, 0.35):
            if p_plus + p_minus > 1:
                continue
            for side in ("lower", "upper"):
                assert _trinomial_coverage(n, p_plus, p_minus, alpha, side) >= 1 - alpha - 1e-12


def test_shortcut_premise_fails_above_one_half():
    """LB <= point estimate holds while the share level is at most 1/2; at alpha 0.8 it does not."""
    assert paired_mean_bound(5, 90, 5, alpha=0.05, side="lower").lower <= 0.0
    assert paired_mean_bound(5, 90, 5, alpha=0.8, side="lower").lower > 0.0


def test_related_bounds_and_tests():
    assert harmful_flip_rate_bound(3, 100, alpha=0.05, side="upper").upper == \
        clopper_pearson(3, 100, alpha=0.05, side="upper").upper
    assert flip_direction_bound(0, 0, alpha=0.05, side="two-sided").to_dict()["upper"] == 1.0
    assert mcnemar_exact_pvalue(10, 0, alternative="greater") == pytest.approx(0.5 ** 10)
    assert binomial_pvalue(0, 0, 0.5, alternative="two-sided") == 1.0
    assert residual_counts([1, 0, -1, 1]) == (2, 1, 1)


@pytest.mark.parametrize("call,reason", [
    (lambda: residual_counts([True]), "BOOL_NOT_NUMERIC"),
    (lambda: residual_counts([1.0]), "WRONG_TYPE"),
    (lambda: residual_counts([2]), "VALUE_OUT_OF_BOUNDS"),
    (lambda: residual_counts([float("nan")]), "NONFINITE"),
    (lambda: paired_mean_bound(-1, 5, 5, alpha=0.05, side="lower"), "NEGATIVE_COUNT"),
    (lambda: paired_mean_bound(1.0, 5, 5, alpha=0.05, side="lower"), "WRONG_TYPE"),
    (lambda: paired_mean_bound(True, 5, 5, alpha=0.05, side="lower"), "BOOL_NOT_NUMERIC"),
    (lambda: paired_mean_bound(1, 5, 5, alpha=float("nan"), side="lower"), "NONFINITE"),
    (lambda: paired_mean_bound(1, 5, 5, alpha=1.0, side="lower"), "ALPHA_OUT_OF_RANGE"),
    (lambda: paired_mean_bound(1, 5, 5, alpha=0.05, side="left"), "UNKNOWN_ENUM"),
    (lambda: paired_mean_bound(1, 5, 5, alpha=0.05, side="lower", nuisance_fraction=1.0), "PROBABILITY_OUT_OF_RANGE"),
    (lambda: clopper_pearson(5, 4, alpha=0.05, side="upper"), "INCONSISTENT_COUNTS"),
    (lambda: binomial_pvalue(1, 4, 1.5, alternative="less"), "PROBABILITY_OUT_OF_RANGE"),
])
def test_out_of_domain_inputs_are_refused(call, reason):
    with pytest.raises(StatsInputError) as e:
        call()
    assert e.value.reason == reason
