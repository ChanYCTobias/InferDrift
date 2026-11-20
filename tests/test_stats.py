# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounds and verdict rules, checked against closed forms and exact rational arithmetic."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from inferdrift.plan import DEVIATES, EQUIVALENT, INCONCLUSIVE, Plan, final_decision, look_decision, look_schedule
from inferdrift.stats import StatsBackend, cp_lower, cp_upper


def exact_cdf(k: int, n: int, p: float) -> Fraction:
    """P(X <= k) for X ~ Binomial(n, p), with p taken exactly as the float it is."""
    q = Fraction(p)
    return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k + 1))


def exact_sf(k: int, n: int, p: float) -> Fraction:
    """P(X >= k)."""
    q = Fraction(p)
    return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k, n + 1))


@pytest.mark.parametrize("n", [1, 2, 7, 32, 100, 500, 1000])
@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.1])
def test_closed_forms_at_the_edges(n, alpha):
    assert cp_upper(0, n, alpha) == pytest.approx(1.0 - alpha ** (1.0 / n), rel=1e-12)
    assert cp_lower(n, n, alpha) == pytest.approx(alpha ** (1.0 / n), rel=1e-12)
    assert cp_lower(0, n, alpha) == 0.0
    assert cp_upper(n, n, alpha) == 1.0


def test_bounds_solve_their_defining_equations_exactly():
    """UB: P(X <= k | UB) = alpha; LB: P(X >= k | LB) = alpha (checked in exact rational arithmetic)."""
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(2, 60)
        k = rng.randint(1, n - 1)
        alpha = rng.choice([0.05, 0.01, 0.1, 0.025])
        ub, lb = cp_upper(k, n, alpha), cp_lower(k, n, alpha)
        assert 0.0 < lb < k / n < ub < 1.0
        # outward rounding keeps coverage: at the bound the tail is at most alpha (up to float log-sum error)
        assert float(exact_cdf(k, n, ub)) <= alpha * (1 + 1e-9)
        assert float(exact_sf(k, n, lb)) <= alpha * (1 + 1e-9)
        # and the bound is tight: slightly inside it, the tail exceeds alpha
        assert float(exact_cdf(k, n, ub - 1e-7)) > alpha
        assert float(exact_sf(k, n, lb + 1e-7)) > alpha


def test_monotone_in_counts():
    for n in (32, 128, 200):
        uppers = [cp_upper(k, n, 0.05) for k in range(n + 1)]
        lowers = [cp_lower(k, n, 0.05) for k in range(n + 1)]
        assert all(a < b for a, b in zip(uppers, uppers[1:]))
        assert all(a < b for a, b in zip(lowers, lowers[1:]))


def test_invalid_counts_refused():
    with pytest.raises(ValueError):
        cp_upper(5, 4, 0.05)
    with pytest.raises(TypeError):
        cp_upper(True, 4, 0.05)
    with pytest.raises(ValueError):
        cp_lower(1, 4, 1.5)


def test_backend_modes():
    assert StatsBackend("auto").name == StatsBackend("builtin").name == "builtin exact binomial"
    with pytest.raises(ValueError):
        StatsBackend("other")


def test_accuracy_change_bound_brackets_point():
    b = StatsBackend()
    for g, z, h in [(0, 500, 0), (10, 470, 20), (3, 0, 0), (0, 0, 5), (40, 400, 60)]:
        bd = b.accuracy_change(g, z, h, 0.05)
        assert -1.0 <= bd.lower <= bd.point <= bd.upper <= 1.0


def test_excess_rate_bounds_are_wider_than_paired():
    b = StatsBackend()
    h, n = 2, 64
    assert b.excess_rate(h, n, 0, n, 0.05).upper > b.rate_upper(h, n, 0.05)


def test_look_schedule():
    assert look_schedule(500) == [32, 64, 128, 256, 500]
    assert look_schedule(64) == [32, 64]
    assert look_schedule(10) == [10]


def test_verdict_rules():
    b = StatsBackend()
    plan = Plan(300, 0.01, 0.05)
    # zero harmful flips in 300 items: upper bound 1 - 0.05 ** (1/300) < 0.01
    assert final_decision(b, plan, 300, 0, reached_n_max=True, any_deviates=False)[0] == EQUIVALENT
    # the same counts without reaching n_max are not enough
    assert final_decision(b, plan, 299, 0, reached_n_max=False, any_deviates=False)[0] == INCONCLUSIVE
    # a DEVIATES look decides the run
    assert final_decision(b, plan, 300, 0, reached_n_max=True, any_deviates=True)[0] == DEVIATES
    deviates, lb, _ = look_decision(b, plan, 0, 32, 10)
    assert deviates and lb > 0.01
    deviates, lb, _ = look_decision(b, plan, 0, 32, 0)
    assert not deviates and lb == 0.0


def test_sequential_error_rates_at_the_boundary():
    """Simulated error rates of the sequential rule stay below alpha at p = tau (seeded, so deterministic)."""
    b = StatsBackend()
    plan = Plan(500, 0.01, 0.05)
    dev_min = {nk: next(k for k in range(nk + 1) if b.rate_lower(k, nk, plan.alpha_per_look) > 0.01)
               for nk in plan.looks}
    eq_max = max(k for k in range(501) if b.rate_upper(k, 500, 0.05) <= 0.01)
    rng = random.Random(11)
    runs = 1500
    false_dev = false_eq = 0
    for _ in range(runs):
        flips = [rng.random() < 0.01 for _ in range(500)]
        false_dev += any(sum(flips[:nk]) >= dev_min[nk] for nk in plan.looks)
    for _ in range(runs):
        flips = [rng.random() < 0.0101 for _ in range(500)]
        dev = any(sum(flips[:nk]) >= dev_min[nk] for nk in plan.looks)
        false_eq += (not dev) and sum(flips) <= eq_max
    assert false_dev / runs <= 0.05
    assert false_eq / runs <= 0.05
