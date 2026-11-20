# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The paired criteria's verdict rules on constructed counts."""

from __future__ import annotations

import pytest

from inferdrift import criteria as cr
from inferdrift.exactstats import paired_mean_bound


def C(bad, good, n):
    return cr.Counts(bad, good, n)


def test_residuals_point_the_loss_direction_up():
    assert cr.residual_net(True, False) == 1 and cr.residual_net(False, True) == -1 and cr.residual_net(True, True) == 0
    # h - h': cross-stack harmful flip minus replicate harmful flip
    assert cr.residual_excess(True, False, True) == 1
    assert cr.residual_excess(True, True, False) == -1
    assert cr.residual_excess(True, False, False) == 0
    assert cr.residual_excess(False, True, True) == 0  # helpful flips never offset in harmful_excess


def test_bounds_are_the_library_bounds():
    c = C(7, 3, 90)
    assert cr.upper(c, 0.05) == paired_mean_bound(7, 80, 3, alpha=0.05, side="upper").upper
    assert cr.lower(c, 0.05) == paired_mean_bound(7, 80, 3, alpha=0.05, side="lower").lower


def test_verdict_precedence():
    d = dict(deviates=True, complete=True, all_within=True)
    assert cr.decide(underpowered=True, **d) == cr.UNDERPOWERED
    assert cr.decide(underpowered=False, **d) == cr.DEVIATES
    assert cr.decide(underpowered=False, deviates=False, complete=True, all_within=True) == cr.EQUIVALENT
    assert cr.decide(underpowered=False, deviates=False, complete=False, all_within=True) == cr.INCONCLUSIVE
    assert cr.decide(underpowered=False, deviates=False, complete=True, all_within=False) == cr.INCONCLUSIVE


def test_one_capability_over_tau_blocks_the_pass():
    ok, ubs = cr.final_check({"a": C(0, 0, 400), "b": C(30, 0, 400)}, 0.05, 0.05)
    assert ubs["a"] <= 0.05 < ubs["b"] and not ok
    ok, _ = cr.final_check({"a": C(0, 0, 400), "b": C(0, 0, 400)}, 0.05, 0.05)
    assert ok


def test_bonferroni_over_capabilities_and_looks():
    """A lower bound above tau at alpha / K but not at alpha / (m K) is not DEVIATES."""
    alpha, tau, K = 0.05, 0.05, 4
    hit = None
    for bad in range(10, 80):
        c = C(bad, 0, 400)
        if cr.lower(c, alpha / K) > tau >= cr.lower(c, alpha / (3 * K)):
            hit = c
            break
    assert hit is not None
    assert cr.look_check({"a": hit, "b": C(0, 0, 400), "c": C(0, 0, 400)}, alpha / (3 * K), tau)[0] is False
    assert cr.look_check({"a": hit}, alpha / K, tau)[0] is True


@pytest.mark.parametrize("alpha", [0.05, 0.3])
def test_quick_shortcuts_agree_with_full_evaluation(alpha):
    tau = 0.05
    for n in (60, 200):
        for bad in range(0, 25, 3):
            for good in range(0, 25, 4):
                c = {"a": C(bad, good, n)}
                assert cr.look_check(c, alpha, tau, quick=True)[0] == cr.look_check(c, alpha, tau)[0]
                assert cr.final_check(c, alpha, tau, quick=True)[0] == cr.final_check(c, alpha, tau)[0]


def test_shortcuts_are_off_where_their_premise_fails():
    assert cr.shortcuts_valid(0.05) and cr.shortcuts_valid(0.55)
    assert not cr.shortcuts_valid(0.8)
    c = {"a": C(5, 5, 100)}  # point estimate 0; at alpha 0.8 the lower bound is above it
    assert cr.look_check(c, 0.8, 0.0001, quick=True) == cr.look_check(c, 0.8, 0.0001)


def test_evaluate_stream_rules():
    planned = {"a": 300}
    inc = [{"a": (0, 0, 100)}, {"a": (0, 0, 100)}, {"a": (0, 0, 100)}]
    assert cr.evaluate_stream(inc, planned, 0.05, 0.05)[0] == cr.EQUIVALENT
    bad = [{"a": (60, 0, 100)}, {"a": (0, 0, 100)}, {"a": (0, 0, 100)}]
    assert cr.evaluate_stream(bad, planned, 0.05, 0.05) == (cr.DEVIATES, 0)
    short = [{"a": (0, 0, 100)}, {"a": (0, 0, 100)}]
    assert cr.evaluate_stream(short, planned, 0.05, 0.05)[0] == cr.INCONCLUSIVE
