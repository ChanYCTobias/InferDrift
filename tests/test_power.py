# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The exact power planner."""

from __future__ import annotations

import math
import random

import pytest

from inferdrift import criteria as cr
from inferdrift import power as pw
from inferdrift.exactstats import paired_mean_bound


@pytest.mark.parametrize("tau", [0.01, 0.02, 0.05, 0.1])
def test_zero_floor_size_is_the_closed_form(tau):
    n_star = math.ceil(math.log(0.05 * 0.1 / 2) / math.log(1 - tau))
    r = pw.items_needed({"a": 1}, {"a": pw.net_change_model(0.0)}, 0.05, tau, 0.9)
    assert r["items_needed"] == n_star
    assert pw.pass_probability(n_star, pw.net_change_model(0.0), 0.05, tau) == 1.0
    assert pw.pass_probability(n_star - 1, pw.net_change_model(0.0), 0.05, tau) == 0.0


def test_analytic_inversion_matches_a_direct_search_with_the_production_bound():
    rng = random.Random(3)
    for _ in range(150):
        n = rng.randint(1, 400)
        D = rng.randint(0, n)
        alpha = rng.choice([0.01, 0.05])
        tau = rng.choice([0.01, 0.05, 0.1])
        g = pw.critical_gains(D, n, alpha, tau)
        direct = next((x for x in range(D + 1)
                       if paired_mean_bound(D - x, n - D, x, alpha=alpha, side="upper").upper <= tau), D + 1)
        assert g == direct, (n, D, alpha, tau)


@pytest.mark.parametrize("model,m", [(pw.net_change_model(0.05), 1), (pw.net_change_model(0.1, 0.01), 1),
                                     (pw.harmful_excess_model(0.1), 1), (pw.net_change_model(0.05), 3)])
def test_items_needed_is_powered_and_one_less_is_not(model, m):
    weights = {f"c{i}": 1 for i in range(m)}
    models = {c: model for c in weights}
    r = pw.items_needed(weights, models, 0.05, 0.15, 0.9)
    n = r["items_needed"]
    assert pw.power_total(pw.allocate(n, weights), models, 0.05, 0.15) >= 0.9
    assert pw.power_total(pw.allocate(n - 1, weights), models, 0.05, 0.15) < 0.9


def test_items_needed_grows_with_the_floor():
    sizes = [pw.items_needed({"a": 1}, {"a": pw.net_change_model(f)}, 0.05, 0.1, 0.9)["items_needed"]
             for f in (0.0, 0.05, 0.1, 0.2)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_harmful_excess_model_has_mean_equal_to_the_excess():
    for f in (0.02, 0.1, 0.3):
        for ex in (-0.005, 0.0, 0.01):
            m = pw.harmful_excess_model(f, ex)
            assert m.p_loss - m.p_gain == pytest.approx(ex, abs=1e-15)


def test_exact_power_agrees_with_a_simulation_through_the_verdict_rule():
    rng = random.Random(11)
    n, model, alpha, tau = 300, pw.net_change_model(0.1), 0.05, 0.1
    exact = pw.pass_probability(n, model, alpha, tau)
    reps, hits = 300, 0
    for _ in range(reps):
        bad = good = 0
        for _ in range(n):
            u = rng.random()
            bad += u < model.p_loss
            good += model.p_loss <= u < model.p_loss + model.p_gain
        verdict, _ = cr.evaluate_stream([{"a": (bad, good, n)}], {"a": n}, alpha, tau)
        hits += verdict == cr.EQUIVALENT
    se = math.sqrt(exact * (1 - exact) / reps)
    assert abs(hits / reps - exact) <= 4 * se + 1e-9


def test_smallest_tau_brackets_the_power():
    counts, models = {"a": 400}, {"a": pw.net_change_model(0.05)}
    t = pw.smallest_tau(counts, models, 0.05, 0.9)
    assert pw.power_total(counts, models, 0.05, t) >= 0.9
    assert pw.power_total(counts, models, 0.05, t - 1e-4) < 0.9


def test_models_refuse_impossible_parameters():
    from inferdrift.errors import BadInputError

    with pytest.raises(BadInputError):
        pw.net_change_model(0.1, -0.2)
    with pytest.raises(BadInputError):
        pw.harmful_excess_model(1.5)


def test_a_size_with_unpowered_sizes_just_above_it_is_not_reported():
    """Exact power is a sawtooth: at floor 0.02 and tau 0.03, 675 items reach power 0.9 while 681-684
    and 691 fall below it again; the reported number is above all of them."""
    m = {"a": pw.net_change_model(0.02)}
    assert pw.powered({"a": 675}, m, 0.05, 0.03, 0.9) and not pw.powered({"a": 683}, m, 0.05, 0.03, 0.9)
    r = pw.items_needed({"a": 1}, m, 0.05, 0.03, 0.9)
    assert r["items_needed"] == 692 and r["checked_up_to"] == 742


@pytest.mark.parametrize("crit,floor,tau,m", [("net_change", 0.02, 0.03, 1), ("net_change", 0.03, 0.02, 1),
                                              ("harmful_excess", 0.1, 0.05, 1), ("net_change", 0.05, 0.05, 3),
                                              ("net_change", 0.0, 0.05, 2)])
def test_items_needed_is_checked_above_and_does_not_depend_on_where_the_search_starts(crit, floor, tau, m):
    weights = {f"c{i}": 1 for i in range(m)}
    models = {c: pw.model_for(crit, floor) for c in weights}
    r = pw.items_needed(weights, models, 0.05, tau, 0.9)
    n, top = r["items_needed"], r["checked_up_to"]
    assert top == n + pw.check_window(n) and pw.check_window(n) >= 50
    assert all(pw.powered(pw.allocate(k, weights), models, 0.05, tau, 0.9) for k in range(n, top + 1))
    assert not pw.powered(pw.allocate(n - 1, weights), models, 0.05, tau, 0.9)
    for above in (n - 1, n - 10, n // 2, 1):  # compare searches above its planned count
        assert pw.items_needed(weights, models, 0.05, tau, 0.9, above=above)["items_needed"] == n, above


def test_a_capped_search_states_where_it_stopped():
    models = {c: pw.net_change_model(0.0) for c in ("big", "tiny")}
    r = pw.items_needed({"big": 999, "tiny": 1}, models, 0.05, 0.01, 0.9, cap_per_capability=20_000)
    assert r["items_needed"] is None and r["exceeds"] == 20_000 and r["searched_to"] == 20_020
    assert r["per_capability_at_stop"] == {"big": 20_000, "tiny": 20} and r["largest_capability"] == "big"
    assert r["power_at_stop"] == 0.0 and r["note"].startswith("not powered at 20020 items")


def test_items_needed_in_order_uses_the_counts_of_the_order_itself():
    order = ["a"] * 150 + ["b"] * 150 + ["c"] * 150
    models = {c: pw.net_change_model(0.0) for c in "abc"}
    r = pw.items_needed_in_order(order, models, 0.05, 0.05, 0.9, above=340)
    assert r["items_needed"] == 417 and r["per_capability"] == {"a": 150, "b": 150, "c": 117}
    assert r["checked_up_to"] == 450  # the check reaches the end of the order
    assert pw.items_needed_in_order(order[:400], models, 0.05, 0.05, 0.9, above=340)["items_needed"] is None
