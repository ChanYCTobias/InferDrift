# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Noise routing, the power gate and the known-wrong cases, against stand-ins that emulate
deterministic and noisy engines (127.0.0.1 only, no model, no network)."""

from __future__ import annotations

import json

import pytest

from helpers import (Standins, correct_output, items_of, make_single_suite_bank, pick, run_main, url,
                     write_bank, write_noisy_replay, write_replay, wrong_output)
from inferdrift import compare as cmp
from inferdrift import determinism
from inferdrift import power as pw
from inferdrift.adapters.base import GenResult

N = 300
CRITERIA = ("net_change", "harmful_excess", "harmful_abs")


@pytest.fixture
def standins():
    s = Standins()
    yield s
    s.close()


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """One capability of 300 items; a deterministic replay; a noisy one (40% of items volatile, each
    request right or wrong with probability 1/2: answer-level floor 0.2); and fault variants."""
    d = tmp_path_factory.mktemp("noise")
    items = make_single_suite_bank(N)
    vol = pick(items, 0.4, 1)
    harm = pick(items, 0.4, 2, exclude=vol)  # stable items made wrong in the harmful candidate
    f = {"items": items, "bank": write_bank(items, d / "bank.jsonl"), "dir": d, "vol": vol}
    f["det"] = write_replay(items, d / "det.jsonl", correct_output)
    f["noisy"] = write_noisy_replay(items, d / "noisy.jsonl", correct_output, wrong_output,
                                    lambda it: 0.5 if it["id"] in vol else 0.0)
    f["noisy_harmful"] = write_noisy_replay(
        items, d / "noisy_harmful.jsonl", lambda it: wrong_output(it) if it["id"] in harm else correct_output(it),
        wrong_output, lambda it: 0.5 if it["id"] in vol else 0.0)
    return f


def run(standins, fx, out, ref_backend, cand_backend, extra=(), ref_kw=None, cand_kw=None, cand_query=""):
    _, pa = standins.start(backend=f"replay:{ref_backend}", **(ref_kw or {}))
    _, pb = standins.start(backend=f"replay:{cand_backend}", **(cand_kw or {}))
    code, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb) + cand_query, "--bank", fx["bank"],
                             "--out", out] + list(extra))
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    return code, stdout, rep


# ------------------------------------------------------------------ routing
def test_deterministic_stacks_measure_a_zero_floor_on_the_probe(tmp_path, standins, fx):
    code, _, rep = run(standins, fx, tmp_path / "det", fx["det"], fx["det"], ["--tau", "0.1"])
    assert rep["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU"
    nz = rep["noise"]
    assert nz["routing"]["reference_every_item"] is False and nz["routing"]["candidate_every_item"] is False
    assert nz["floor"]["value"] == 0.0 and nz["floor"]["items"] == {"reference": 64, "candidate": 64}
    assert nz["floor"]["basis"] == {"reference": "probe items", "candidate": "probe items"}
    assert rep["verdict"]["gates"]["floor_source"] == "measured"
    assert rep["deterministic_mode"]["reference"]["status"] == "observed deterministic"
    recs = items_of(tmp_path / "det")
    assert sum("rep" in r for r in recs) == 64 and sum("cand_rep" in r for r in recs) == 64


def test_noisy_reference_runs_twice_on_every_item(tmp_path, standins, fx):
    _, _, rep = run(standins, fx, tmp_path / "noisy", fx["noisy"], fx["noisy"], ["--tau", "0.15"],
                    ref_kw={"noise_seed": 0}, cand_kw={"noise_seed": 1})
    nz = rep["noise"]
    assert nz["routing"]["reference_every_item"] is True and nz["routing"]["candidate_every_item"] is True
    assert all("rep" in r and "cand_rep" in r for r in items_of(tmp_path / "noisy"))
    assert abs(nz["floor"]["value"] - 0.2) < 0.07  # 40% volatile items flipping with probability 1/2
    assert nz["floor"]["basis"]["reference"] == "all items"
    assert rep["verdict"]["verdict"] not in ("DEVIATES", "INCONCLUSIVE_UNDERPOWERED")
    assert rep["deterministic_mode"]["reference"]["status"] == "observed nondeterministic"


def test_noise_on_the_candidate_alone_replicates_the_reference_too(tmp_path, standins, fx):
    _, _, rep = run(standins, fx, tmp_path / "cand_noise", fx["det"], fx["noisy"], ["--tau", "0.15"])
    nz = rep["noise"]
    assert nz["routing"]["reference_every_item"] is True and nz["routing"]["candidate_every_item"] is True
    assert nz["floor"]["stack"] == "candidate" and nz["floor"]["value"] > 0.1
    assert nz["per_stack"]["reference"]["answer_flips"] == 0


def test_plan_floor_is_used_and_a_contradicting_measurement_overrides_it(tmp_path, standins, fx):
    _, _, rep = run(standins, fx, tmp_path / "pf", fx["noisy"], fx["noisy"], ["--tau", "0.15", "--plan-floor", "0.2"],
                    cand_kw={"noise_seed": 7})
    assert rep["verdict"]["gates"]["floor_source"] == "supplied"
    assert rep["noise"]["gate_floor"]["supplied_contradicted"] is False
    _, _, rep = run(standins, fx, tmp_path / "pf0", fx["noisy"], fx["noisy"], ["--tau", "0.15", "--plan-floor", "0"],
                    cand_kw={"noise_seed": 7})
    assert rep["noise"]["gate_floor"]["supplied_contradicted"] is True
    assert rep["verdict"]["gates"]["floor_source"] == "measured"
    assert any("supplied --plan-floor is below" in x for x in rep["limitations"])


def test_unmeasured_floor_gives_no_pass_or_fail(tmp_path, standins, fx):
    code, _, rep = run(standins, fx, tmp_path / "unmeasured", fx["det"], fx["det"],
                       ["--tau", "0.1", "--floor-probe-items", "0", "--skip-channels", "--exit-on-verdict"])
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and code == 12
    assert rep["power"]["floor_source"] == "unmeasured" and rep["power"]["items_needed_lower_bound"] == 57
    _, _, rep = run(standins, fx, tmp_path / "supplied_only", fx["det"], fx["det"],
                    ["--tau", "0.1", "--floor-probe-items", "0", "--skip-channels", "--plan-floor", "0"])
    assert rep["verdict"]["gates"]["floor_source"] == "supplied"
    assert any("nothing in this run measured the floor" in x for x in rep["limitations"])


def test_stop_if_underpowered_stops_after_the_probe(tmp_path, standins, fx):
    code, _, rep = run(standins, fx, tmp_path / "stop", fx["det"], fx["det"], ["--stop-if-underpowered",
                                                                              "--exit-on-verdict"])
    assert code == 12 and rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and rep["verdict"]["n"] == 64
    assert rep["verdict"]["items_needed"] == 597 and rep["verdict"]["gates"]["stopped_after_probe_underpowered"]


def test_declared_deterministic_mode_with_observed_noise_is_flagged(tmp_path, standins, fx, monkeypatch):
    monkeypatch.setitem(determinism.CONFIRMED_SETTINGS, "openai", [
        {"setting": "example-batch-invariance", "how_to_enable": "start the engine with its deterministic option",
         "detect": {"route": "/engine_settings", "key": "batch_invariant"}}])
    _, _, rep = run(standins, fx, tmp_path / "declared", fx["det"], fx["noisy"], ["--tau", "0.15"],
                    cand_kw={"extra_routes": {"/engine_settings": {"batch_invariant": True}}},
                    cand_query="?deterministic=on")
    dm = rep["deterministic_mode"]["candidate"]
    assert dm["declared"]["spec"] == "on"
    assert dm["declared"]["engine_settings"] == [{"setting": "example-batch-invariance", "route": "/engine_settings",
                                                 "key": "batch_invariant", "value": True}]
    assert dm["declared_on_but_nondeterministic"] is True and "deterministic option" in dm["recommendation"]
    assert rep["deterministic_mode"]["reference"]["declared"]["engine_settings"][0]["value"] == "unobservable"
    assert any("declares a deterministic mode" in x for x in rep["limitations"])


# ------------------------------------------------------------------ known-wrong case: a byte-identical candidate never DEVIATES
@pytest.mark.parametrize("criterion", CRITERIA)
def test_identical_deterministic_candidate_never_deviates(tmp_path, standins, fx, criterion):
    _, _, rep = run(standins, fx, tmp_path / criterion, fx["det"], fx["det"], ["--tau", "0.05", "--criterion", criterion])
    assert rep["estimates"]["harmful"] == 0 and rep["verdict"]["verdict"] != "DEVIATES"


@pytest.mark.parametrize("criterion", CRITERIA)
def test_same_noisy_engine_byte_identical_candidate_never_deviates(tmp_path, standins, fx, criterion):
    """Same replay and same noise seed: the candidate equals the reference's first run byte for byte,
    while the reference's second run differs."""
    _, _, rep = run(standins, fx, tmp_path / criterion, fx["noisy"], fx["noisy"],
                    ["--tau", "0.05", "--criterion", criterion], ref_kw={"noise_seed": 3}, cand_kw={"noise_seed": 3})
    recs = items_of(tmp_path / criterion)
    assert all(r["identical"] for r in recs) and rep["estimates"]["harmful"] == 0
    if criterion != "harmful_abs":
        assert rep["noise"]["per_stack"]["reference"]["answer_flips"] > 0
        assert rep["verdict"]["pre_gate_deviation"] is None
    assert rep["verdict"]["verdict"] != "DEVIATES"


def test_independent_noisy_engines_do_not_deviate_but_the_010_rule_does(tmp_path, standins, fx):
    abs_dev = 0
    for seed in (11, 12):
        for criterion in ("net_change", "harmful_excess"):  # tau 0.15 is powered here (about 170 items needed)
            out = tmp_path / f"{criterion}_{seed}"
            _, _, rep = run(standins, fx, out, fx["noisy"], fx["noisy"], ["--tau", "0.15", "--criterion", criterion],
                            ref_kw={"noise_seed": 0}, cand_kw={"noise_seed": seed})
            assert rep["verdict"]["gates"]["underpowered"] is False
            assert rep["verdict"]["verdict"] != "DEVIATES" and rep["verdict"]["pre_gate_deviation"] is None, out
        _, _, rep = run(standins, fx, tmp_path / f"abs_{seed}", fx["noisy"], fx["noisy"],
                        ["--tau", "0.05", "--criterion", "harmful_abs"], ref_kw={"noise_seed": 0},
                        cand_kw={"noise_seed": seed})
        abs_dev += rep["verdict"]["verdict"] == "DEVIATES"
    assert abs_dev >= 1  # the fixture has teeth: the 0.1.0 rule counts the engine's own noise as harm
    # positive control: a harmful candidate on the same noisy engine
    for criterion in ("net_change", "harmful_excess"):
        _, _, rep = run(standins, fx, tmp_path / f"harm_{criterion}", fx["noisy"], fx["noisy_harmful"],
                        ["--tau", "0.15", "--criterion", criterion], ref_kw={"noise_seed": 0}, cand_kw={"noise_seed": 5})
        assert rep["verdict"]["verdict"] == "DEVIATES", criterion


# ------------------------------------------------------------------ known-wrong case: one item short of the number needed
def test_bank_one_item_short_is_underpowered(tmp_path, standins, fx):
    n_star = pw.items_needed({"t_one": 1}, {"t_one": pw.net_change_model(0.0)}, 0.05, 0.05, 0.9)["items_needed"]
    assert n_star == 117
    for n, expect in ((n_star - 1, "INCONCLUSIVE_UNDERPOWERED"), (n_star, "EQUIVALENT_WITHIN_TAU")):
        bank = write_bank(fx["items"][:n], tmp_path / f"bank{n}.jsonl")
        _, pa = standins.start(backend=f"replay:{fx['det']}")
        _, pb = standins.start(backend=f"replay:{fx['det']}")
        code, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--tau", "0.05",
                                 "--out", tmp_path / f"o{n}", "--exit-on-verdict"])
        line = json.loads(stdout.strip().splitlines()[-1])
        assert line["verdict"] == expect, n
        if n < n_star:
            assert code == 12 and line["items_needed"] == n_star


def test_bank_one_item_short_at_a_supplied_noisy_floor(tmp_path, standins, fx):
    model = {"t_one": pw.net_change_model(0.2)}
    n_star = pw.items_needed({"t_one": 1}, model, 0.05, 0.15, 0.9)["items_needed"]
    for n in (n_star - 1, n_star):
        bank = write_bank(fx["items"][:n], tmp_path / f"bank{n}.jsonl")
        _, pa = standins.start(backend=f"replay:{fx['noisy']}", noise_seed=0)
        _, pb = standins.start(backend=f"replay:{fx['noisy']}", noise_seed=1)
        _, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--tau", "0.15",
                              "--plan-floor", "0.2", "--out", tmp_path / f"o{n}"])
        line = json.loads(stdout.strip().splitlines()[-1])
        if n < n_star:
            assert line["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and line["items_needed"] == n_star
        else:
            assert line["verdict"] != "INCONCLUSIVE_UNDERPOWERED" and line["floor"]["source"] == "supplied"
            assert line["floor"]["value"] == 0.2


# ------------------------------------------------------------------ the items needed: a number a rerun can use
def test_a_bank_just_above_the_first_powered_size_gets_a_number_that_passes(tmp_path, standins):
    """At a floor of 0.02 and tau 0.03, 675 items reach power 0.9 but 683 do not (exact power is a
    sawtooth).  The number reported for 683 items passes, and it is the number `inferdrift power` gives."""
    items = make_single_suite_bank(692)
    _, out = run_main(["power", "--floor", "0.02", "--tau", "0.03"])
    n_power = json.loads(out)["items_needed"]
    assert n_power == 692
    det = write_replay(items, tmp_path / "det.jsonl", correct_output)
    for n, expect in ((683, "INCONCLUSIVE_UNDERPOWERED"), (692, "EQUIVALENT_WITHIN_TAU")):
        bank = write_bank(items[:n], tmp_path / f"bank{n}.jsonl")
        _, pa = standins.start(backend=f"replay:{det}")
        _, pb = standins.start(backend=f"replay:{det}")
        _, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--tau", "0.03",
                              "--plan-floor", "0.02", "--out", tmp_path / f"o{n}"])
        line = json.loads(stdout.strip().splitlines()[-1])
        assert line["verdict"] == expect, n
        assert line["floor"] == {"value": 0.02, "source": "supplied", "measured": 0.0}  # the floor the gate used
        if n == 683:
            assert line["items_needed"] == n_power and "every size from 692 to 742 was checked" in line[
                "items_needed_note"]


def test_max_items_gets_a_max_items_number_on_the_runs_own_order(tmp_path, standins):
    """Three capabilities of 700 items, tau 0.01 (597 each at a zero floor): --max-items 1800 leaves one
    capability short in the seeded order.  The reported number is a --max-items value on that order,
    and rerunning with it passes."""
    suites = ("t_a", "t_b", "t_c")
    items = make_single_suite_bank(2100)
    for i, it in enumerate(items):
        it["suite"] = suites[i % 3]
    bank = write_bank(items, tmp_path / "bank.jsonl")
    det = write_replay(items, tmp_path / "det.jsonl", correct_output)
    n = 1800
    for expect in ("INCONCLUSIVE_UNDERPOWERED", "EQUIVALENT_WITHIN_TAU"):
        _, pa = standins.start(backend=f"replay:{det}")
        _, pb = standins.start(backend=f"replay:{det}")
        _, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--max-items", str(n),
                              "--out", tmp_path / f"o{n}"])
        rep = json.loads((tmp_path / f"o{n}" / "report.json").read_text())
        assert rep["verdict"]["verdict"] == expect, n
        if expect == "INCONCLUSIVE_UNDERPOWERED":
            pw_ = rep["power"]
            assert pw_["items_needed_basis"] == "max_items" and 1800 < pw_["items_needed"] <= 2100
            short = pw_["short_capabilities"]
            assert [s["capability"] for s in short] == ["t_c"] and short[0]["planned"] < 597
            assert short[0]["needed"] == short[0]["needed_alone"] == 597
            assert f"--max-items {pw_['items_needed']}" in rep["verdict"]["items_needed_note"]
            n = pw_["items_needed"]


def test_a_one_item_capability_is_named_when_the_search_stops_at_its_cap(tmp_path, standins):
    items = make_single_suite_bank(1000, "t_big")
    items[-1]["suite"] = "t_tiny"
    bank = write_bank(items, tmp_path / "bank.jsonl")
    det = write_replay(items, tmp_path / "det.jsonl", correct_output)
    _, pa = standins.start(backend=f"replay:{det}")
    _, pb = standins.start(backend=f"replay:{det}")
    code, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--out", tmp_path / "o",
                             "--exit-on-verdict"])
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    pw_ = rep["power"]
    assert code == 12 and pw_["items_needed"] is None and pw_["items_needed_exceeds"] == 20000
    srch = pw_["items_needed_search"]
    assert srch["searched_to"] == 20020 and srch["per_capability_at_stop"] == {"t_big": 20000, "t_tiny": 20}
    assert srch["weakest_at_stop"]["capability"] == "t_tiny"
    assert [(s["capability"], s["planned"], s["needed_alone"]) for s in pw_["short_capabilities"]] == [
        ("t_tiny", 1, 597)]
    line = json.loads(stdout.strip().splitlines()[-1])
    assert line["items_needed"] is None and "20020 items in this bank's proportions" in line["items_needed_note"]
    summary = (tmp_path / "o" / "summary.md").read_text()
    assert "per capability would be needed" not in summary and "t_tiny would have 20 items" in summary
    assert "t_tiny 1 planned" in summary


# ------------------------------------------------------------------ deviations, the power gate and the stopping path
def test_a_deviates_first_verdict_that_the_final_gate_withdraws_is_not_reported_as_one(tmp_path, standins):
    """The candidate is wrong on the first 64 items, and both stacks are noisy only on the last 150, so the
    floor is 0 at the early looks and about 0.25 at the end.  With early stop the run stops at DEVIATES;
    without it, the final power gate finds the plan underpowered and decides."""
    items = make_single_suite_bank(300)
    bank = write_bank(items, tmp_path / "bank.jsonl")
    late = {it["id"] for it in items[150:]}
    early = {it["id"] for it in items[:64]}
    ref = write_noisy_replay(items, tmp_path / "ref.jsonl", correct_output, wrong_output,
                             lambda it: 0.5 if it["id"] in late else 0.0)
    cand = write_noisy_replay(items, tmp_path / "cand.jsonl",
                              lambda it: wrong_output(it) if it["id"] in early else correct_output(it), wrong_output,
                              lambda it: 0.5 if it["id"] in late else 0.0)
    reps = {}
    for label, extra in (("early_stop", []), ("no_early_stop", ["--no-early-stop"])):
        _, pa = standins.start(backend=f"replay:{ref}", noise_seed=0)
        _, pb = standins.start(backend=f"replay:{cand}", noise_seed=1)
        run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--order", "bank", "--tau", "0.1",
                  "--replicate", "always", "--out", tmp_path / label] + extra)
        reps[label] = json.loads((tmp_path / label / "report.json").read_text())
    v = reps["early_stop"]["verdict"]
    assert v["verdict"] == "DEVIATES" and v["first_verdict"]["verdict"] == "DEVIATES" and v["first_verdict"]["n"] == 32
    assert v["gates"]["deviates_look"]["n"] == 32 and v["withdrawn_first_verdict"] is None
    v = reps["no_early_stop"]["verdict"]
    assert v["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and v["first_verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED"
    assert v["withdrawn_first_verdict"]["n"] == 32 and v["gates"]["deviates_look"] is None
    assert any("final power gate withdrew" in x for x in reps["no_early_stop"]["limitations"])
    summary = (tmp_path / "no_early_stop" / "summary.md").read_text()
    assert "First verdict: DEVIATES" not in summary and "was withdrawn" in summary


def test_a_deviates_verdict_after_a_blocked_look_is_not_called_evidence(tmp_path, standins):
    """Both stacks are noisy on the first 64 items only, and the candidate is wrong on items 0-399: the
    look at n = 64 deviates but the power gate blocks it, and the look at n = 128 deviates and is allowed."""
    items = make_single_suite_bank(600)
    bank = write_bank(items, tmp_path / "bank.jsonl")
    noisy_ids = {it["id"] for it in items[:64]}
    wrong_ids = {it["id"] for it in items[:400]}
    ref = write_noisy_replay(items, tmp_path / "ref.jsonl", correct_output, wrong_output,
                             lambda it: 0.5 if it["id"] in noisy_ids else 0.0)
    cand = write_noisy_replay(items, tmp_path / "cand.jsonl",
                              lambda it: wrong_output(it) if it["id"] in wrong_ids else correct_output(it),
                              wrong_output, lambda it: 0.5 if it["id"] in noisy_ids else 0.0)
    _, pa = standins.start(backend=f"replay:{ref}", noise_seed=0)
    _, pb = standins.start(backend=f"replay:{cand}", noise_seed=1)
    run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--order", "bank", "--tau", "0.08",
              "--out", tmp_path / "o"])
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    v = rep["verdict"]
    blocked = [lk["n"] for lk in rep["looks"] if lk["deviates"] and not lk["deviates_allowed_by_power_gate"]]
    assert v["verdict"] == "DEVIATES" and blocked == [64] and v["first_verdict"]["n"] == 128
    assert v["pre_gate_deviation"]["n"] == 64 and v["gates"]["deviates_look"]["n"] == 128
    assert not any("did not let become a verdict" in x for x in rep["limitations"])
    assert "Evidence of a deviation" not in (tmp_path / "o" / "summary.md").read_text()


def test_stop_if_underpowered_keeps_the_evidence_of_the_probe_items(tmp_path, standins, fx):
    """A candidate wrong on every item: with or without the stop, the looks the probe covers show the deviation."""
    bad = write_replay(fx["items"], tmp_path / "bad.jsonl", wrong_output)
    devs = {}
    for label, extra in (("stop", ["--stop-if-underpowered"]), ("full", [])):
        _, pa = standins.start(backend=f"replay:{fx['det']}")
        _, pb = standins.start(backend=f"replay:{bad}")
        run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", fx["bank"], "--out", tmp_path / label]
                 + extra)
        rep = json.loads((tmp_path / label / "report.json").read_text())
        assert rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED"
        devs[label] = rep["verdict"]["pre_gate_deviation"]
        assert "Evidence of a deviation" in (tmp_path / label / "summary.md").read_text()
    assert devs["stop"] and devs["stop"]["n"] == 32 and devs["stop"]["allowed_by_power_gate"] is False
    assert {k: devs["stop"][k] for k in ("look", "n", "lower_bounds")} == {
        k: devs["full"][k] for k in ("look", "n", "lower_bounds")}


def test_channel_probe_errors_alone_route_nothing_and_say_so(tmp_path, standins, fx):
    _, _, rep = run(standins, fx, tmp_path / "errs", fx["det"], fx["det"], ["--tau", "0.1"],
                    cand_kw={"fail_every": 7})
    ch = rep["channels"]["candidate"]
    assert ch["greedy_repeat_identity_rate"] < 1.0 and ch["greedy_repeat_identity_rate_without_errors"] == 1.0
    flags = rep["coupling"]["flags"]
    assert not any("routed to the noise floor" in x for x in flags)
    assert any("errors are not counted as nondeterminism" in x for x in flags)
    assert rep["noise"]["routing"]["candidate_every_item"] is False


# ------------------------------------------------------------------ known-wrong case: a replicate silently skipped is refused
class _DropOne:
    """Wraps an adapter; on the ``nth`` generate call returns one output fewer, without raising."""

    def __init__(self, inner, nth: int) -> None:
        self.inner, self.nth, self.calls = inner, nth, 0

    def generate(self, reqs, meter):
        self.calls += 1
        out = self.inner.generate(reqs, meter)
        return out[:-1] if self.calls == self.nth else out

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("nth,code", [(2, "REPLICATE_INCOMPLETE"), (1, "RESULTS_INCOMPLETE")])
def test_a_silently_skipped_run_is_refused(tmp_path, standins, fx, monkeypatch, nth, code):
    real = cmp.build_adapter

    def build(role, *a, **kw):
        ad = real(role, *a, **kw)
        return _DropOne(ad, nth) if role == "reference" else ad

    monkeypatch.setattr(cmp, "build_adapter", build)
    _, pa = standins.start(backend=f"replay:{fx['det']}")
    _, pb = standins.start(backend=f"replay:{fx['det']}")
    rc, stdout = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", fx["bank"], "--skip-channels",
                           "--out", tmp_path / "skip", "--exit-on-verdict"])
    rep = json.loads((tmp_path / "skip" / "report.json").read_text())
    assert rc == 3 and rep["status"] == "refused" and rep["refusal"]["code"] == code
    assert rep["verdict"]["verdict"] is None


# ------------------------------------------------------------------ replicate construction under sampling
def test_sampled_replicates_are_exchangeable_with_the_cross_pair(tmp_path, standins, fx, monkeypatch):
    """Two seed-honouring stand-ins with the same noise seed (the same sampler) and a harmful candidate.
    Every run gets its own seed, so harmful_excess sees the harm; a replicate built as in 0.1.0 (the
    candidate sharing the reference's seed) would subtract the sampling noise from a noise-free cross pair
    and pass it."""
    items = fx["items"]
    vol = pick(items, 0.8, 21)
    d = tmp_path
    ref = write_noisy_replay(items, d / "r.jsonl", correct_output, wrong_output,
                             lambda it: 0.5 if it["id"] in vol else 0.0)
    cand = write_noisy_replay(items, d / "c.jsonl",
                              lambda it: correct_output(it) if it["id"] in vol else wrong_output(it), wrong_output,
                              lambda it: 0.5 if it["id"] in vol else 0.0)
    args = ["--decode", "sample", "--seed", "4", "--tau", "0.15", "--criterion", "harmful_excess"]
    _, _, rep = run(standins, fx, d / "now", ref, cand, args)
    assert rep["coupling"]["mode"] == "seed_shared_unverified"
    assert rep["noise"]["same_coupling_replicates"] is False
    assert rep["verdict"]["verdict"] != "EQUIVALENT_WITHIN_TAU"
    # control: with the 0.1.0 replicate construction the same harmful candidate passes
    real = cmp.run_indices
    monkeypatch.setattr(cmp, "run_indices", lambda crit, dec, mode: ({"R1": 0, "C1": 0, "R2": 1, "C2": 1}, False,
                                                                     "0.1.0 construction (test control)"))
    _, _, rep = run(standins, fx, d / "then", ref, cand, args)
    assert rep["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU"
    monkeypatch.setattr(cmp, "run_indices", real)


# ------------------------------------------------------------------ errors
class _ErrorsFor:
    """Wraps an adapter; outputs for the given item ids are always errors (the retry fails too)."""

    def __init__(self, inner, ids: set, only_call: int | None = None) -> None:
        self.inner, self.ids, self.only_call, self.calls = inner, ids, only_call, 0

    def generate(self, reqs, meter):
        self.calls += 1
        out = self.inner.generate(reqs, meter)
        if self.only_call is not None and self.calls not in self.only_call:
            return out
        return [GenResult(r.item_id, "", "error", None, None, "estimated_worst_case", error="injected error")
                if r.item_id in self.ids else r for r in out]

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_errors_never_count_as_noise_and_follow_the_stated_rule(tmp_path, standins, fx, monkeypatch):
    """A reference failing in its second run leaves the floor unchanged (left out, never a repeat flip);
    a candidate error counts as a loss; a reference first-run error leaves the item out and blocks a pass."""
    ids = {it["id"] for it in fx["items"][:300:7]}
    real = cmp.build_adapter
    # (1) reference errors in its second run only (calls 2 and 3: the probe's second run and its retry)
    monkeypatch.setattr(cmp, "build_adapter", lambda role, *a, **kw: _ErrorsFor(real(role, *a, **kw), ids, {2, 3})
                        if role == "reference" else real(role, *a, **kw))
    _, _, rep = run(standins, fx, tmp_path / "rep_err", fx["det"], fx["det"], ["--tau", "0.1", "--skip-channels"])
    nz = rep["noise"]
    assert nz["errors_per_pass"]["R2"]["errored_after_retry"] > 0
    assert nz["floor"]["value"] == 0.0 and nz["noisy_on_probe"]["reference"] is False
    assert nz["per_stack"]["reference"]["errored_pairs_left_out"] > 0
    assert rep["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU"
    # (2) candidate errors in its first run: graded wrong, counted as losses
    monkeypatch.setattr(cmp, "build_adapter", lambda role, *a, **kw: _ErrorsFor(real(role, *a, **kw), ids)
                        if role == "candidate" else real(role, *a, **kw))
    _, _, rep = run(standins, fx, tmp_path / "cand_err", fx["det"], fx["det"], ["--tau", "0.1", "--skip-channels"])
    recs = items_of(tmp_path / "cand_err")
    errs = [r for r in recs if r["cand"]["finish"] == "error"]
    assert errs and all(r["residual_net"] == 1 for r in errs)
    assert rep["noise"]["errors_per_pass"]["C1"]["errored_after_retry"] == len(errs)
    # (3) reference errors in its first run: left out of the residuals, and the run cannot pass
    monkeypatch.setattr(cmp, "build_adapter", lambda role, *a, **kw: _ErrorsFor(real(role, *a, **kw), ids, {1, 2})
                        if role == "reference" else real(role, *a, **kw))
    _, _, rep = run(standins, fx, tmp_path / "ref_err", fx["det"], fx["det"], ["--tau", "0.1", "--skip-channels"])
    assert rep["verdict"]["excluded_items"] > 0 and rep["verdict"]["verdict"] == "INCONCLUSIVE"
    assert any("left out of the residuals" in x for x in rep["limitations"])


# ------------------------------------------------------------------ benign churn and a template fault
def test_benign_churn_and_a_template_fault(tmp_path, standins, fx):
    items = fx["items"]
    wrong_ref = pick(items, 0.3, 31)
    swap_bad = pick(items, 0.15, 32, exclude=wrong_ref)  # right -> wrong
    swap_good = pick([it for it in items if it["id"] in wrong_ref], 0.5, 33)  # 15% of all: wrong -> right
    d = tmp_path
    ref = write_replay(items, d / "r.jsonl", lambda it: wrong_output(it) if it["id"] in wrong_ref else correct_output(it))
    cand = write_replay(items, d / "c.jsonl", lambda it: wrong_output(it) if (
        (it["id"] in wrong_ref and it["id"] not in swap_good) or it["id"] in swap_bad) else correct_output(it))
    _, _, rep = run(standins, fx, d / "benign_abs", ref, cand, ["--tau", "0.05", "--criterion", "harmful_abs"])
    assert rep["verdict"]["verdict"] == "DEVIATES"
    _, _, rep = run(standins, fx, d / "benign_net", ref, cand, ["--tau", "0.15"])
    assert rep["estimates"]["net_change"]["point"] == 0.0
    assert rep["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU"
    fault = write_replay(items, d / "fault.jsonl", lambda it: "I cannot answer that.")
    for criterion in CRITERIA:
        _, _, rep = run(standins, fx, d / f"fault_{criterion}", ref, fault, ["--tau", "0.15", "--criterion", criterion])
        assert rep["verdict"]["verdict"] == "DEVIATES", criterion
