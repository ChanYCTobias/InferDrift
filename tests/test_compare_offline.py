# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end comparisons against local stand-in endpoints (127.0.0.1 only, no model, no network).

The release 0.1.0 scenarios run with ``--criterion harmful_abs`` (the 0.1.0 rule); each has a
counterpart under the default criterion where the behaviour differs."""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request

import pytest

from helpers import (Standins, compare, correct_output, make_single_suite_bank, make_test_bank, new_test_key,
                     run_main, scan_for, url, write_bank, write_replay, wrong_output)

ABS = ["--criterion", "harmful_abs"]
from inferdrift.redact import REDACTED, scrub_text


@pytest.fixture
def standins():
    s = Standins()
    yield s
    s.close()


@pytest.fixture
def known(tmp_path):
    """A 300-item bank with known answers, and a replay file that answers every item correctly."""
    items = make_test_bank(300)
    bank = write_bank(items, tmp_path / "bank.jsonl")
    replay = write_replay(items, tmp_path / "replay_ok.jsonl", correct_output)
    return items, bank, replay


def test_identical_stacks_are_equivalent(tmp_path, standins, known):
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}")
    code, stdout, rep = compare(pa, pb, bank, tmp_path / "same", ABS)
    assert code == 0 and json.loads(stdout.strip().splitlines()[-1])["verdict"] == "EQUIVALENT_WITHIN_TAU"
    assert rep["status"] == "completed"
    assert rep["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU" and rep["verdict"]["n"] == 300
    est = rep["estimates"]
    assert est["harmful"] == 0 and est["identical_outputs"] == 300
    assert est["accuracy"]["reference"] == 1.0
    inh = rep["inheritance"]
    calls = inh["candidate_grading_calls_with_inheritance"]
    assert calls == inh["coupling_overhead_grading_calls"]["audit_regrades"] and 0 < calls <= 0.1 * 300
    assert (tmp_path / "same" / "summary.md").read_text().startswith("# inferdrift compare: EQUIVALENT_WITHIN_TAU")
    assert rep["tool"]["name"] == "inferdrift"


def test_identical_stacks_default_criterion(tmp_path, standins, known):
    """Default criterion: three capabilities of 100 items are too few for 1 point; with a tolerance the
    bank supports, identical stacks pass."""
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}")
    code, stdout, rep = compare(pa, pb, bank, tmp_path / "under", ["--exit-on-verdict"])
    line = json.loads(stdout.strip().splitlines()[-1])
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and code == 12
    assert line["items_needed"] == rep["verdict"]["items_needed"] == 3 * 597  # 597 per capability, zero floor
    assert line["floor"] == {"value": 0.0, "source": "measured"}
    assert rep["noise"]["floor"]["items"] == {"reference": 64, "candidate": 64}
    code, _, rep2 = compare(pa, pb, bank, tmp_path / "pass", ["--tau", "0.1", "--exit-on-verdict"])
    assert rep2["verdict"]["verdict"] == "EQUIVALENT_WITHIN_TAU" and code == 0
    assert rep2["schema"] == "inferdrift/report/2" and rep2["verdict"]["criterion"] == "net_change"
    assert set(rep2["verdict"]["per_capability"]) == {"t_numeric", "t_choice", "t_tool"}


def test_harmful_change_deviates(tmp_path, standins, known):
    items, bank, replay = known
    bad = {it["id"] for it in random.Random(5).sample(items, 30)}  # 10% of answers become wrong
    replay_bad = write_replay(items, tmp_path / "replay_bad.jsonl",
                              lambda it: wrong_output(it) if it["id"] in bad else correct_output(it))
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pc = standins.start(backend=f"replay:{replay_bad}")
    code, _, rep = compare(pa, pc, bank, tmp_path / "bad", ["--exit-on-verdict"] + ABS)
    assert rep["verdict"]["verdict"] == "DEVIATES" and code == 10
    assert rep["verdict"]["first_verdict"]["n"] < 300  # stopped at an early look
    assert rep["estimates"]["harmful"] > 0 and rep["estimates"]["helpful"] == 0
    # default criterion: 1 point per capability is underpowered, so no fail is issued
    code, stdout, rep = compare(pa, pc, bank, tmp_path / "bad_default", ["--exit-on-verdict"])
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and code == 12
    assert rep["verdict"]["first_verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED"


def test_harmful_change_deviates_default_criterion(tmp_path, standins):
    items = make_single_suite_bank(200)
    bank = write_bank(items, tmp_path / "one.jsonl")
    bad = {it["id"] for it in random.Random(9).sample(items, 80)}  # 40% of answers become wrong
    ref = write_replay(items, tmp_path / "r.jsonl", correct_output)
    cand = write_replay(items, tmp_path / "c.jsonl", lambda it: wrong_output(it) if it["id"] in bad else correct_output(it))
    _, pa = standins.start(backend=f"replay:{ref}")
    _, pc = standins.start(backend=f"replay:{cand}")
    for crit in ("net_change", "harmful_excess"):
        code, _, rep = compare(pa, pc, bank, tmp_path / crit, ["--tau", "0.1", "--criterion", crit, "--exit-on-verdict"])
        assert rep["verdict"]["verdict"] == "DEVIATES" and code == 10, crit
        assert rep["verdict"]["first_verdict"]["verdict"] == "DEVIATES" and rep["verdict"]["first_verdict"]["n"] < 200
        assert rep["verdict"]["bound"]["lower"] > 0.1 and rep["verdict"]["bound"]["capability"] == "t_one"


def test_too_few_items_is_inconclusive(tmp_path, standins, known):
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}")
    code, _, rep = compare(pa, pb, bank, tmp_path / "few", ["--max-items", "64", "--exit-on-verdict"] + ABS)
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE" and code == 11
    code, _, rep = compare(pa, pb, bank, tmp_path / "few_default", ["--max-items", "64", "--exit-on-verdict"])
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED" and code == 12


def test_silent_truncation_is_flagged(tmp_path, standins, known):
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}", truncate_frac=0.5)
    _, _, rep = compare(pa, pb, bank, tmp_path / "trunc", ["--no-early-stop", "--max-items", "96"] + ABS)
    tr = rep["diagnostics"]["truncation"]
    assert tr["silent_truncation_suspected"] is True and tr["candidate_is_proper_prefix_with_stop"] >= 50
    assert rep["verdict"]["verdict"] == "DEVIATES"
    assert any("truncated" in x for x in rep["limitations"])
    _, pc = standins.start(backend=f"replay:{replay}")
    _, _, rep2 = compare(pa, pc, bank, tmp_path / "trunc_control", ["--max-items", "96"] + ABS)
    assert rep2["diagnostics"]["truncation"]["silent_truncation_suspected"] is False
    _, _, rep3 = compare(pa, pb, bank, tmp_path / "trunc_default", ["--no-early-stop", "--max-items", "96"])
    assert rep3["diagnostics"]["truncation"]["silent_truncation_suspected"] is True
    dev = rep3["verdict"]["pre_gate_deviation"]  # visible although the plan is underpowered
    assert rep3["verdict"]["verdict"] == "INCONCLUSIVE_UNDERPOWERED"
    assert dev and dev["allowed_by_power_gate"] is False and dev["capabilities"]
    assert (tmp_path / "trunc_default" / "summary.md").read_text().count("Evidence of a deviation") == 1


def test_ignored_seed_switches_to_independent_bounds(tmp_path, standins, known):
    items, bank, _ = known
    _, pa = standins.start(backend="synthetic")
    _, pb = standins.start(backend="synthetic", ignore_seed=True)
    _, _, rep = compare(pa, pb, bank, tmp_path / "seed", ["--decode", "sample", "--max-items", "64"] + ABS)
    assert rep["channels"]["candidate"]["seed_honored"] is False
    assert rep["channels"]["reference"]["seed_honored"] is True
    assert rep["coupling"]["mode"] == "independent" and rep["coupling"]["independent_bounds"] is True
    assert rep["verdict"]["statistic"] == "excess_harmful_flip_rate"
    assert any("seed not honoured" in x for x in rep["limitations"])
    records = [json.loads(x) for x in (tmp_path / "seed" / "items.jsonl").read_text().splitlines() if x.strip()]
    assert records and all("rep" in r for r in records)  # the reference replicate ran
    _, _, rep = compare(pa, pb, bank, tmp_path / "seed_default", ["--decode", "sample", "--max-items", "64"])
    assert rep["coupling"]["mode"] == "independent" and rep["verdict"]["statistic"] == "net_loss"
    assert rep["noise"]["routing"]["reference_every_item"] is True
    assert rep["noise"]["same_coupling_replicates"] is False
    records = [json.loads(x) for x in (tmp_path / "seed_default" / "items.jsonl").read_text().splitlines() if x.strip()]
    assert records and all("rep" in r for r in records)


def test_nondeterministic_engine_is_flagged(tmp_path, standins, known):
    """harmful_abs (0.1.0): greedy-decoding noise is flagged, but it is not subtracted from the verdict.
    The default criterion routes it to the noise floor: both stacks run twice on every item."""
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}", nondeterministic=1.0)
    _, _, rep = compare(pa, pb, bank, tmp_path / "noisy", ["--max-items", "96"] + ABS)
    assert rep["channels"]["candidate"]["greedy_repeat_identity_rate"] < 1.0
    assert any("not deterministic under greedy decoding" in x for x in rep["coupling"]["flags"])
    assert rep["verdict"]["statistic"] == "harmful_flip_rate"
    assert any("harmful_abs counts a stack's own nondeterminism" in x for x in rep["limitations"])
    _, _, rep = compare(pa, pb, bank, tmp_path / "noisy_default", ["--max-items", "96"])
    assert any("routed to the noise floor" in x for x in rep["coupling"]["flags"])
    routing = rep["noise"]["routing"]
    assert routing["reference_every_item"] is True and routing["candidate_every_item"] is True
    assert rep["deterministic_mode"]["candidate"]["status"] == "observed nondeterministic"
    assert rep["deterministic_mode"]["candidate"]["recommendation"]
    assert rep["noise"]["per_stack"]["candidate"]["answer_flips"] == 0  # byte noise, answers stable


def test_missing_logprobs_are_detected(tmp_path, standins, known):
    items, bank, replay = known
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}", no_logprobs=True)
    _, _, rep = compare(pa, pb, bank, tmp_path / "nolp", ["--max-items", "64"])
    assert rep["channels"]["candidate"]["top_logprobs"] is False
    assert rep["channels"]["reference"]["top_logprobs"] is True
    assert rep["status"] == "completed" and rep["verdict"]["verdict"] is not None


def test_keys_never_reach_outputs(tmp_path, standins, known, monkeypatch):
    items, bank, replay = known
    server_key, wrong_key = new_test_key(), new_test_key()
    _, pa = standins.start(backend=f"replay:{replay}")
    _, pb = standins.start(backend=f"replay:{replay}", api_key=server_key, leak_key_in_errors=True)
    # control: this endpoint really echoes the presented key in its error body
    req = urllib.request.Request(f"{url(pb)}/models", headers={"Authorization": f"Bearer {wrong_key}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert wrong_key in e.value.read().decode()
    monkeypatch.setenv("INFERDRIFT_TEST_CAND_KEY", wrong_key)
    out = tmp_path / "wrong_key"
    code, stdout, rep = compare(pa, pb, bank, out, ["--max-items", "64", "--cand-api-key-env",
                                                    "INFERDRIFT_TEST_CAND_KEY"])
    assert scan_for([out], [wrong_key, server_key], [stdout]) == []
    assert rep["status"] == "error" and REDACTED in json.dumps(rep["error"])
    # a valid key, with an endpoint that fails every third request and echoes its key
    _, pc = standins.start(backend=f"replay:{replay}", api_key=server_key, leak_key_in_errors=True, fail_every=3)
    monkeypatch.setenv("INFERDRIFT_TEST_CAND_KEY", server_key)
    out2 = tmp_path / "valid_key"
    _, stdout2, rep2 = compare(pa, pc, bank, out2, ["--max-items", "64", "--cand-api-key-env",
                                                    "INFERDRIFT_TEST_CAND_KEY"])
    assert scan_for([out2], [server_key], [stdout2]) == []
    assert rep2["noise"]["errors_per_pass"]["C1"]["errored"] > 0  # errors are retried once, and counted
    assert scrub_text(f"x {server_key} y") == f"x {REDACTED} y"


def test_dollar_cap_binds_against_the_server_ledger(tmp_path, standins, known):
    items, bank, replay = known
    sa, pa = standins.start(backend=f"replay:{replay}", price_in=10.0, price_out=30.0)
    sb, pb = standins.start(backend=f"replay:{replay}", price_in=10.0, price_out=30.0)
    code, stdout, rep = compare(pa, pb, bank, tmp_path / "cap", ["--skip-channels", "--ref-price", "10,30",
                                                                 "--cand-price", "10,30", "--max-usd", "0.05"] + ABS)
    server_usd = sa.totals["usd"] + sb.totals["usd"]
    cost = rep["cost"]
    assert code == 0 and rep["status"] == "completed" and cost["stopped_by"] == "usd"
    assert 0 < server_usd <= 0.05
    assert abs(cost["usd_total"] - server_usd) <= 1e-9
    assert cost["caps"] == {"max_usd": 0.05, "max_tokens": None, "max_wall_s": None}
    assert rep["verdict"]["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("dialect", ["llamacpp", "vllm", "sglang", "openai"])
def test_dialects(tmp_path, standins, dialect):
    bank = write_bank(make_test_bank(64), tmp_path / "bank64.jsonl")
    _, pa = standins.start(backend="synthetic", dialect=dialect)
    _, pb = standins.start(backend="synthetic", dialect=dialect)
    _, _, rep = compare(pa, pb, bank, tmp_path / f"d_{dialect}")
    assert rep["status"] == "completed"
    assert rep["stacks"]["candidate"]["kind"] == dialect
    assert rep["estimates"]["identical_outputs"] == 64


def test_judge_items_through_the_cli(tmp_path, standins):
    items = [{"id": f"j-{i:03d}", "suite": "t_judge", "messages": [{"role": "user", "content": f"Explain {i}."}],
              "tools": None, "template_kwargs": {}, "max_new_tokens": 32,
              "grader": {"type": "judge", "rubric": "a short explanation"}} for i in range(20)]
    bank = write_bank(items, tmp_path / "judge_bank.jsonl")
    ref_rp = write_replay(items, tmp_path / "judge_ref.jsonl", lambda it: f"Because of reason {it['id']}.")
    cand_rp = write_replay(items, tmp_path / "judge_cand.jsonl",
                           lambda it: f"Because of reason {it['id']}." if int(it["id"][2:]) < 15 else "No idea.")
    _, pa = standins.start(backend=f"replay:{ref_rp}")
    _, pb = standins.start(backend=f"replay:{cand_rp}")
    js, pj = standins.start(backend="synthetic")
    cache = tmp_path / "judge_cache.jsonl"
    extra = ["--judge", url(pj), "--judge-cache", cache, "--skip-channels"]
    _, _, rep = compare(pa, pb, bank, tmp_path / "rj1", extra)
    first_calls = js.totals["requests"]
    assert rep["status"] == "completed"
    assert first_calls == 25  # 20 reference outputs plus 5 divergent candidate outputs
    compare(pa, pb, bank, tmp_path / "rj2", extra)
    assert js.totals["requests"] == first_calls  # the verdict cache is reused across runs
    code, _ = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--out", tmp_path / "rj3"])
    assert code == 3  # judge-graded items need a judge
