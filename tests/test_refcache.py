# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The reference cache: reuse across candidates, strict loading, grader identity."""

from __future__ import annotations

import json

import pytest

from helpers import (Standins, correct_output, items_of, make_single_suite_bank, pick, run_main, url, write_bank,
                     write_noisy_replay, write_replay, wrong_output)


@pytest.fixture
def standins():
    s = Standins()
    yield s
    s.close()


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    items = make_single_suite_bank(200)
    bad = pick(items, 0.1, 4)
    vol = pick(items, 0.4, 5)
    return {"items": items, "bank": write_bank(items, d / "bank.jsonl"),
            "ref": write_replay(items, d / "ref.jsonl", correct_output),
            "cand_a": write_replay(items, d / "a.jsonl", correct_output),
            "cand_b": write_replay(items, d / "b.jsonl",
                                   lambda it: wrong_output(it) if it["id"] in bad else correct_output(it)),
            "noisy": write_noisy_replay(items, d / "noisy.jsonl", correct_output, wrong_output,
                                        lambda it: 0.5 if it["id"] in vol else 0.0)}


def cmp(ref_port, cand_port, bank, out, cache, extra=()):
    code, stdout = run_main(["compare", "--ref", url(ref_port), "--cand", url(cand_port), "--bank", bank, "--out", out,
                             "--ref-cache", cache, "--tau", "0.1"] + list(extra))
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    return code, rep


@pytest.mark.parametrize("ref_key", ["ref", "noisy"])
def test_second_candidate_needs_no_reference_generation(tmp_path, standins, fx, ref_key):
    sr, pr = standins.start(backend=f"replay:{fx[ref_key]}")
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    _, pb = standins.start(backend=f"replay:{fx['cand_b']}")
    cache = tmp_path / "cache"
    _, rep1 = cmp(pr, pa, fx["bank"], tmp_path / "a", cache)
    assert rep1["reference_cache"]["written"] is True and rep1["reference_cache"]["entry_found"] is False
    before = sr.totals["requests"]
    _, rep2 = cmp(pr, pb, fx["bank"], tmp_path / "b", cache)
    assert sr.totals["requests"] == before  # the stand-in's own ledger: no reference generation at all
    rc = rep2["reference_cache"]
    assert rc["entry_found"] is True and rc["reference_generations"] == 0 and rc["misses"] == 0
    assert rep2["channels"]["reference"].get("from_reference_cache") is True
    r1, r2 = items_of(tmp_path / "a"), items_of(tmp_path / "b")
    assert [r["id"] for r in r1] == [r["id"] for r in r2]
    for x, y in zip(r1, r2):
        assert x["ref"]["text"] == y["ref"]["text"] and x["ref_grade"] == y["ref_grade"]
        assert y["ref_source"] == "cache"
        if "rep" in x:
            assert y["rep"]["text"] == x["rep"]["text"] and y["rep_source"] == "cache"
    calls = rep2["inheritance"]["grading_calls"]
    assert calls.get("reference_calls", 0) == 0 and calls.get("reference_cache_inherited", 0) > 0


def test_a_larger_plan_generates_only_the_new_items(tmp_path, standins, fx):
    sr, pr = standins.start(backend=f"replay:{fx['ref']}")
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    cache = tmp_path / "cache"
    cmp(pr, pa, fx["bank"], tmp_path / "small", cache, ["--max-items", "100"])
    before = sr.totals["requests"]
    _, rep = cmp(pr, pa, fx["bank"], tmp_path / "full", cache)
    assert sr.totals["requests"] - before == 100 == rep["reference_cache"]["reference_generations"]


def test_another_reference_is_a_miss_and_a_tampered_entry_is_refused(tmp_path, standins, fx):
    _, pr = standins.start(backend=f"replay:{fx['ref']}")
    _, pr2 = standins.start(backend=f"replay:{fx['ref']}")
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    cache = tmp_path / "cache"
    _, rep1 = cmp(pr, pa, fx["bank"], tmp_path / "one", cache)
    _, rep2 = cmp(pr2, pa, fx["bank"], tmp_path / "two", cache)  # another endpoint: another key
    assert rep2["reference_cache"]["entry_found"] is False
    entry = cache / rep1["reference_cache"]["key"]
    outputs = next(entry.glob("outputs-*.jsonl"))
    lines = outputs.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["side"]["text"] = "tampered"
    outputs.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    before = len(list((tmp_path).rglob("report.json")))
    code, rep3 = cmp(pr, pa, fx["bank"], tmp_path / "three", cache, ["--exit-on-verdict"])
    assert code == 3 and rep3["status"] == "refused" and rep3["refusal"]["code"] == "CACHE_CORRUPT"
    assert before >= 2


def test_a_cache_claiming_missing_replicate_lines_is_refused(tmp_path, standins, fx):
    _, pr = standins.start(backend=f"replay:{fx['noisy']}")
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    cache = tmp_path / "cache"
    _, rep1 = cmp(pr, pa, fx["bank"], tmp_path / "one", cache)
    entry = cache / rep1["reference_cache"]["key"]
    meta = json.loads((entry / "meta.json").read_text())
    assert meta["counts"]["1"] > 0
    outputs = entry / meta["outputs_file"]
    kept = [x for x in outputs.read_text().splitlines() if json.loads(x)["replicate"] == 0]
    text = "".join(x + "\n" for x in kept)
    import hashlib

    sha = hashlib.sha256(text.encode()).hexdigest()
    (entry / f"outputs-{sha}.jsonl").write_text(text)
    meta["outputs_file"], meta["outputs_sha256"] = f"outputs-{sha}.jsonl", sha  # counts still claim replicate 1
    (entry / "meta.json").write_text(json.dumps(meta))
    code, rep2 = cmp(pr, pa, fx["bank"], tmp_path / "two", cache, ["--exit-on-verdict"])
    assert code == 3 and rep2["refusal"]["code"] == "CACHE_CORRUPT" and "replicate 1" in rep2["refusal"]["message"]


def test_a_missing_outputs_file_is_a_miss_and_errors_are_never_cached(tmp_path, standins, fx):
    _, pr = standins.start(backend=f"replay:{fx['ref']}", fail_every=9)
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    cache = tmp_path / "cache"
    _, rep1 = cmp(pr, pa, fx["bank"], tmp_path / "one", cache)
    entry = cache / rep1["reference_cache"]["key"]
    meta = json.loads((entry / "meta.json").read_text())
    lines = [json.loads(x) for x in (entry / meta["outputs_file"]).read_text().splitlines()]
    assert lines and all(x["side"]["finish"] != "error" for x in lines)
    (entry / meta["outputs_file"]).unlink()  # as if a concurrent writer had replaced it
    _, rep2 = cmp(pr, pa, fx["bank"], tmp_path / "two", cache)
    assert rep2["status"] == "completed" and rep2["reference_cache"]["entry_found"] is False
    assert any("missing" in n for n in rep2["reference_cache"]["notes"]) and rep2["reference_cache"]["written"]


def test_grades_made_by_another_judge_are_not_inherited(tmp_path, standins):
    items = [{"id": f"j-{i:03d}", "suite": "t_judge", "messages": [{"role": "user", "content": f"Explain {i}."}],
              "tools": None, "template_kwargs": {}, "max_new_tokens": 32,
              "grader": {"type": "judge", "rubric": "a short explanation"}} for i in range(20)]
    bank = write_bank(items, tmp_path / "judge_bank.jsonl")
    rp = write_replay(items, tmp_path / "jr.jsonl", lambda it: f"Because of reason {it['id']}.")
    _, pr = standins.start(backend=f"replay:{rp}")
    _, pc = standins.start(backend=f"replay:{rp}")
    ja, pja = standins.start(backend="synthetic", model_name="judge-a")
    jb, pjb = standins.start(backend="synthetic", model_name="judge-b")
    cache = tmp_path / "cache"
    common = ["--skip-channels", "--floor-probe-items", "0", "--plan-floor", "0"]
    cmp(pr, pc, bank, tmp_path / "a1", cache, common + ["--judge", url(pja)])
    _, rep = cmp(pr, pc, bank, tmp_path / "a2", cache, common + ["--judge", url(pja)])
    assert rep["inheritance"]["grading_calls"].get("reference_cache_inherited") == 20
    before = jb.totals["requests"]
    _, rep = cmp(pr, pc, bank, tmp_path / "b", cache, common + ["--judge", url(pjb)])
    calls = rep["inheritance"]["grading_calls"]
    assert calls.get("reference_cache_regraded") == 20 and calls.get("reference_cache_inherited") is None
    assert jb.totals["requests"] - before >= 20  # judge B graded the cached reference outputs itself
    assert rep["reference_cache"]["reference_generations"] == 0


def test_the_cache_is_refused_with_the_010_rule_before_any_request(tmp_path, standins, fx):
    sr, pr = standins.start(backend=f"replay:{fx['ref']}")
    _, pa = standins.start(backend=f"replay:{fx['cand_a']}")
    code, stdout = run_main(["compare", "--ref", url(pr), "--cand", url(pa), "--bank", fx["bank"], "--out",
                             tmp_path / "o", "--ref-cache", tmp_path / "c", "--criterion", "harmful_abs"])
    assert code == 3 and "harmful_abs" in stdout and sr.totals["requests"] == 0
    assert not (tmp_path / "o").exists() and not (tmp_path / "c").exists()
