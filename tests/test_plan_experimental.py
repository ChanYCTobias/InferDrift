# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The experimental ``compare --plan`` path, on a small synthetic plan built with the package's own functions."""

from __future__ import annotations

import copy
import json
import random

from helpers import run_main, strict_last_line
from inferdrift import plan_audit
from inferdrift.graders import GRADER_VERSION
from inferdrift.jsonio import loads_strict
from inferdrift.requalify import PLAN_SCHEMA, RECORDED_GRADE_SCHEMA

N = 60


def _grade_rows() -> list[dict]:
    rng = random.Random(3)
    rows = []
    for i in range(N):
        rows.append({
            "answer_span_margin": None, "change_class": "synthetic", "entropy": None,
            "feature_cost_tokens": 0, "features_complete": False, "features_reused": False, "family": None,
            "generation_cost_tokens": 100, "grader_id": f"numeric@{GRADER_VERSION}", "harmful": int(rng.random() < 0.1),
            "id": f"item-{i:03d}", "min_greedy_margin": None, "output_length": None, "output_tokens": 20,
            "primary_measurement": False, "schema": RECORDED_GRADE_SCHEMA, "self_consistency": None,
            "self_consistency_k": None, "self_consistency_temperature": None, "source": "synthetic",
            "suite": "synthetic_suite",
            "synthetic": True, "task_type": "numeric"})
    return rows


def _build_plan(rows: list[dict]) -> dict:
    rng = random.Random(4)
    ids = sorted(r["id"] for r in rows)
    risks = {item_id: round(rng.random(), 6) for item_id in ids}
    census_rate = {"numerator": 1, "denominator": 10}
    run_rate = {"numerator": 1, "denominator": 2}
    min_inclusion_rate = {"numerator": 1, "denominator": 10}
    census_count = N * census_rate["numerator"] // census_rate["denominator"]
    audit_budget = N * run_rate["numerator"] // run_rate["denominator"] - census_count
    frozen = plan_audit.freeze([{"id": i} for i in ids], [risks[i] for i in ids], census_count=census_count,
                               n_strata=3)
    partition = plan_audit.digest(plan_audit.partition_body(frozen["census_ids"], frozen["populations"]))
    seed = plan_audit.derived_seed(partition)
    drawn = plan_audit.draw(frozen, audit_budget=audit_budget, min_inclusion_rate=(1, 10), min_sample=2, seed=seed)
    strata = []
    for row in frozen["populations"]:
        pop, sample = list(row["population_ids"]), list(drawn["samples"][row["id"]])
        strata.append({"anticipated_sd": None if row["census"] else drawn["anticipated_sd"][row["id"]],
                       "census": row["census"], "id": row["id"], "inclusion_rate": len(sample) / len(pop),
                       "population_ids": pop, "population_size": len(pop), "sample_ids": sample,
                       "sample_size": len(sample)})
    selected = sorted(i for s in strata for i in s["sample_ids"])
    census = set(frozen["census_ids"])
    plan = {
        "allocation": "stratified_neyman", "alpha": 0.05, "alpha_split": "proportional_to_size",
        "bank_digest": None, "census_ids": frozen["census_ids"],
        "census_rate": census_rate, "expected_cost_tokens": 100 * len(selected), "feature_charge_rule":
        "sum_feature_cost_tokens", "feature_cost_status": "features_absent", "feature_cost_tokens": 0,
        "full_cost_tokens": 100 * N, "full_generation_cost_tokens": 100 * N,
        "generation_cost_by_id": {i: 100 for i in ids}, "grader_by_id": {i: f"numeric@{GRADER_VERSION}" for i in ids},
        "items_to_run": [{"expected_cost_tokens": 100, "id": i, "role": "census" if i in census else "audit"}
                         for i in selected],
        "min_inclusion_rate": min_inclusion_rate, "min_sample": 2, "n_strata_requested": 3,
        "output_tokens_by_id": {i: 20 for i in ids}, "partition_sha256": partition, "population": N,
        "primary_measurement": False,
        "predictor": {"kind": "synthetic", "model_sha256": None, "standard_feature_set": False, "training_ids": [],
                      "training_ids_sha256": plan_audit.digest([])},
        "reference_digest": plan_audit.digest(sorted(rows, key=lambda r: r["id"])), "risk_by_id": risks,
        "rng": "random.Random.sample", "run_rate": run_rate, "schema": PLAN_SCHEMA, "seed": seed,
        "selected_generation_cost_tokens": 100 * len(selected), "selected_ids": selected,
        "source_label": "synthetic", "strata": strata, "suite": "synthetic_suite", "tau": 0.3,
    }
    plan["digest"] = plan_audit.digest(plan)
    return plan


def _write(tmp_path, plan: dict, rows: list[dict]):
    d = tmp_path / "plan"
    d.mkdir(exist_ok=True)
    (d / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (d / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d / "plan.json"


def test_certificate_from_recorded_grades(tmp_path):
    rows = _grade_rows()
    plan = _build_plan(rows)
    code, out = run_main(["compare", "--plan", _write(tmp_path, plan, rows)])
    cert = loads_strict(out)
    assert code == 0
    assert cert["schema"] == "inferdrift/requalification-certificate/0"
    assert cert["method"] == "inferdrift.plan_stats.stratified_total_bound"
    assert cert["grades_used"] == plan["selected_ids"]
    assert cert["population"] == N
    assert cert["verdict"] in ("CERTIFIED", "NOT_CERTIFIED")
    harmful = sum(r["harmful"] for r in rows)
    census_harmful = sum(r["harmful"] for r in rows if r["id"] in set(plan["census_ids"]))
    assert census_harmful <= cert["upper_count"] <= N
    assert cert["bound"]["census_total"] == census_harmful
    assert 0 <= harmful <= N


def test_a_targeted_audit_is_refused(tmp_path):
    rows = _grade_rows()
    plan = _build_plan(rows)
    tampered = copy.deepcopy(plan)
    audit = next(s for s in tampered["strata"] if not s["census"])
    outside = next(i for i in audit["population_ids"] if i not in audit["sample_ids"])
    audit["sample_ids"] = sorted(audit["sample_ids"][1:] + [outside])
    tampered["selected_ids"] = sorted(i for s in tampered["strata"] for i in s["sample_ids"])
    census = set(tampered["census_ids"])
    tampered["items_to_run"] = [{"expected_cost_tokens": 100, "id": i, "role": "census" if i in census else "audit"}
                                for i in tampered["selected_ids"]]
    tampered.pop("digest")
    tampered["digest"] = plan_audit.digest(tampered)  # a correctly resealed plan that names another sample
    code, out = run_main(["compare", "--plan", _write(tmp_path, tampered, rows)])
    line = strict_last_line(out)
    assert code == 3 and "TARGETED_AUDIT" in line["message"]


def test_an_edited_plan_fails_its_digest(tmp_path):
    rows = _grade_rows()
    plan = _build_plan(rows)
    plan["tau"] = 0.5
    code, out = run_main(["compare", "--plan", _write(tmp_path, plan, rows)])
    line = strict_last_line(out)
    assert code == 3 and line["field"] == "digest"


def test_an_empty_plan_path_is_refused():
    code, out = run_main(["compare", "--plan", ""])
    assert code == 3 and strict_last_line(out)["field"] == "--plan"
