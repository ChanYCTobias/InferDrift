# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Experimental: read a requalification plan and reproduce its certificate.

This module backs ``compare --plan``. It has not met its own validation bar,
and this release ships no command that writes a plan; the format may change.

The plan is data. This module parses it with the strict JSON reader (duplicate
keys and non-finite numbers are refused) and checks each statistical field
against its closed domain. It never calls ``exec``, ``eval``,
``compile``, ``importlib`` or ``subprocess`` on anything taken from the file.

Two ways to obtain the grades the certificate needs:

* recorded grades, the ``items.jsonl`` beside the plan (``RECORDED_GRADE_SCHEMA`` rows);
* a candidate run, whose per-item flips are already graded by ``inferdrift``.

Census items are counted exactly. Each audited sub-stratum contributes an exact
hypergeometric bound. ``plan_stats.stratified_total_bound`` combines those
bounds, and the certificate names that implementation. The audit sample is re-derived from
the risks and refused when it is not the seeded draw of the frozen partition.
The calibration flag is attached after the bound and is not an input to it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import domains
from .errors import BadInputError
from .graders import GRADER_VERSION
from .jsonio import StrictJSONError, loads_strict, read_jsonl_strict

PLAN_SCHEMA = "inferdrift/requalification-plan/0"
CERTIFICATE_SCHEMA = "inferdrift/requalification-certificate/0"
RECORDED_GRADE_SCHEMA = "inferdrift/recorded-grade/0"
MAX_PLAN_BYTES = 32 * 1024 * 1024
EXECUTABLE_KEYS = frozenset({"__import__", "compile", "eval", "exec", "os.system", "subprocess"})
PLAN_KEYS = (
    "allocation",
    "alpha",
    "alpha_split",
    "bank_digest",
    "census_ids",
    "census_rate",
    "digest",
    "expected_cost_tokens",
    "feature_charge_rule",
    "feature_cost_status",
    "feature_cost_tokens",
    "full_cost_tokens",
    "full_generation_cost_tokens",
    "generation_cost_by_id",
    "grader_by_id",
    "items_to_run",
    "min_inclusion_rate",
    "min_sample",
    "n_strata_requested",
    "output_tokens_by_id",
    "partition_sha256",
    "population",
    "predictor",
    "primary_measurement",
    "reference_digest",
    "risk_by_id",
    "rng",
    "run_rate",
    "schema",
    "seed",
    "selected_generation_cost_tokens",
    "selected_ids",
    "source_label",
    "strata",
    "suite",
    "tau",
)
_PREDICTOR_REQUIRED = ("kind", "model_sha256", "standard_feature_set", "training_ids", "training_ids_sha256")
_PREDICTOR_OPTIONAL = ("detail", "shuffle_seed", "source_kind")
ITEM_KEYS = (
    "answer_span_margin",
    "change_class",
    "entropy",
    "feature_cost_tokens",
    "features_complete",
    "features_reused",
    "family",
    "generation_cost_tokens",
    "grader_id",
    "harmful",
    "id",
    "min_greedy_margin",
    "output_length",
    "output_tokens",
    "primary_measurement",
    "schema",
    "self_consistency",
    "self_consistency_k",
    "self_consistency_temperature",
    "source",
    "suite",
    "synthetic",
    "task_type",
)
STRATUM_KEYS = (
    "anticipated_sd",
    "census",
    "id",
    "inclusion_rate",
    "population_ids",
    "population_size",
    "sample_ids",
    "sample_size",
)
_OPTIONAL_REAL = (
    "answer_span_margin",
    "entropy",
    "min_greedy_margin",
    "output_length",
    "self_consistency",
    "self_consistency_temperature",
)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _reject_executable(obj: Any, field: str, depth: int = 0) -> None:
    """Refuse a key that would be an instruction to run code. Strings are data."""
    if depth > 64:
        raise BadInputError(field, "nesting deeper than 64 levels")
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in EXECUTABLE_KEYS:
                raise BadInputError(key, "the plan is data; this key is not executed")
            _reject_executable(value, field, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            _reject_executable(value, field, depth + 1)


def _text(value: Any, field: str, *, max_len: int = 200) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len or any(ord(ch) < 32 for ch in value):
        raise BadInputError(field, "must be a short string without control characters", value)
    return value


def _count(value: Any, field: str, *, ge: int = 0, le: int | None = None) -> int:
    return domains.integer(value, field, ge=ge, le=domains.MAX_EXACT_INT if le is None else le)


def _id_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise BadInputError(field, "must be a list of ids", value)
    out = [_text(item, field) for item in value]
    if len(out) != len(set(out)):
        raise BadInputError(field, "ids must be unique")
    return out


def _rate(value: Any, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"denominator", "numerator"}:
        raise BadInputError(field, "must be an object with numerator and denominator")
    num = _count(value["numerator"], f"{field}.numerator")
    den = _count(value["denominator"], f"{field}.denominator", ge=1)
    if num > den:
        raise BadInputError(field, "numerator must be <= denominator", value)


def _string_map(value: Any, field: str, check) -> dict:
    if not isinstance(value, dict):
        raise BadInputError(field, "must be an object")
    for key, item in value.items():
        _text(key, field)
        check(item, f"{field}.{key}")
    return value


def _partition_body(census_ids: list[str], strata: list[dict]) -> dict:
    return {
        "census_ids": list(census_ids),
        "strata": [{"id": row["id"], "population_ids": list(row["population_ids"])} for row in strata],
    }


def load_plan(path: str) -> dict:
    """Read and validate a plan. Raises ``BadInputError`` with a specific field."""
    file = Path(path)
    if not file.is_file():
        raise BadInputError("--plan", "is not a file", path)
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise BadInputError("--plan", f"cannot be read ({type(exc).__name__})") from None
    if len(data) > MAX_PLAN_BYTES:
        raise BadInputError("--plan", f"exceeds {MAX_PLAN_BYTES} bytes", len(data))
    try:
        obj = loads_strict(data)
    except StrictJSONError as exc:
        raise BadInputError("--plan", f"is not strict JSON ({exc})") from None
    return validate_plan_object(obj)


def validate_plan_object(obj: Any) -> dict:
    """Closed plan: domains, digest, partition hash, and the seeded audit draw."""
    if not isinstance(obj, dict):
        raise BadInputError("--plan", "must be a JSON object", obj)
    _reject_executable(obj, "--plan")
    domains.all_finite(obj, "--plan")
    if obj.get("schema") != PLAN_SCHEMA:
        raise BadInputError("schema", f"must be {PLAN_SCHEMA}", obj.get("schema"))
    unknown = sorted(set(obj) - set(PLAN_KEYS))
    if unknown:
        raise BadInputError(unknown[0], "unknown key")
    missing = [key for key in PLAN_KEYS if key not in obj]
    if missing:
        raise BadInputError(missing[0], "missing field")
    _validate_plan_domains(obj)
    body = {key: value for key, value in obj.items() if key != "digest"}
    if _digest(body) != obj["digest"]:
        raise BadInputError("digest", "the plan digest does not match its body")
    populations = [{"id": row["id"], "population_ids": row["population_ids"]} for row in obj["strata"]]
    if _digest(_partition_body(obj["census_ids"], populations)) != obj["partition_sha256"]:
        raise BadInputError("partition_sha256", "the partition hash does not match the stratum membership")
    from .plan_audit import assert_honest_draw

    assert_honest_draw(obj)
    return obj


def _validate_plan_domains(plan: dict) -> None:
    domains.choice(plan["allocation"], "allocation", ("full_census", "stratified_neyman"))
    domains.number(plan["alpha"], "alpha", gt=0.0, lt=1.0)
    if plan["alpha_split"] != "proportional_to_size":
        raise BadInputError("alpha_split", "must be proportional_to_size", plan["alpha_split"])
    domains.boolean(plan["primary_measurement"], "primary_measurement")
    domains.choice(plan["source_label"], "source_label", ("measured", "stored_grades", "synthetic"))
    if plan["primary_measurement"] and plan["source_label"] != "measured":
        raise BadInputError("source_label", "a plan marked primary_measurement must have source_label measured")
    domains.number(plan["tau"], "tau", gt=0.0, lt=1.0)
    _count(plan["seed"], "seed", le=2**32 - 1)
    _count(plan["population"], "population", ge=1)
    _count(plan["min_sample"], "min_sample", ge=1, le=10**7)
    _count(plan["n_strata_requested"], "n_strata_requested", ge=1, le=100)
    if plan["rng"] != "random.Random.sample":
        raise BadInputError("rng", "must be random.Random.sample", plan["rng"])
    if plan["feature_charge_rule"] != "sum_feature_cost_tokens":
        raise BadInputError("feature_charge_rule", "must be sum_feature_cost_tokens", plan["feature_charge_rule"])
    domains.choice(plan["feature_cost_status"], "feature_cost_status",
                   ("features_absent", "partial", "reused"))
    for name in ("expected_cost_tokens", "feature_cost_tokens", "full_cost_tokens",
                 "full_generation_cost_tokens", "selected_generation_cost_tokens"):
        _count(plan[name], name)
    _rate(plan["census_rate"], "census_rate")
    _rate(plan["run_rate"], "run_rate")
    _rate(plan["min_inclusion_rate"], "min_inclusion_rate")
    digest = plan["digest"]
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise BadInputError("digest", "must be 64 lowercase hex characters")
    part = plan["partition_sha256"]
    if not isinstance(part, str) or len(part) != 64 or any(ch not in "0123456789abcdef" for ch in part):
        raise BadInputError("partition_sha256", "must be 64 lowercase hex characters")
    _hex64(plan["reference_digest"], "reference_digest")
    if plan["bank_digest"] is not None:
        _hex64(plan["bank_digest"], "bank_digest")
    _text(plan["suite"], "suite")
    census_ids = _id_list(plan["census_ids"], "census_ids")
    if not isinstance(plan["strata"], list) or not plan["strata"]:
        raise BadInputError("strata", "must be a non-empty list")
    population: list[str] = []
    samples: list[str] = []
    saw_census = False
    for index, row in enumerate(plan["strata"]):
        field = f"strata[{index}]"
        if not isinstance(row, dict) or set(row) != set(STRATUM_KEYS):
            raise BadInputError(field, "stratum keys are not the declared set")
        domains.boolean(row["census"], f"{field}.census")
        _text(row["id"], f"{field}.id", max_len=80)
        pop = _id_list(row["population_ids"], f"{field}.population_ids")
        sample = _id_list(row["sample_ids"], f"{field}.sample_ids")
        pop_n = _count(row["population_size"], f"{field}.population_size", ge=1)
        sample_n = _count(row["sample_size"], f"{field}.sample_size", ge=1)
        if pop_n != len(pop) or sample_n != len(sample) or sample_n > pop_n:
            raise BadInputError(field, "sample size is outside the stratum")
        if any(item not in set(pop) for item in sample):
            raise BadInputError(field, "sample is not inside its stratum")
        rate = row["inclusion_rate"]
        domains.number(rate, f"{field}.inclusion_rate", ge=0.0, le=1.0)
        if rate != sample_n / pop_n:
            raise BadInputError(f"{field}.inclusion_rate", "must equal sample_size / population_size", rate)
        if row["id"] == "census":
            if saw_census:
                raise BadInputError(field, "more than one census stratum")
            saw_census = True
            if row["anticipated_sd"] is not None:
                raise BadInputError(f"{field}.anticipated_sd", "a census stratum has no anticipated sd")
            if row["population_ids"] != census_ids or row["sample_ids"] != census_ids:
                raise BadInputError("census_ids", "must match the census stratum")
        else:
            sd = row["anticipated_sd"]
            if sd is not None:
                domains.number(sd, f"{field}.anticipated_sd", ge=0.0, le=1e6)
        population.extend(pop)
        samples.extend(sample)
    if not saw_census and census_ids:
        raise BadInputError("census_ids", "names items but the plan has no census stratum")
    if len(population) != len(set(population)):
        raise BadInputError("strata", "stratum populations overlap")
    if plan["population"] != len(population):
        raise BadInputError("population", "must equal the number of stratum members", plan["population"])
    selected = _id_list(plan["selected_ids"], "selected_ids")
    if selected != sorted(samples):
        raise BadInputError("selected_ids", "must be the sorted census and audit sample")
    _string_map(plan["grader_by_id"], "grader_by_id", lambda value, field: _text(value, field))
    _string_map(plan["generation_cost_by_id"], "generation_cost_by_id", lambda value, field: _count(value, field))
    _string_map(plan["output_tokens_by_id"], "output_tokens_by_id", lambda value, field: _count(value, field))
    _string_map(plan["risk_by_id"], "risk_by_id", lambda value, field: domains.number(value, field, ge=0.0, le=1.0))
    member_ids = set(population)
    for name in ("grader_by_id", "generation_cost_by_id", "output_tokens_by_id", "risk_by_id"):
        if set(plan[name]) != member_ids:
            raise BadInputError(name, "must name exactly the stratum population")
    if not isinstance(plan["items_to_run"], list):
        raise BadInputError("items_to_run", "must be a list")
    run_ids = []
    census_set = set(census_ids)
    for index, row in enumerate(plan["items_to_run"]):
        field = f"items_to_run[{index}]"
        if not isinstance(row, dict) or set(row) != {"expected_cost_tokens", "id", "role"}:
            raise BadInputError(field, "must have id, role and expected_cost_tokens")
        item_id = _text(row["id"], f"{field}.id")
        role = row["role"]
        if role not in ("audit", "census"):
            raise BadInputError(f"{field}.role", "must be census or audit", role)
        if (item_id in census_set) != (role == "census"):
            raise BadInputError(f"{field}.role", "role does not match census membership", role)
        cost = _count(row["expected_cost_tokens"], f"{field}.expected_cost_tokens")
        if cost != plan["generation_cost_by_id"][item_id]:
            raise BadInputError(f"{field}.expected_cost_tokens", "must equal the item's generation cost", cost)
        run_ids.append(item_id)
    if run_ids != selected:
        raise BadInputError("items_to_run", "must list the selected census and audit items in order")
    _predictor(plan["predictor"])


def _hex64(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise BadInputError(field, "must be 64 lowercase hex characters")


def _predictor(value: Any) -> None:
    if not isinstance(value, dict):
        raise BadInputError("predictor", "must be an object")
    allowed = set(_PREDICTOR_REQUIRED) | set(_PREDICTOR_OPTIONAL)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BadInputError(unknown[0], "unknown key")
    missing = [key for key in _PREDICTOR_REQUIRED if key not in value]
    if missing:
        raise BadInputError(f"predictor.{missing[0]}", "missing field")
    domains.boolean(value["standard_feature_set"], "predictor.standard_feature_set")
    _text(value["kind"], "predictor.kind", max_len=80)
    model = value["model_sha256"]
    if model is not None:
        _hex64(model, "predictor.model_sha256")
    ids = _id_list(value["training_ids"], "predictor.training_ids")
    if value["training_ids_sha256"] != _digest(ids):
        raise BadInputError("predictor.training_ids_sha256", "does not match training_ids")
    if "detail" in value:
        detail = value["detail"]
        if not isinstance(detail, str) or len(detail) > 500:
            raise BadInputError("predictor.detail", "must be a string of at most 500 characters", detail)
    if "shuffle_seed" in value:
        _count(value["shuffle_seed"], "predictor.shuffle_seed", le=2**32 - 1)
    if "source_kind" in value:
        _text(value["source_kind"], "predictor.source_kind", max_len=80)


def _optional_real(value: Any, field: str) -> None:
    if value is None:
        return
    domains.number(value, field, ge=-1e6, le=1e6)


def _check_grade_row(row: Any, where: str) -> dict:
    if not isinstance(row, dict):
        raise BadInputError(where, "must be an object")
    _reject_executable(row, where)
    unknown = sorted(set(row) - set(ITEM_KEYS))
    if unknown:
        raise BadInputError(unknown[0], "unknown key")
    missing = [key for key in ITEM_KEYS if key not in row]
    if missing:
        raise BadInputError(missing[0], "missing field")
    domains.all_finite(row, where)
    if row["schema"] != RECORDED_GRADE_SCHEMA:
        raise BadInputError("schema", f"must be {RECORDED_GRADE_SCHEMA}", row["schema"])
    _text(row["id"], f"{where}.id")
    _text(row["suite"], f"{where}.suite")
    _text(row["grader_id"], f"{where}.grader_id")
    _text(row["change_class"], f"{where}.change_class")
    _text(row["task_type"], f"{where}.task_type")
    domains.choice(row["source"], f"{where}.source", ("measured", "stored", "synthetic"))
    for name in ("features_complete", "features_reused", "primary_measurement", "synthetic"):
        domains.boolean(row[name], f"{where}.{name}")
    if row["family"] is not None:
        _text(row["family"], f"{where}.family")
    harmful = row["harmful"]
    if isinstance(harmful, bool) or harmful not in (0, 1):
        raise BadInputError(f"{where}.harmful", "must be 0 or 1", harmful)
    for name in _OPTIONAL_REAL:
        _optional_real(row[name], f"{where}.{name}")
    for name in ("feature_cost_tokens", "generation_cost_tokens", "output_tokens"):
        _count(row[name], f"{where}.{name}")
    if row["self_consistency_k"] is not None:
        _count(row["self_consistency_k"], f"{where}.self_consistency_k", ge=1)
    if row["output_length"] is not None and row["output_length"] < 0:
        raise BadInputError(f"{where}.output_length", "must be >= 0", row["output_length"])
    if row["self_consistency"] is not None and not 0.0 <= float(row["self_consistency"]) <= 1.0:
        raise BadInputError(f"{where}.self_consistency", "must lie in [0, 1]", row["self_consistency"])
    return row


def load_recorded_grades(plan_path: str, plan: Mapping[str, Any]) -> dict[str, dict]:
    """Grades from ``items.jsonl`` in the plan's directory. One suite, the plan's population."""
    path = Path(plan_path).resolve().parent / "items.jsonl"
    if not path.is_file():
        raise BadInputError("grades", "recorded grades are not beside the plan (items.jsonl)")
    try:
        parsed = read_jsonl_strict(path)
    except StrictJSONError as exc:
        raise BadInputError("grades", f"items.jsonl is not strict JSON ({exc})") from None
    except OSError as exc:
        raise BadInputError("grades", f"items.jsonl cannot be read ({type(exc).__name__})") from None
    rows = [_check_grade_row(row, f"grades line {no}") for no, row in parsed]
    if not rows:
        raise BadInputError("grades", "items.jsonl has no grades")
    suites = {row["suite"] for row in rows}
    if len(suites) != 1 or plan["suite"] not in suites:
        raise BadInputError("suite", "plan is for another suite")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise BadInputError("id", "duplicate item id")
    if _digest(sorted(rows, key=lambda row: row["id"])) != plan["reference_digest"]:
        raise BadInputError("reference_digest", "plan is for another suite")
    if set(ids) != set(plan["grader_by_id"]):
        raise BadInputError("suite", "plan is for another suite")
    return {row["id"]: {"grader_id": row["grader_id"], "harmful": int(row["harmful"])} for row in rows}


def selected_bank_items(plan: Mapping[str, Any], items: list[dict]) -> list[dict]:
    """Bank rows for the census and the audit, in the bank's order.

    Items that are not selected are not returned, so a later compare does not
    send them to the candidate. A bank that does not contain the selected ids
    is a plan for another suite.
    """
    from .banks import bank_digest

    for item in items:
        if item.get("suite") != plan.get("suite"):
            raise BadInputError("suite", "plan is for another suite", item.get("suite"))
    if plan.get("bank_digest") is None:
        raise BadInputError("bank_digest", "this plan records no bank digest, so a bank is refused")
    if bank_digest(items) != plan["bank_digest"]:
        raise BadInputError("bank_digest", "plan is for another suite")
    by_id: dict[str, dict] = {}
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id in by_id:
            raise BadInputError("bank", "duplicate or missing item id", item_id)
        by_id[item_id] = item
    missing = [item_id for item_id in plan["selected_ids"] if item_id not in by_id]
    if missing:
        raise BadInputError("suite", "plan is for another suite", missing[0])
    for item_id in plan["selected_ids"]:
        grader = by_id[item_id].get("grader") or {}
        kind = grader.get("type") if isinstance(grader, dict) else None
        expected = f"{kind}@{GRADER_VERSION}"
        if plan["grader_by_id"].get(item_id) != expected:
            raise BadInputError(
                "grader_id",
                f"{item_id} bank grader {expected} differs from the frozen grader "
                f"{plan['grader_by_id'].get(item_id)!r}",
            )
    selected = set(plan["selected_ids"])
    return [item for item in items if item["id"] in selected]


def grades_from_compare_records(plan: Mapping[str, Any], records: list[dict]) -> dict[str, dict]:
    """Harmful flips from a compare ``items.jsonl``. Unselected ids are refused."""
    grades: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise BadInputError("items", "a compare record must be an object")
        item_id = record.get("id")
        flip = record.get("flip")
        cand = record.get("cand_grade") if isinstance(record.get("cand_grade"), dict) else {}
        grader = cand.get("grader")
        if not isinstance(item_id, str) or not isinstance(grader, str):
            raise BadInputError("items", "a compare record needs an id and a candidate grader")
        if isinstance(flip, bool) or not isinstance(flip, int):
            raise BadInputError("flip", "must be an integer", flip)
        if item_id in grades:
            raise BadInputError("id", "duplicate compare record", item_id)
        grades[item_id] = {"grader_id": f"{grader}@{GRADER_VERSION}", "harmful": 1 if flip < 0 else 0}
    selected = list(plan["selected_ids"])
    missing = [item_id for item_id in selected if item_id not in grades]
    if missing:
        raise BadInputError("plan", "the candidate run did not grade every census and audit item", missing[0])
    extra = sorted(set(grades) - set(selected))
    if extra:
        raise BadInputError("plan", "the candidate run graded an item outside the census and the audit", extra[0])
    return grades


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + stop) / 2.0 + 1.0
        for index in range(start, stop + 1):
            ranks[order[index]] = average
        start = stop + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mean_x = math.fsum(rx) / n
    mean_y = math.fsum(ry) / n
    num = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    dx = math.sqrt(math.fsum((a - mean_x) ** 2 for a in rx))
    dy = math.sqrt(math.fsum((b - mean_y) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _calibration(risks: list[float], harms: list[int]) -> dict:
    coefficient = _spearman(risks, [float(value) for value in harms])
    if len(risks) < 8:
        return {"coefficient": coefficient, "flagged": False, "reason": "TOO_FEW_POINTS"}
    if coefficient is None:
        return {"coefficient": None, "flagged": False, "reason": "NO_RISK_VARIATION"}
    if coefficient < 0:
        return {"coefficient": coefficient, "flagged": True, "reason": "NEGATIVE_RANK_CORRELATION"}
    return {"coefficient": coefficient, "flagged": False, "reason": "RANK_CORRELATION_NON_NEGATIVE"}


def certify(plan: Mapping[str, Any], grades: Mapping[str, Mapping[str, Any]]) -> dict:
    """The plan's certificate. Grades outside the census and audit are not read for harms."""
    from .plan_stats import PlanStatsError, population_bound

    selected = set(plan["selected_ids"])
    missing = sorted(selected - set(grades))
    if missing:
        raise BadInputError("grades", "missing grade for a census or audit item", missing[0])
    used = {item_id: grades[item_id] for item_id in selected}
    ignored = sorted(set(grades) - selected)
    bound_rows = []
    estimate_rows = []
    grade_ids: list[str] = []
    for stratum in plan["strata"]:
        sample_ids = list(stratum["sample_ids"])
        harms = []
        for item_id in sample_ids:
            grade = used[item_id]
            if grade.get("grader_id") != plan["grader_by_id"].get(item_id):
                raise BadInputError(
                    "grader_id",
                    f"{item_id} grader {grade.get('grader_id')!r} differs from the frozen grader "
                    f"{plan['grader_by_id'].get(item_id)!r}",
                )
            harmful = grade.get("harmful")
            if isinstance(harmful, bool) or harmful not in (0, 1):
                raise BadInputError("harmful", f"{item_id} must be 0 or 1", harmful)
            harms.append(int(harmful))
            grade_ids.append(item_id)
        bound_rows.append({
            "population_size": len(stratum["population_ids"]),
            "sample_size": len(sample_ids),
            "successes": sum(harms),
        })
        estimate_rows.append({
            "population_size": len(stratum["population_ids"]),
            "values": [float(value) for value in harms],
        })
    try:
        bound, estimate, method = population_bound(
            bound_rows, estimate_rows, alpha=plan["alpha"], alpha_split=plan["alpha_split"],
        )
    except PlanStatsError as exc:
        raise BadInputError("plan", f"{exc.reason}: {exc.field}: {exc.detail}") from None
    upper_rate = bound["upper_count"] / bound["population"]
    grades_used = sorted(grade_ids)
    risks = [float(plan["risk_by_id"][item_id]) for item_id in grades_used]
    harms = [int(used[item_id]["harmful"]) for item_id in grades_used]
    certified = upper_rate <= plan["tau"]
    return {
        "alpha": plan["alpha"],
        "bound": bound,
        "calibration": _calibration(risks, harms),
        "certified": certified,
        "estimate": estimate,
        "grades_used": grades_used,
        "ignored_grade_ids": ignored,
        "method": method,
        "plan_digest": plan["digest"],
        "population": bound["population"],
        "schema": CERTIFICATE_SCHEMA,
        "tau": plan["tau"],
        "upper_count": bound["upper_count"],
        "upper_rate": upper_rate,
        "verdict": "CERTIFIED" if certified else "NOT_CERTIFIED",
    }
