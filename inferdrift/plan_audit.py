# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Experimental: freeze a risk partition and draw its simple random audit.

A plan writer and both plan consumers call these functions, so a resealed plan
cannot name a sample this draw did not produce. The stored seed is the first
32 bits of the partition hash. A caller-chosen seed is not the draw.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Sequence

from .errors import BadInputError
from .plan_stats import PlanStatsError, allocate

FEATURE_CHARGE_RULE = "sum_feature_cost_tokens"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def derived_seed(partition_sha256: str) -> int:
    return int(partition_sha256[:8], 16)


def _refuse(code: str, field: str, problem: str) -> None:
    error = BadInputError(field, f"{code}: {problem}")
    error.code = code
    raise error


def _sd(risks: Sequence[float]) -> float:
    mean = math.fsum(risks) / len(risks)
    variance = mean * (1.0 - mean)
    sd = math.sqrt(variance) if variance > 0 else 0.0
    return sd if sd > 0 else 1e-6


def _chunks(ids: list[str], requested: int) -> list[list[str]]:
    if not ids:
        return []
    width = min(requested, len(ids))
    base, extra = divmod(len(ids), width)
    out = []
    start = 0
    for index in range(width):
        size = base + (1 if index < extra else 0)
        out.append(ids[start:start + size])
        start += size
    return out


def partition_body(census_ids: Sequence[str], strata_populations: Sequence[dict]) -> dict:
    return {
        "census_ids": list(census_ids),
        "strata": [{"id": row["id"], "population_ids": list(row["population_ids"])} for row in strata_populations],
    }


def freeze(items: Sequence[dict], risks: Sequence[float], *, census_count: int, n_strata: int) -> dict:
    """Stratum membership from risks. No sample is drawn here."""
    order = sorted(range(len(items)), key=lambda index: (-risks[index], items[index]["id"]))
    census_ids = sorted(items[index]["id"] for index in order[:census_count])
    rest = sorted(order[census_count:], key=lambda index: (risks[index], items[index]["id"]))
    rest_ids = [items[index]["id"] for index in rest]
    by_id = {item["id"]: (item, risks[index]) for index, item in enumerate(items)}
    populations = [{"census": True, "id": "census", "population_ids": census_ids}] if census_ids else []
    for index, chunk in enumerate(_chunks(rest_ids, n_strata)):
        populations.append({"census": False, "id": f"audit-{index}", "population_ids": list(chunk)})
    return {"by_id": by_id, "census_ids": census_ids, "populations": populations}


def draw(
    partition: dict,
    *,
    audit_budget: int,
    min_inclusion_rate: tuple[int, int],
    min_sample: int,
    seed: int,
) -> dict:
    """Simple random sample inside each frozen audit stratum, one RNG stream."""
    audit = [row for row in partition["populations"] if not row["census"]]
    samples = {row["id"]: list(row["population_ids"]) for row in partition["populations"] if row["census"]}
    if not audit:
        return {"anticipated_sd": {}, "samples": samples}
    if audit_budget <= 0:
        raise BadInputError("audit_budget", "a non-empty remainder needs an audit sample")
    sizes = [len(row["population_ids"]) for row in audit]
    anticipated = [_sd([partition["by_id"][item_id][1] for item_id in row["population_ids"]]) for row in audit]
    rate = float(min_inclusion_rate[0]) / float(min_inclusion_rate[1])
    try:
        sample_sizes = allocate(sizes, anticipated, budget=audit_budget, min_sample=min_sample, min_inclusion_rate=rate)
    except PlanStatsError as exc:
        raise BadInputError("allocation", f"{exc.reason}: {exc.field}: {exc.detail}") from None
    rng = random.Random(seed)
    anticipated_by_id = {}
    for row, size, sd in zip(audit, sample_sizes, anticipated):
        population = list(row["population_ids"])
        if size == len(population):
            chosen = population
        else:
            chosen = sorted(rng.sample(population, int(size)))
        samples[row["id"]] = chosen
        anticipated_by_id[row["id"]] = sd
    return {"anticipated_sd": anticipated_by_id, "samples": samples}


def _counts(plan: dict) -> tuple[int, int]:
    population = plan["population"]
    if plan["allocation"] == "full_census":
        return population, 0
    census_count = (plan["census_rate"]["numerator"] * population) // plan["census_rate"]["denominator"]
    run_count = max(census_count, (plan["run_rate"]["numerator"] * population) // plan["run_rate"]["denominator"])
    if run_count > population:
        run_count = population
    return census_count, run_count - census_count


def assert_honest_draw(plan: dict) -> None:
    """Refuse a plan whose census, strata or audit sample is not this draw.

    The seed check runs first. A plan that keeps the honest sample and changes
    only the seed is ``AUDIT_DRAWN_AFTER_CENSUS``. A plan that keeps the seed
    and changes the sample is ``TARGETED_AUDIT``. An audit id recorded in the
    predictor's training set is ``PREDICTOR_RETUNED_ON_AUDIT``.
    """
    expected_seed = derived_seed(plan["partition_sha256"])
    if plan["seed"] != expected_seed:
        _refuse(
            "AUDIT_DRAWN_AFTER_CENSUS",
            "seed",
            "the audit seed is not derived from the frozen partition",
        )
    ids = sorted(plan["risk_by_id"])
    items = [{"id": item_id} for item_id in ids]
    risks = [float(plan["risk_by_id"][item_id]) for item_id in ids]
    census_count, audit_budget = _counts(plan)
    frozen = freeze(items, risks, census_count=census_count, n_strata=plan["n_strata_requested"])
    if list(frozen["census_ids"]) != list(plan["census_ids"]):
        _refuse("TARGETED_AUDIT", "census_ids", "census membership is not the frozen partition of these risks")
    if [row["id"] for row in frozen["populations"]] != [row["id"] for row in plan["strata"]]:
        _refuse("TARGETED_AUDIT", "strata", "stratum ids are not the frozen partition of these risks")
    if [list(row["population_ids"]) for row in frozen["populations"]] != [list(row["population_ids"]) for row in plan["strata"]]:
        _refuse("TARGETED_AUDIT", "population_ids", "stratum membership is not the frozen partition of these risks")
    inclusion = plan["min_inclusion_rate"]
    drawn = draw(
        frozen,
        audit_budget=audit_budget,
        min_inclusion_rate=(inclusion["numerator"], inclusion["denominator"]),
        min_sample=plan["min_sample"],
        seed=expected_seed,
    )
    for row in plan["strata"]:
        if list(row["sample_ids"]) != list(drawn["samples"][row["id"]]):
            _refuse(
                "TARGETED_AUDIT",
                "sample_ids",
                "the audit sample is not the seeded random draw of the frozen partition",
            )
    training = set(plan["predictor"]["training_ids"])
    audit_ids = [item_id for row in plan["strata"] if row["id"] != "census" for item_id in row["sample_ids"]]
    overlap = sorted(set(audit_ids) & training)
    if overlap:
        _refuse(
            "PREDICTOR_RETUNED_ON_AUDIT",
            "predictor.training_ids",
            "an audit id is in the predictor training set",
        )
