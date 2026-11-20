# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Stratified bound, Horvitz-Thompson estimate and Neyman allocation for a plan.

Experimental (used only by ``compare --plan``). Standard library only. The
arithmetic is exact integer hypergeometric tails, the float64 Bonferroni split,
and Neyman's exact integer allocation including ``Fraction`` of the float
inclusion rate. The certificate's ``method`` names this implementation.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

DBL_MIN = sys.float_info.min
MAX_ABS_VALUE = 1e100
MAX_EXACT_INT = 2**53 - 1
MAX_POPULATION = 100_000
MAX_STRATA = 10_000
MAX_TOTAL_POPULATION = 10_000_000
HT_KEYS = ("population_size", "values")
BOUND_KEYS = ("population_size", "sample_size", "successes")


class PlanStatsError(Exception):
    def __init__(self, reason: str, field: str, detail: str):
        self.reason = reason
        self.field = field
        self.detail = detail
        super().__init__(f"{reason}: {field}: {detail}")


def _fail(reason: str, field: str, detail: str) -> None:
    raise PlanStatsError(reason, field, detail)


def _count(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("WRONG_TYPE", field, "counts must be integers")
    if value < 0:
        _fail("NEGATIVE_COUNT", field, f"count {value} is negative")
    if value > maximum:
        _fail("COUNT_OUT_OF_RANGE", field, f"count {value} exceeds {maximum}")
    return value


def _real(value: object, field: str) -> float:
    if isinstance(value, bool):
        _fail("BOOL_NOT_NUMERIC", field, "booleans are not numbers")
    if isinstance(value, int):
        if abs(value) > MAX_EXACT_INT:
            _fail("VALUE_OUT_OF_BOUNDS", field, "integer outside the exact float64 range")
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE", field, f"value is {value!r}")
        return value
    _fail("WRONG_TYPE", field, f"expected a number, got {type(value).__name__}")
    return 0.0


def exact_level(value: float, exact: Fraction) -> Fraction:
    return Fraction(value) if value >= DBL_MIN else exact


def hyper_support(n: int, population: int, successes: int) -> tuple[int, int]:
    return max(0, n - (population - successes)), min(n, successes)


def hyper_term_sum(n: int, population: int, successes: int, j_from: int, j_to: int) -> int:
    lo, hi = hyper_support(n, population, successes)
    start = max(j_from, lo)
    stop = min(j_to, hi)
    if start > stop:
        return 0
    term = math.comb(successes, start) * math.comb(population - successes, n - start)
    total = term
    for index in range(start, stop):
        term = term * (successes - index) * (n - index) // ((index + 1) * (population - successes - n + index + 1))
        total += term
    return total


def hyper_cdf_count(k: int, n: int, population: int, successes: int) -> int:
    lo, hi = hyper_support(n, population, successes)
    if k < lo:
        return 0
    if k >= hi:
        return math.comb(population, n)
    if (k - lo) <= (hi - k):
        return hyper_term_sum(n, population, successes, lo, k)
    return math.comb(population, n) - hyper_term_sum(n, population, successes, k + 1, hi)


def hyper_sf_count(k: int, n: int, population: int, successes: int) -> int:
    lo, hi = hyper_support(n, population, successes)
    if k <= lo:
        return math.comb(population, n)
    if k > hi:
        return 0
    if (hi - k) <= (k - lo):
        return hyper_term_sum(n, population, successes, k, hi)
    return math.comb(population, n) - hyper_term_sum(n, population, successes, lo, k - 1)


def exceeds_alpha(count: int, total: int, alpha: Fraction) -> bool:
    return count * alpha.denominator > alpha.numerator * total


def hyper_upper_k(k: int, n: int, population: int, alpha: float | Fraction) -> int:
    level = alpha if isinstance(alpha, Fraction) else Fraction(alpha)
    total = math.comb(population, n)
    lo = k
    hi = population - (n - k)
    if exceeds_alpha(hyper_cdf_count(k, n, population, hi), total, level):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exceeds_alpha(hyper_cdf_count(k, n, population, mid), total, level):
            lo = mid
        else:
            hi = mid
    return lo


def hyper_lower_k(k: int, n: int, population: int, alpha: float | Fraction) -> int:
    level = alpha if isinstance(alpha, Fraction) else Fraction(alpha)
    total = math.comb(population, n)
    hi = population - (n - k)
    lo = k
    if exceeds_alpha(hyper_sf_count(k, n, population, lo), total, level):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exceeds_alpha(hyper_sf_count(k, n, population, mid), total, level):
            hi = mid
        else:
            lo = mid
    return hi


def _split(alpha_side: float, alpha_side_exact: Fraction, sizes: Sequence[int], how: str) -> list[Fraction]:
    if how == "equal":
        share = exact_level(alpha_side / len(sizes), alpha_side_exact / len(sizes))
        return [share] * len(sizes)
    total = float(sum(sizes))
    total_exact = sum(sizes)
    return [
        exact_level(alpha_side * (size / total), alpha_side_exact * Fraction(size, total_exact))
        for size in sizes
    ]


@dataclass(frozen=True)
class StratifiedCountInterval:
    lower_count: int
    upper_count: int
    population: int
    alpha: float
    side: str
    method: str
    census_total: int
    stratum_alphas: tuple[float, ...]
    stratum_lower: tuple[int, ...]
    stratum_upper: tuple[int, ...]
    stratum_log_alphas: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        body = asdict(self)
        body["lower_fraction"] = self.lower_count / self.population
        body["upper_fraction"] = self.upper_count / self.population
        return body


@dataclass(frozen=True)
class HTEstimate:
    total: float
    mean: float
    variance_total: float | None
    variance_mean: float | None
    variance_estimable: bool
    population: int
    sample_sizes: tuple[int, ...]
    method: str = "stratified_horvitz_thompson"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Allocation:
    sample_sizes: tuple[int, ...]
    census: tuple[bool, ...]
    lower_bounds: tuple[int, ...]
    continuous_neyman: tuple[float, ...]
    anticipated_variance_total: float
    method: str = "neyman_exact_integer"


def _bound_rows(strata: object) -> list[tuple[int, int, int]]:
    if not isinstance(strata, (list, tuple)) or not strata:
        _fail("EMPTY_INPUT", "strata", "no strata given")
    if len(strata) > MAX_STRATA:
        _fail("COUNT_OUT_OF_RANGE", "strata", f"more than {MAX_STRATA} strata")
    out = []
    for index, row in enumerate(strata):
        if not isinstance(row, Mapping):
            _fail("WRONG_TYPE", f"strata[{index}]", "expected a mapping")
        missing = [key for key in BOUND_KEYS if key not in row]
        if missing:
            _fail("MISSING_FIELD", f"strata[{index}]", f"missing {missing}")
        extra = sorted(str(key) for key in set(row) - set(BOUND_KEYS))
        if extra:
            _fail("UNKNOWN_KEY", f"strata[{index}]", f"unknown keys {extra}")
        population = _count(row["population_size"], f"strata[{index}].population_size", maximum=MAX_POPULATION)
        sample = _count(row["sample_size"], f"strata[{index}].sample_size", maximum=MAX_POPULATION)
        successes = _count(row["successes"], f"strata[{index}].successes", maximum=MAX_POPULATION)
        if population < 1 or sample < 1 or sample > population or successes > sample:
            _fail("INCONSISTENT_COUNTS", f"strata[{index}]", "need 1 <= successes <= sample <= population")
        out.append((successes, sample, population))
    return out


def stratified_total_bound(strata: Sequence[Mapping[str, int]], *, alpha: float, side: str, alpha_split: str) -> StratifiedCountInterval:
    parsed = _bound_rows(strata)
    level = _real(alpha, "alpha")
    if not 0.0 < level < 1.0:
        _fail("ALPHA_OUT_OF_RANGE", "alpha", f"alpha {level!r} is not in (0, 1)")
    if side not in ("lower", "upper", "two-sided"):
        _fail("UNKNOWN_ENUM", "side", side)
    if alpha_split not in ("equal", "proportional_to_size"):
        _fail("UNKNOWN_ENUM", "alpha_split", alpha_split)
    population = sum(size for (_k, _n, size) in parsed)
    census_total = sum(successes for (successes, sample, size) in parsed if sample == size)
    sampled = [(successes, sample, size) for (successes, sample, size) in parsed if sample < size]
    if not sampled:
        return StratifiedCountInterval(
            census_total, census_total, population, level, side,
            "stratified_hypergeometric_bonferroni", census_total, (), (), (),
        )
    if side == "two-sided":
        per_side, per_side_exact = level / 2.0, Fraction(level) / 2
    else:
        per_side, per_side_exact = level, Fraction(level)
    alphas = _split(per_side, per_side_exact, [size for (_k, _n, size) in sampled], alpha_split)
    lowers: list[int] = []
    uppers: list[int] = []
    for (successes, sample, size), share in zip(sampled, alphas):
        if side in ("lower", "two-sided"):
            lowers.append(hyper_lower_k(successes, sample, size, share))
        else:
            lowers.append(successes)
        if side in ("upper", "two-sided"):
            uppers.append(hyper_upper_k(successes, sample, size, share))
        else:
            uppers.append(size - (sample - successes))
    return StratifiedCountInterval(
        census_total + sum(lowers),
        census_total + sum(uppers),
        population,
        level,
        side,
        "stratified_hypergeometric_bonferroni",
        census_total,
        tuple(float(share) for share in alphas),
        tuple(lowers),
        tuple(uppers),
        tuple(math.log(share.numerator) - math.log(share.denominator) for share in alphas),
    )


def _sample_var(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def ht_estimate(strata: Sequence[Mapping[str, object]]) -> HTEstimate:
    if not isinstance(strata, (list, tuple)) or not strata:
        _fail("EMPTY_INPUT", "strata", "no strata given")
    if len(strata) > MAX_STRATA:
        _fail("COUNT_OUT_OF_RANGE", "strata", f"more than {MAX_STRATA} strata")
    parsed: list[tuple[int, list[float]]] = []
    total_n = 0
    for index, row in enumerate(strata):
        if not isinstance(row, Mapping):
            _fail("WRONG_TYPE", f"strata[{index}]", "expected a mapping")
        missing = [key for key in HT_KEYS if key not in row]
        if missing:
            _fail("MISSING_FIELD", f"strata[{index}]", f"missing {missing}")
        extra = sorted(str(key) for key in set(row) - set(HT_KEYS))
        if extra:
            _fail("UNKNOWN_KEY", f"strata[{index}]", f"unknown keys {extra}")
        population = _count(row["population_size"], f"strata[{index}].population_size", maximum=MAX_TOTAL_POPULATION)
        if population < 1:
            _fail("INCONSISTENT_COUNTS", f"strata[{index}].population_size", "must be >= 1")
        raw = row["values"]
        if not isinstance(raw, (list, tuple)):
            _fail("WRONG_TYPE", f"strata[{index}].values", "expected a sequence")
        values = [_real(value, f"strata[{index}].values[{i}]") for i, value in enumerate(raw)]
        if any(abs(value) > MAX_ABS_VALUE for value in values):
            _fail("VALUE_OUT_OF_BOUNDS", f"strata[{index}].values", "value outside the declared magnitude")
        if len(values) > population or len(values) < 1:
            _fail("INCONSISTENT_COUNTS", f"strata[{index}].values", "need 1 <= n_h <= N_h")
        total_n += population
        parsed.append((population, values))
    if total_n > MAX_TOTAL_POPULATION:
        _fail("COUNT_OUT_OF_RANGE", "strata", f"total population exceeds {MAX_TOTAL_POPULATION}")
    population = sum(size for size, _values in parsed)
    totals = []
    variances = []
    estimable = True
    for size, values in parsed:
        sample = len(values)
        totals.append(size * (math.fsum(values) / sample))
        if sample == size:
            continue
        if sample < 2:
            estimable = False
            continue
        variances.append(size * size * (1.0 - sample / size) * _sample_var(values) / sample)
    total = math.fsum(totals)
    sizes = tuple(len(values) for _size, values in parsed)
    if estimable:
        variance = math.fsum(variances)
        return HTEstimate(total, total / population, variance, variance / (population * population), True, population, sizes)
    return HTEstimate(total, total / population, None, None, False, population, sizes)


def neyman_allocation(
    population_sizes: Sequence[int],
    anticipated_sds: Sequence[float],
    *,
    budget: int,
    census: Sequence[bool],
    min_sample: int,
    min_inclusion_rate: float,
) -> Allocation:
    if not isinstance(population_sizes, (list, tuple)):
        _fail("WRONG_TYPE", "population_sizes", "expected a sequence of counts")
    sizes = [_count(value, f"population_sizes[{i}]", maximum=MAX_TOTAL_POPULATION) for i, value in enumerate(population_sizes)]
    width = len(sizes)
    if width == 0:
        _fail("EMPTY_INPUT", "population_sizes", "no strata given")
    if width > MAX_STRATA:
        _fail("COUNT_OUT_OF_RANGE", "population_sizes", f"more than {MAX_STRATA} strata")
    for index, size in enumerate(sizes):
        if size < 1:
            _fail("INCONSISTENT_COUNTS", f"population_sizes[{index}]", "stratum size must be >= 1")
    if sum(sizes) > MAX_TOTAL_POPULATION:
        _fail("COUNT_OUT_OF_RANGE", "population_sizes", f"total exceeds {MAX_TOTAL_POPULATION}")
    if not isinstance(anticipated_sds, (list, tuple)) or len(anticipated_sds) != width:
        _fail("INCONSISTENT_COUNTS", "anticipated_sds", "one standard deviation per stratum")
    deviations = []
    for index, value in enumerate(anticipated_sds):
        number = _real(value, f"anticipated_sds[{index}]")
        if not 0.0 <= number <= MAX_ABS_VALUE:
            _fail("VALUE_OUT_OF_BOUNDS", f"anticipated_sds[{index}]", f"{number!r} is outside [0, {MAX_ABS_VALUE}]")
        deviations.append(number)
    if not isinstance(census, (list, tuple)) or len(census) != width or any(not isinstance(flag, bool) for flag in census):
        _fail("WRONG_TYPE", "census", "one boolean census flag per stratum")
    n_total = _count(budget, "budget", maximum=MAX_TOTAL_POPULATION)
    minimum = _count(min_sample, "min_sample", maximum=MAX_TOTAL_POPULATION)
    if minimum < 1:
        _fail("INFEASIBLE", "min_sample", "every sampled stratum needs at least one unit")
    rate = _real(min_inclusion_rate, "min_inclusion_rate")
    if not 0.0 <= rate <= 1.0:
        _fail("PROBABILITY_OUT_OF_RANGE", "min_inclusion_rate", f"{rate!r} is not in [0, 1]")
    fraction = Fraction(rate)
    lower = []
    for index, size in enumerate(sizes):
        if census[index]:
            lower.append(size)
        else:
            lower.append(min(size, max(minimum, math.ceil(fraction * size))))
    if n_total < sum(lower):
        _fail("INFEASIBLE", "budget", f"budget {n_total} < required minimum {sum(lower)}")
    if n_total > sum(sizes):
        _fail("INFEASIBLE", "budget", f"budget {n_total} > population {sum(sizes)}")
    chosen = list(lower)
    remaining = n_total - sum(lower)
    heap: list[tuple[float, int]] = []
    for index, size in enumerate(sizes):
        if not census[index] and chosen[index] < size:
            heap.append((-(size * float(deviations[index])) / math.sqrt(chosen[index] * (chosen[index] + 1.0)), index))
    heapq.heapify(heap)
    while remaining > 0:
        _priority, index = heapq.heappop(heap)
        chosen[index] += 1
        remaining -= 1
        if chosen[index] < sizes[index]:
            heapq.heappush(
                heap,
                (-(sizes[index] * float(deviations[index])) / math.sqrt(chosen[index] * (chosen[index] + 1.0)), index),
            )
    sampled = [index for index, flag in enumerate(census) if not flag]
    sample_budget = n_total - sum(sizes[index] for index, flag in enumerate(census) if flag)
    weight = math.fsum(sizes[index] * float(deviations[index]) for index in sampled)
    continuous = []
    for index, size in enumerate(sizes):
        if census[index]:
            continuous.append(float(size))
        elif weight > 0:
            continuous.append(sample_budget * size * float(deviations[index]) / weight)
        else:
            continuous.append(sample_budget / len(sampled))
    variance = math.fsum(
        sizes[index] * sizes[index] * (1.0 - chosen[index] / sizes[index]) * float(deviations[index]) ** 2 / chosen[index]
        for index in sampled
    )
    return Allocation(tuple(chosen), tuple(census), tuple(lower), tuple(continuous), variance)


METHOD = "inferdrift.plan_stats.stratified_total_bound"


def method_name() -> str:
    return METHOD


def ensure_ready() -> str:
    """The plan bound can run. Called before a ``--plan`` compare sends a request."""
    return method_name()


def _json_ready(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, list):
        return [_json_ready(value) for value in obj]
    return obj


def allocate(sizes: Sequence[int], deviations: Sequence[float], *, budget: int, min_sample: int, min_inclusion_rate: float) -> tuple[int, ...]:
    return neyman_allocation(
        list(sizes),
        list(deviations),
        budget=budget,
        census=tuple(False for _ in sizes),
        min_sample=min_sample,
        min_inclusion_rate=min_inclusion_rate,
    ).sample_sizes


def population_bound(bound_rows: Sequence[Mapping[str, int]], estimate_rows: Sequence[Mapping[str, object]], *, alpha: float, alpha_split: str) -> tuple[dict, dict, str]:
    """Bound and estimate dicts, plus the name of the implementation that ran."""
    bound = stratified_total_bound(bound_rows, alpha=alpha, side="upper", alpha_split=alpha_split)
    estimate = ht_estimate(estimate_rows)
    return _json_ready(bound.to_dict()), _json_ready(estimate.to_dict()), METHOD
