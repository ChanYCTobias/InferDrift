# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential plan and verdict rules.

Items are evaluated in a seeded random order.  The plan fixes, before any item
is run, a maximum sample size ``n_max`` and a set of looks
L = {32, 64, 128, ...} below n_max, plus n_max itself; K = |L|.

Verdict at (tau, alpha), with h the harmful-flip count after n items:

* DEVIATES at look k if the one-sided lower bound LB(h_k, n_k; alpha / K) > tau.
  The run stops at the first such look (unless ``early_stop`` is off).
* EQUIVALENT_WITHIN_TAU only at the final planned look n_max, if the one-sided
  upper bound UB(h, n_max; alpha) <= tau.
* INCONCLUSIVE otherwise, including any run that a cap stops before n_max
  without a DEVIATES look.

Validity (exact, no asymptotics).  Let p be the true harmful-flip rate.
If p <= tau, a false DEVIATES needs LB_k > p at some look; by the union bound
this has probability at most K * alpha / K = alpha.  If p > tau, a false
EQUIVALENT needs UB(h, n_max) < p, which has probability at most alpha whatever
happened at earlier looks (stopping early only removes chances to declare
EQUIVALENT).  For a given p only one of the two errors is possible, so each
direction may use the full alpha.

In independent-sampling mode the same rules apply to the excess harmful-flip
rate (cross-stack rate minus the reference's replicate rate), whose bounds come
from a union of two Clopper-Pearson bounds and are therefore wider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import domains
from .errors import BadInputError
from .stats import Bound, StatsBackend

EQUIVALENT = "EQUIVALENT_WITHIN_TAU"
DEVIATES = "DEVIATES"
INCONCLUSIVE = "INCONCLUSIVE"
UNDERPOWERED = "INCONCLUSIVE_UNDERPOWERED"
VERDICTS = (EQUIVALENT, DEVIATES, INCONCLUSIVE)
VERDICTS_ALL = VERDICTS + (UNDERPOWERED,)  # the last only from the paired criteria (criteria.py)


def look_schedule(n_max: int, first: int = 32) -> list[int]:
    n_max = domains.integer(n_max, "n_max", ge=1)
    first = domains.integer(first, "first look", ge=1)
    looks = []
    n = first
    while n < n_max:
        looks.append(n)
        n *= 2
    looks.append(n_max)
    return looks


@dataclass
class LookResult:
    look: int
    n: int
    harmful: int
    helpful: int
    alpha_look: float
    harmful_lower_at_look_alpha: float
    deviates: bool
    elapsed_s: float
    excess: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    n_max: int
    tau: float
    alpha: float
    looks: list[int] = field(default_factory=list)
    early_stop: bool = True

    def __post_init__(self) -> None:
        self.n_max = domains.integer(self.n_max, "n_max", ge=1)
        self.tau = domains.number(self.tau, "tau", gt=0.0, lt=1.0)
        self.alpha = domains.number(self.alpha, "alpha", gt=0.0, lt=1.0)
        self.early_stop = domains.boolean(self.early_stop, "early_stop")
        if self.looks is None or (isinstance(self.looks, (list, tuple)) and len(self.looks) == 0):
            self.looks = look_schedule(self.n_max)
        if not isinstance(self.looks, (list, tuple)):
            raise BadInputError("looks", "must be a list of integers", self.looks)
        self.looks = [domains.integer(x, "looks", ge=1) for x in self.looks]
        if any(b <= a for a, b in zip(self.looks, self.looks[1:])):
            raise BadInputError("looks", "must be strictly increasing", self.looks)
        if self.looks[-1] != self.n_max:
            raise BadInputError("looks", f"must end at n_max = {self.n_max}", self.looks)

    @property
    def alpha_per_look(self) -> float:
        return self.alpha / len(self.looks)

    def to_dict(self, criterion: str = "harmful_abs", capabilities: int = 1) -> dict:
        if criterion == "harmful_abs":
            return self._to_dict_010()
        d = self._to_dict_010()
        m = max(1, int(capabilities))
        d["capabilities"] = m
        d["alpha_per_look_deviates"] = self.alpha / (m * len(self.looks))
        d["rule"] = ("Per capability c (the item's suite), with r = +1 a loss: DEVIATES at look k iff "
                     "LB(theta_c; alpha/(m K)) > tau for some c; EQUIVALENT_WITHIN_TAU iff every planned item was "
                     "evaluated and UB(theta_c; alpha) <= tau for every c; INCONCLUSIVE_UNDERPOWERED when the plan "
                     "lacks the requested power at the measured floor; otherwise INCONCLUSIVE. Bounds are exact "
                     "paired bounds; theta is the net loss (net_change) or the excess harmful-flip rate over the "
                     "replicate floor (harmful_excess).")
        return d

    def _to_dict_010(self) -> dict:
        return {
            "n_max": self.n_max,
            "tau": self.tau,
            "alpha": self.alpha,
            "looks": list(self.looks),
            "alpha_per_look_deviates": self.alpha_per_look,
            "alpha_final_equivalence": self.alpha,
            "early_stop": self.early_stop,
            "rule": (
                "DEVIATES at look k iff LB(h_k, n_k; alpha/K) > tau; EQUIVALENT_WITHIN_TAU iff the run reaches "
                "n_max and UB(h, n_max; alpha) <= tau; otherwise INCONCLUSIVE. Bounds are one-sided exact "
                "Clopper-Pearson bounds on the harmful-flip rate (independent-sampling mode: on the excess rate)."
            ),
        }


def look_decision(stats: StatsBackend, plan: Plan, look_index: int, n: int, harmful: int,
                  *, excess_counts: tuple[int, int, int, int] | None = None) -> tuple[bool, float, dict | None]:
    """Return (deviates, lower_bound_used, excess_detail) at a look."""
    a = plan.alpha_per_look
    if excess_counts is None:
        lb = stats.rate_lower(harmful, n, a)
        return lb > plan.tau, lb, None
    h_rc, n_rc, h_rr, n_rr = excess_counts
    b = stats.excess_rate(h_rc, n_rc, h_rr, n_rr, a)
    return b.lower > plan.tau, b.lower, b.to_dict()


def final_decision(stats: StatsBackend, plan: Plan, n: int, harmful: int, *, reached_n_max: bool,
                   any_deviates: bool, excess_counts: tuple[int, int, int, int] | None = None) -> tuple[str, Bound]:
    """Verdict and the (alpha-level) bound reported with it."""
    if excess_counts is None:
        bound = stats.rate_bound(harmful, n, plan.alpha) if n else Bound(None, 0.0, 1.0, plan.alpha, "empty")
    else:
        h_rc, n_rc, h_rr, n_rr = excess_counts
        bound = stats.excess_rate(h_rc, n_rc, h_rr, n_rr, plan.alpha)
    if any_deviates:
        return DEVIATES, bound
    if reached_n_max and n == plan.n_max and bound.upper <= plan.tau:
        return EQUIVALENT, bound
    return INCONCLUSIVE, bound
