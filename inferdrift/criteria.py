# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""What a verdict certifies: the paired criteria and their look and verdict rules.

Every residual r_i is in {-1, 0, +1} and oriented so that +1 is the loss
direction; theta = E[r] = P(+1) - P(-1).

* ``net_change`` (the default): r_i = y_ref - y_cand, so theta is the net
  accuracy loss acc(reference) - acc(candidate).  Harmful and helpful flips
  offset each other inside a capability.
* ``harmful_excess`` (stricter): r_i = h_i - h'_i, with h_i = 1 if the reference's
  first run is right and the candidate is wrong (a cross-stack harmful flip) and
  h'_i = 1 if the reference's first run is right and its second run is wrong (a
  replicate harmful flip).  theta is the excess harmful-flip rate over the
  reference's own noise floor; helpful flips do not offset harmful ones.
* ``harmful_abs``: the 0.1.0 rule on the harmful-flip count (``plan.py``).

For a capability c with counts (bad, good, n) at look k of K, with m
capabilities among the planned items:

* DEVIATES at look k when the lower bound LB(theta_c; alpha / (m K)) > tau for
  some c (Bonferroni over capabilities and looks: a false DEVIATES has
  probability at most alpha, whatever the dependence between looks).
* EQUIVALENT_WITHIN_TAU when every planned item was evaluated and the upper
  bound UB(theta_c; alpha) <= tau for every c.  This is an intersection-union
  test: claiming that every capability is within tau needs no multiplicity
  adjustment, and a false claim has probability at most alpha.
* INCONCLUSIVE otherwise; INCONCLUSIVE_UNDERPOWERED when the power gate says the
  planned items are too few (never a pass or a fail).

Bounds are ``exactstats.paired_mean_bound`` (exact, conservative).  Every
condition other than the two bounds (the power gate, reaching every planned
item, the routing) can only turn a verdict into a non-verdict, so it cannot
raise either error rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .exactstats.paired import DEFAULT_NUISANCE_FRACTION, delta_lower, levels

NET_CHANGE = "net_change"
HARMFUL_EXCESS = "harmful_excess"
HARMFUL_ABS = "harmful_abs"
CRITERIA = (NET_CHANGE, HARMFUL_EXCESS, HARMFUL_ABS)
PAIRED_CRITERIA = (NET_CHANGE, HARMFUL_EXCESS)
STATISTIC = {NET_CHANGE: "net_loss", HARMFUL_EXCESS: "paired_excess_harmful_flip_rate"}
PHI = DEFAULT_NUISANCE_FRACTION
METHOD = "exact paired bound (two-sided Clopper-Pearson on the discordance at gamma = 0.1 alpha, one-sided " \
         "conditional Clopper-Pearson on the direction at alpha - gamma, union bound)"

EQUIVALENT = "EQUIVALENT_WITHIN_TAU"
DEVIATES = "DEVIATES"
INCONCLUSIVE = "INCONCLUSIVE"
UNDERPOWERED = "INCONCLUSIVE_UNDERPOWERED"


def residual_net(ref_correct: bool, cand_correct: bool) -> int:
    """+1 a loss (reference right, candidate wrong), -1 a gain, 0 no change."""
    return int(bool(ref_correct)) - int(bool(cand_correct))


def residual_excess(ref_correct: bool, cand_correct: bool, rep_correct: bool) -> int:
    """h - h': a cross-stack harmful flip minus a replicate harmful flip."""
    h = int(bool(ref_correct) and not cand_correct)
    h2 = int(bool(ref_correct) and not rep_correct)
    return h - h2


@dataclass
class Counts:
    """Residual counts of one capability: bad = #(+1), good = #(-1), n items."""

    bad: int = 0
    good: int = 0
    n: int = 0

    def add(self, r: int) -> None:
        self.n += 1
        if r > 0:
            self.bad += 1
        elif r < 0:
            self.good += 1

    @property
    def point(self) -> float | None:
        return (self.bad - self.good) / self.n if self.n else None

    def copy(self) -> "Counts":
        return Counts(self.bad, self.good, self.n)

    def to_dict(self) -> dict:
        return asdict(self)


_CACHE: dict = {}


def _side(bad: int, good: int, n: int, alpha: float, side: str) -> float:
    key = (bad, good, n, alpha, side)
    v = _CACHE.get(key)
    if v is None:
        if len(_CACHE) > 500_000:
            _CACHE.clear()
        lv = levels(alpha, math.log(alpha), PHI)
        v = delta_lower(bad, good, n, lv) if side == "lower" else -delta_lower(good, bad, n, lv)
        _CACHE[key] = v
    return v


def upper(c: Counts, alpha: float) -> float:
    """One-sided (1 - alpha) upper bound on theta."""
    return _side(c.bad, c.good, c.n, alpha, "upper")


def lower(c: Counts, alpha: float) -> float:
    """One-sided (1 - alpha) lower bound on theta."""
    return _side(c.bad, c.good, c.n, alpha, "lower")


def shortcuts_valid(alpha: float) -> bool:
    """LB <= point estimate <= UB holds when both derived levels are at most 1/2
    (a Clopper-Pearson bound at a level above 1/2 can cross the estimate)."""
    return (1.0 - PHI) * alpha <= 0.5 and PHI * alpha / 2.0 <= 0.5


def bound_dict(c: Counts, alpha: float) -> dict:
    return {"point": c.point, "lower": lower(c, alpha) if c.n else -1.0, "upper": upper(c, alpha) if c.n else 1.0,
            "alpha": alpha, "method": METHOD, "n": c.n, "bad": c.bad, "good": c.good}


def look_check(per_cap: dict, alpha_look: float, tau: float, *, quick: bool = False) -> tuple[bool, dict]:
    """DEVIATES test at one look.  Returns (deviates, {capability: lower bound or None}).

    With ``quick`` (simulations only) a capability whose point estimate is at most
    tau is not evaluated: its lower bound cannot exceed tau."""
    out = {}
    dev = False
    skip = quick and shortcuts_valid(alpha_look)
    for cap, c in per_cap.items():
        if c.n == 0:
            out[cap] = None
            continue
        if skip and c.point <= tau:
            out[cap] = None
            continue
        lb = lower(c, alpha_look)
        out[cap] = lb
        dev = dev or lb > tau
    return dev, out


def final_check(per_cap: dict, alpha: float, tau: float, *, quick: bool = False) -> tuple[bool, dict]:
    """Every capability's upper bound at alpha within tau?  Returns (all_within, {cap: UB or None})."""
    out = {}
    ok = True
    skip = quick and shortcuts_valid(alpha)
    for cap, c in per_cap.items():
        if c.n == 0:
            out[cap] = 1.0
            ok = False
            continue
        if skip and c.point > tau:
            out[cap] = None
            ok = False
            if quick:
                return False, out
            continue
        ub = upper(c, alpha)
        out[cap] = ub
        ok = ok and ub <= tau
    return ok, out


def decide(*, underpowered: bool, deviates: bool, complete: bool, all_within: bool) -> str:
    """Verdict precedence: UNDERPOWERED, then DEVIATES, then EQUIVALENT, else INCONCLUSIVE."""
    if underpowered:
        return UNDERPOWERED
    if deviates:
        return DEVIATES
    if complete and all_within:
        return EQUIVALENT
    return INCONCLUSIVE


def evaluate_stream(increments: list[dict], planned: dict, alpha: float, tau: float, *, early_stop: bool = True,
                    quick: bool = True) -> tuple[str, int | None]:
    """The look and verdict rule over per-look residual count increments.

    ``increments[k]`` maps each capability to (bad, good, n) added at look k; the
    looks end at the planned counts ``planned`` (capability -> n_c).  No power gate
    (a fixed plan's gate is a constant).  Returns (verdict, deviating look or None)."""
    K = len(increments)
    m = len(planned)
    per = {c: Counts() for c in planned}
    alpha_look = alpha / (m * K)
    first_dev = None
    for k, inc in enumerate(increments):
        for c, (b, g, n) in inc.items():
            pc = per[c]
            pc.bad += b
            pc.good += g
            pc.n += n
        dev, _ = look_check(per, alpha_look, tau, quick=quick)
        if dev and first_dev is None:
            first_dev = k
            if early_stop:
                return DEVIATES, k
    complete = all(per[c].n == planned[c] for c in planned)
    if first_dev is not None:
        return DEVIATES, first_dev
    ok, _ = final_check(per, alpha, tau, quick=quick) if complete else (False, {})
    return (EQUIVALENT if ok else INCONCLUSIVE), None
