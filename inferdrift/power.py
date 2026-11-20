# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact power of the paired criteria, and the number of items they need.

Flip models (what "a floor f and an effect l" mean):

* ``net_change``: per item the residual is +1 (a loss: reference right,
  candidate wrong) with probability p+ = f/2 + l and -1 (a gain) with
  probability p- = f/2.  Under no real change (l = 0) the flips are symmetric
  noise at the answer-level repeat-flip rate f.
* ``harmful_excess``: the residual is h - h', a cross-stack harmful flip minus a
  replicate harmful flip.  Replicate pairs are exchangeable, so
  P(y1 = 1, y2 = 0) = P(y1 = 0, y2 = 1) = f/2 =: h.  With the cross pair's
  harmful rate h + l and the two indicators independent within an item,
  p+ = (h + l)(1 - h), p- = h(1 - h - l), so the mean residual is exactly l.
  Positive correlation between h and h' (the same volatile items flip in both
  pairs) lowers p+ + p- and raises power.

Exact pass probability of one capability with n items, for the bound the
verdict uses (``exactstats.paired_mean_bound``, upper side, at alpha):

    Pow(n) = sum over D of Bin(D; n, pi) * P[Bin(D, p-/pi) >= g*(D)],   pi = p+ + p-,

where D is the number of discordant items and g*(D) the smallest number of
gains (-1 residuals) among them for which the upper bound is at most tau.  The
bound is UB = (1 - 2 r_L) * pi_U (capped at 1), or at most 0 when r_L >= 1/2,
with pi_U the upper end of the two-sided Clopper-Pearson interval for pi at
level gamma and r_L the one-sided lower bound for the gain share at level
alpha - gamma.  r_L is nondecreasing in the gain count, so the pass set is
{g >= g*(D)}; g*(D) is found by binary search on the Clopper-Pearson defining
inequality at r = (1 - tau / pi_U) / 2 (the bisection's own test).  The sum runs
over the D values holding all but 1e-12 of the Bin(n, pi) mass, so the reported
power is a lower bound that is off by at most 1e-12.

Several capabilities are disjoint item sets with independent items, so the
total power is the product of the per-capability powers.

The power of exact tests is not monotone in n (it is a sawtooth): a size can
reach the power while a few sizes just above it fall slightly below it.
``threshold`` therefore returns a size N that reaches the power, whose one-smaller
size does not, and above which every size up to ``check_window(N)`` more (5% of
N, at least 50 and at most 200) was checked and also reaches it.  When a size in
that window falls short, the search continues above it.  Sizes beyond the
checked window are not checked, and a smaller powered size may exist below a
gap.  The search is deterministic and starts from 0, so ``inferdrift power`` and
``compare`` report the same number for the same composition and floor.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import asdict, dataclass

from . import domains
from .errors import BadInputError
from .exactstats import _numeric as nm
from .exactstats.paired import DEFAULT_NUISANCE_FRACTION, levels

EPS = 1e-12
MAX_ITEMS_PER_CAPABILITY = 100_000
COMPARE_SEARCH_CAP = 20_000  # per capability, for the search inside every compare (the power command uses 100k)
WINDOW_MIN, WINDOW_FRACTION, WINDOW_MAX = 50, 0.05, 200  # sizes checked above a found size (see check_window)
_CP_CACHE: dict = {}
_POW_CACHE: dict = {}


@dataclass(frozen=True)
class FlipModel:
    """Per-item residual probabilities: p_loss = P(+1), p_gain = P(-1)."""

    p_loss: float
    p_gain: float
    kind: str = "custom"
    floor: float | None = None
    effect: float | None = None

    def __post_init__(self) -> None:
        for name in ("p_loss", "p_gain"):
            domains.number(getattr(self, name), name, ge=0.0, le=1.0)
        if self.p_loss + self.p_gain > 1.0 + 1e-15:
            raise BadInputError("flip model", "p_loss + p_gain must be at most 1", (self.p_loss, self.p_gain))

    def to_dict(self) -> dict:
        return asdict(self)


def net_change_model(floor: float, loss: float = 0.0) -> FlipModel:
    f = domains.number(floor, "floor", ge=0.0, le=1.0)
    lo = domains.number(loss, "loss", ge=-1.0, le=1.0)
    p_loss, p_gain = f / 2.0 + lo, f / 2.0
    if p_loss < 0.0 or p_loss + p_gain > 1.0:
        raise BadInputError("loss", f"with floor {f} the loss must lie in [{-f / 2}, {1 - f}]", loss)
    return FlipModel(p_loss, p_gain, "net_change", f, lo)


def harmful_excess_model(floor: float, excess: float = 0.0) -> FlipModel:
    f = domains.number(floor, "floor", ge=0.0, le=1.0)
    ex = domains.number(excess, "excess", ge=-1.0, le=1.0)
    h = f / 2.0
    if h + ex < 0.0 or h + ex > 1.0:
        raise BadInputError("excess", f"with floor {f} the excess must lie in [{-h}, {1 - h}]", excess)
    return FlipModel((h + ex) * (1.0 - h), h * (1.0 - h - ex), "harmful_excess", f, ex)


def model_for(criterion: str, floor: float, effect: float = 0.0) -> FlipModel:
    if criterion == "net_change":
        return net_change_model(floor, effect)
    if criterion == "harmful_excess":
        return harmful_excess_model(floor, effect)
    raise BadInputError("criterion", "the power model exists for net_change and harmful_excess", criterion)


# --------------------------------------------------------------------------- the pass rule, inverted
def _levels(alpha: float, phi: float) -> tuple[float, float, float, float]:
    return levels(alpha, math.log(alpha), phi)


def _pi_upper(D: int, n: int, lv) -> float:
    key = (D, n, lv[0], lv[1])
    v = _CP_CACHE.get(key)
    if v is None:
        if len(_CP_CACHE) > 400_000:
            _CP_CACHE.clear()
        v = nm.cp_upper(D, n, lv[0], lv[1])
        _CP_CACHE[key] = v
    return v


def critical_gains(D: int, n: int, alpha: float, tau: float, phi: float = DEFAULT_NUISANCE_FRACTION,
                   lv=None) -> int:
    """Smallest number of gains g among D discordant items (of n) for which the
    upper bound on the net loss is at most tau; D + 1 when no g passes."""
    lv = lv or _levels(alpha, phi)
    pi_hi = _pi_upper(D, n, lv)
    if D == 0:
        return 0 if pi_hi <= tau else 1
    if pi_hi <= tau:
        return 0
    s_star = (1.0 - tau / pi_hi) / 2.0
    rest, log_r = lv[2], lv[3]

    def passes(g: int) -> bool:
        return nm.cp_lower_predicate(g, D, rest, log_r, s_star)

    if not passes(D):
        return D + 1
    lo, hi = 0, D  # passes(hi); passes(0) is False because r_L(0) = 0 < s_star
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _binom_weights(n: int, p: float):
    """(D, P(Bin(n, p) = D)) for the D values holding all but EPS of the mass."""
    if p <= 0.0:
        yield 0, 1.0
        return
    if p >= 1.0:
        yield n, 1.0
        return
    m = min(n, int((n + 1) * p))
    q = 1.0 - p
    yield m, math.exp(nm.log_binom_pmf(m, n, p))
    # downward
    i = m
    while i > 0:
        i -= 1
        t = math.exp(nm.log_binom_pmf(i, n, p))
        yield i, t
        r = (i * q) / ((n - i + 1) * p) if i > 0 else 0.0
        if i == 0 or (r < 1.0 and t * r <= (EPS / 2.0) * (1.0 - r)):
            break
    i = m
    while i < n:
        i += 1
        t = math.exp(nm.log_binom_pmf(i, n, p))
        yield i, t
        r = ((n - i) * p) / ((i + 1) * q) if i < n else 0.0
        if i == n or (r < 1.0 and t * r <= (EPS / 2.0) * (1.0 - r)):
            break


def pass_probability(n: int, model: FlipModel, alpha: float, tau: float,
                     phi: float = DEFAULT_NUISANCE_FRACTION) -> float:
    """Exact probability that one capability with n items passes (upper bound <= tau)."""
    n = domains.integer(n, "n", ge=0, le=10 ** 7)
    key = (n, model.p_loss, model.p_gain, alpha, tau, phi)
    v = _POW_CACHE.get(key)
    if v is not None:
        return v
    if len(_POW_CACHE) > 100_000:
        _POW_CACHE.clear()
    lv = _levels(alpha, phi)
    pi = model.p_loss + model.p_gain
    p_good = (model.p_gain / pi) if pi > 0 else 0.0
    total = 0.0
    if n == 0:
        total = 1.0 if 1.0 <= tau else 0.0
    else:
        for D, w in _binom_weights(n, pi):
            g_star = critical_gains(D, n, alpha, tau, phi, lv)
            if g_star > D:
                continue
            if g_star <= 0:
                total += w
            else:
                total += w * math.exp(nm.log_binom_sf(g_star, D, p_good))
    total = min(1.0, max(0.0, total))
    _POW_CACHE[key] = total
    return total


def power_total(counts: dict, models: dict, alpha: float, tau: float, phi: float = DEFAULT_NUISANCE_FRACTION) -> float:
    """Product over capabilities of the exact pass probabilities."""
    p = 1.0
    for c, n in counts.items():
        p *= pass_probability(n, models[c], alpha, tau, phi)
        if p == 0.0:
            break
    return p


def powered(counts: dict, models: dict, alpha: float, tau: float, target: float,
            phi: float = DEFAULT_NUISANCE_FRACTION) -> bool:
    """power_total(...) >= target, evaluating the smallest capabilities first and stopping as soon as the
    running product falls below the target (every remaining factor is at most 1)."""
    p = 1.0
    for c in sorted(counts, key=lambda c: (counts[c], str(c))):
        p *= pass_probability(counts[c], models[c], alpha, tau, phi)
        if p < target:
            return False
    return True


def allocate(total: int, weights: dict) -> dict:
    """Largest-remainder rounding of total * w_c (weights need not sum to 1)."""
    s = sum(weights.values())
    raw = {c: total * w / s for c, w in weights.items()}
    base = {c: int(math.floor(v)) for c, v in raw.items()}
    left = total - sum(base.values())
    for c in sorted(raw, key=lambda c: (-(raw[c] - base[c]), str(c)))[:left]:
        base[c] += 1
    return base


def check_window(n: int) -> int:
    """How many sizes above a found size N are checked: 5% of N, at least 50 and at most 200."""
    return min(WINDOW_MAX, max(WINDOW_MIN, math.ceil(WINDOW_FRACTION * n)))


def _transition(is_powered, start: int, upper: int) -> int | None:
    """A size N in (start, upper] that is powered while N - 1 is not (or N - 1 == start): doubling
    from max(start + 1, 16), then bisection.  None when no size on the doubling path, up to and
    including ``upper``, is powered."""
    if upper <= start:
        return None
    lo = start
    hi = min(upper, max(start + 1, 16))
    while not is_powered(hi):
        if hi >= upper:
            return None
        lo = hi
        hi = min(upper, hi * 2)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if is_powered(mid):
            hi = mid
        else:
            lo = mid
    return hi


def threshold(is_powered, *, upper: int, above: int = 0) -> dict:
    """The size reported as "needed": N > ``above`` that is powered, whose one-smaller size is not (or
    is ``above``), and such that every size from N to min(upper, N + check_window(N)) is powered.
    When a size in that window is not powered, the search continues above it.

    Returns {"n": N or None, "checked_up_to": the last size checked to be powered, or None}.
    Nothing is claimed about sizes above ``checked_up_to``."""
    start = above
    while True:
        n = _transition(is_powered, start, upper)
        if n is None:
            return {"n": None, "checked_up_to": None}
        top = min(upper, n + check_window(n))
        short = next((k for k in range(n + 1, top + 1) if not is_powered(k)), None)
        if short is None:
            return {"n": n, "checked_up_to": top}
        start = short


def items_needed(weights: dict, models: dict, alpha: float, tau: float, power: float, *, above: int = 0,
                 phi: float = DEFAULT_NUISANCE_FRACTION, cap_per_capability: int = MAX_ITEMS_PER_CAPABILITY) -> dict:
    """The total item count needed with capabilities in the proportions ``weights`` (``threshold`` over
    totals, each total allocated by ``allocate``).

    The search starts from 0, so the number does not depend on ``above``; only when that number is
    at most ``above`` (a size above it dips below the power) does the search continue above ``above``.
    It stops when the largest capability would exceed ``cap_per_capability``.

    Returns {"items_needed": N | None, "per_capability": allocation at N, "checked_up_to": the last total
    checked, "exceeds": None}, or, when the search stops at the cap, items_needed None, "exceeds": the cap,
    and "searched_to", "per_capability_at_stop", "power_at_stop", "largest_capability" and a "note"
    that states only what was checked."""
    def is_powered(N: int) -> bool:
        return powered(allocate(N, weights), models, alpha, tau, power, phi)

    cap_total = int(cap_per_capability / max(weights.values()) * sum(weights.values()))
    r = threshold(is_powered, upper=cap_total)
    if r["n"] is not None and r["n"] <= above:
        r = threshold(is_powered, upper=cap_total, above=above)
    if r["n"] is None:
        at_stop = allocate(cap_total, weights)
        largest = min(at_stop, key=lambda c: (-at_stop[c], str(c)))
        return {"items_needed": None, "per_capability": None, "exceeds": cap_per_capability, "checked_up_to": None,
                "searched_to": cap_total, "per_capability_at_stop": at_stop, "largest_capability": largest,
                "power_at_stop": power_total(at_stop, models, alpha, tau, phi),
                "note": f"not powered at {cap_total} items in these proportions ({at_stop[largest]} in the largest "
                        f"capability, {largest}); the search stopped there"}
    return {"items_needed": r["n"], "per_capability": allocate(r["n"], weights), "exceeds": None,
            "checked_up_to": r["checked_up_to"]}


def items_needed_in_order(order_capabilities: list, models: dict, alpha: float, tau: float, power: float, *,
                          above: int, phi: float = DEFAULT_NUISANCE_FRACTION) -> dict:
    """The number of leading items of a fixed item order (a run's own seeded order, cut by
    ``--max-items``) that is needed: ``threshold`` over prefix lengths in (``above``, len(order)], each
    prefix tested with its own counts per capability.  The search starts from 0, as in ``items_needed``.

    Returns {"items_needed": N | None, "per_capability": counts of the first N items, "checked_up_to"};
    items_needed is None when even the whole order is not powered (on the doubling path)."""
    positions: dict = {}
    for i, c in enumerate(order_capabilities):
        positions.setdefault(c, []).append(i)

    def counts(n: int) -> dict:
        out = {}
        for c in sorted(positions, key=str):
            k = bisect_left(positions[c], n)
            if k:
                out[c] = k
        return out

    def is_powered(n: int) -> bool:
        return powered(counts(n), models, alpha, tau, power, phi)

    total = len(order_capabilities)
    r = threshold(is_powered, upper=total)
    if r["n"] is not None and r["n"] <= above:
        r = threshold(is_powered, upper=total, above=above)
    if r["n"] is None:
        return {"items_needed": None, "per_capability": None, "checked_up_to": None}
    return {"items_needed": r["n"], "per_capability": counts(r["n"]), "checked_up_to": r["checked_up_to"]}


def smallest_tau(counts: dict, models_at: dict, alpha: float, power: float,
                 phi: float = DEFAULT_NUISANCE_FRACTION, steps: int = 30) -> float | None:
    """Smallest tolerance (to about 1e-9, rounded up) that ``counts`` items certify with
    ``power``; None if even tau just below 1 is not powered.  Power is nondecreasing in tau."""
    hi = 1.0 - 1e-9
    if power_total(counts, models_at, alpha, hi, phi) < power:
        return None
    lo = 0.0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if power_total(counts, models_at, alpha, mid, phi) >= power:
            hi = mid
        else:
            lo = mid
    return math.ceil(hi * 1e6) / 1e6


def normal_items_estimate(model: FlipModel, alpha: float, tau: float, power: float) -> float:
    """A rough normal-approximation size (for orientation only, never used for a verdict)."""
    from statistics import NormalDist

    nd = NormalDist()
    theta = model.p_loss - model.p_gain
    sd = math.sqrt(max(1e-12, model.p_loss + model.p_gain - theta ** 2))
    gap = tau - theta
    if gap <= 0:
        return math.inf
    return ((nd.inv_cdf(1 - alpha) + nd.inv_cdf(power)) * sd / gap) ** 2
