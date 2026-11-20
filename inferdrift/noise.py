# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The noise floor: how often a stack disagrees with itself.

A stack's answer-level repeat-flip rate f is the share of items whose grade
differs between two runs of the same request on that stack.  Items where either
run returned an error are left out (an error is not an answer).  The comparison
floor is the larger of the two stacks' rates; it is measured on every item for a
stack that ran twice on every item, else on the probe items.
"""

from __future__ import annotations

from collections import defaultdict

from .stats import cp_lower, cp_upper


def valid(side: dict | None) -> bool:
    return bool(side) and side.get("finish") != "error" and not side.get("error")


def rate_dict(k: int, n: int, alpha: float) -> dict:
    if n == 0:
        return {"point": None, "lower": 0.0, "upper": 1.0, "alpha": alpha, "count": 0, "n": 0}
    return {"point": k / n, "lower": cp_lower(k, n, alpha), "upper": cp_upper(k, n, alpha), "alpha": alpha,
            "count": k, "n": n}


def pair_stats(records: list[dict], first: str, second: str, g1: str, g2: str, alpha: float) -> dict:
    """Repeat statistics of one stack over the records that carry its second run."""
    with_second = [r for r in records if second in r]
    pairs = [r for r in with_second if valid(r.get(first)) and valid(r.get(second))]
    n = len(pairs)
    same = sum(1 for r in pairs if r[first]["text"] == r[second]["text"])
    flips = [r for r in pairs if bool(r[g1]["correct"]) != bool(r[g2]["correct"])]
    harmful = sum(1 for r in flips if r[g1]["correct"])
    per = defaultdict(lambda: [0, 0])
    for r in pairs:
        per[r["capability"]][0] += 1
        per[r["capability"]][1] += int(bool(r[g1]["correct"]) != bool(r[g2]["correct"]))
    return {"items": n, "errored_pairs_left_out": len(with_second) - n,
            "byte_identical": same, "byte_identity_rate": (same / n) if n else None,
            "answer_flips": len(flips), "repeat_flip_rate": rate_dict(len(flips), n, alpha),
            "harmful_repeat_flips": harmful, "helpful_repeat_flips": len(flips) - harmful,
            "harmful_repeat_rate": rate_dict(harmful, n, alpha),
            "helpful_repeat_rate": rate_dict(len(flips) - harmful, n, alpha),
            "per_capability": {c: {"items": v[0], "answer_flips": v[1], "rate": v[1] / v[0] if v[0] else None}
                               for c, v in sorted(per.items())}}


def comparison_floor(ref: dict, cand: dict, ref_all: bool, cand_all: bool, capabilities: list[str],
                     alpha: float) -> dict:
    """The floor that the power gate and the report use.

    Per stack: its rate per capability when it ran twice on every item, else its
    pooled rate (probe items).  Per capability: the larger of the two stacks."""
    out_caps = {}
    measured = (ref["items"] > 0) or (cand["items"] > 0)

    def stack_rate(st: dict, all_items: bool, c: str) -> float:
        if st["items"] == 0:
            return 0.0
        if all_items:
            pc = st["per_capability"].get(c)
            if pc and pc["items"]:
                return pc["rate"]
        return st["repeat_flip_rate"]["point"]

    for c in capabilities:
        out_caps[c] = max(stack_rate(ref, ref_all, c), stack_rate(cand, cand_all, c))
    pooled_r = ref["repeat_flip_rate"]["point"] if ref["items"] else 0.0
    pooled_c = cand["repeat_flip_rate"]["point"] if cand["items"] else 0.0
    which = "reference" if pooled_r >= pooled_c else "candidate"
    st = ref if which == "reference" else cand
    return {"measured": measured, "value": max(pooled_r, pooled_c), "stack": which,
            "bounds": st["repeat_flip_rate"], "per_capability": out_caps,
            "basis": {"reference": ("all items" if ref_all else "probe items") if ref["items"] else "not run twice",
                      "candidate": ("all items" if cand_all else "probe items") if cand["items"] else "not run twice"},
            "items": {"reference": ref["items"], "candidate": cand["items"]}}
