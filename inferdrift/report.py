# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""JSON report (schema ``inferdrift/report/2``) and one-page summary.

Every key of report/1 is kept.  Under ``--criterion harmful_abs`` the report/1
keys keep their 0.1.0 meaning; the paired criteria change what ``verdict.statistic``
and ``verdict.bound`` describe (the binding capability's paired bound), and add
the sections ``criteria``, ``noise``, ``power``, ``deterministic_mode`` and
``reference_cache``."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from . import __version__, criteria, determinism, noise
from .jsonio import atomic_write_text
from .redact import scrub

SCHEMA = "inferdrift/report/2"
BUCKETS = (("<=64", 64), ("<=256", 256))


def _bucket_stats(records: list[dict]) -> dict:
    out = {}
    for name, cap in BUCKETS:
        sel = [r for r in records if isinstance(r["ref"]["completion_tokens"], int)
               and r["ref"]["completion_tokens"] <= cap]
        out[name] = _calls(sel)
    sel = [r for r in records if not (isinstance(r["ref"]["completion_tokens"], int)
                                      and r["ref"]["completion_tokens"] <= 256)]
    out[">256 (reported only)"] = _calls(sel)
    return out


def _calls(sel: list[dict]) -> dict:
    base = len(sel)
    spent = sum(r["cand_grading_calls"] for r in sel)
    how = Counter(r["cand_grade_how"] for r in sel)
    return {"items": base, "baseline_calls_grade_every_candidate_output": base,
            "inheritance_calls_including_audit": spent,
            "fresh_calls": how.get("fresh", 0) + how.get("judge_call", 0),
            "audit_calls": how.get("inherited_audited", 0) + how.get("audit_mismatch", 0),
            "inherited_without_call": how.get("inherited", 0) + how.get("judge_cache_hit", 0),
            "identical_outputs": sum(1 for r in sel if r["identical"]),
            "reduction_factor": (base / spent) if spent else None,
            "reduction_note": None if spent else ("no grading call needed" if base else "no items")}


def _truncation(records: list[dict]) -> dict:
    div = [r for r in records if not r["identical"]]
    prefix = [r["id"] for r in div if r["cand"]["finish"] == "stop" and r["cand"]["text"] != r["ref"]["text"]
              and r["ref"]["text"].startswith(r["cand"]["text"]) and len(r["cand"]["text"]) < len(r["ref"]["text"])]
    inconsistent = [r["id"] for r in records if r["cand"]["finish"] == "length"
                    and isinstance(r["cand"]["completion_tokens"], int)
                    and r["cand"]["completion_tokens"] < r["max_new_tokens"]]
    ratios = [len(r["cand"]["text"]) / len(r["ref"]["text"]) for r in div if r["ref"]["text"]]
    suspected = len(prefix) >= 3 and len(prefix) >= 0.25 * max(1, len(div))
    return {"divergent_pairs": len(div), "candidate_is_proper_prefix_with_stop": len(prefix),
            "examples": prefix[:10], "length_finish_below_cap": len(inconsistent),
            "median_candidate_to_reference_length_ratio_on_divergent": statistics.median(ratios) if ratios else None,
            "silent_truncation_suspected": bool(suspected or len(inconsistent) >= 3)}


def _paired_counts(records: list[dict], fn) -> criteria.Counts:
    c = criteria.Counts()
    for r in records:
        v = fn(r)
        if v is not None:
            c.add(v)
    return c


def _residual_fns() -> dict:
    """Residuals (+1 = the loss direction) of the paired quantities reported beside the verdict."""
    def y(r, key):
        return bool(r[key]["correct"])

    def ok(r, side):
        return noise.valid(r.get(side))

    def need_rep(r):
        return "rep" in r and ok(r, "rep") and ok(r, "ref")

    return {
        "excess_harmful_flip_rate": lambda r: (int(y(r, "ref_grade") and not y(r, "cand_grade")) -
                                               int(y(r, "ref_grade") and not y(r, "rep_grade"))) if need_rep(r) else None,
        "excess_helpful_flip_rate": lambda r: (int(not y(r, "ref_grade") and y(r, "cand_grade")) -
                                               int(not y(r, "ref_grade") and y(r, "rep_grade"))) if need_rep(r) else None,
        "excess_churn": lambda r: (int(y(r, "ref_grade") != y(r, "cand_grade")) -
                                   int(y(r, "ref_grade") != y(r, "rep_grade"))) if need_rep(r) else None,
        "replicate_net_change": lambda r: (int(y(r, "rep_grade")) - int(y(r, "ref_grade"))) if need_rep(r) else None,
    }


def _bound_of(c: criteria.Counts, alpha: float) -> dict:
    return criteria.bound_dict(c, alpha)


def _paired_sections(cfg, records, looks, extra, channels, coupling, profiles) -> tuple[dict, list[str]]:
    """criteria, noise, power, deterministic_mode, per-suite and estimate additions of the paired criteria."""
    px = extra["paired"]
    alpha, tau = cfg.alpha, cfg.tau
    lim: list[str] = []
    gate = px.get("gate") or {}
    caps = sorted({r["capability"] for r in records} | set((px.get("run_counts") or {}).keys()))
    # net change and harmful excess, pooled and per capability
    crit_out = {"gating": cfg.criterion, "nuisance_fraction": criteria.PHI, "method": criteria.METHOD}
    for name, key in (("net_change", "residual_net"), ("harmful_excess", "residual_excess")):
        evaluated = [r for r in records if r.get(key) is not None]
        if name == "harmful_excess" and not evaluated:
            crit_out[name] = {"available": False, "reason": "the reference's second run did not cover the items"}
            continue
        basis = "all evaluated items"
        if name == "harmful_excess" and len(evaluated) < len(records):
            basis = f"{len(evaluated)} of {len(records)} items (those with a valid reference second run)"
        pooled = _paired_counts(evaluated, lambda r, k=key: r[k])
        per = {}
        for c in caps:
            cc = _paired_counts([r for r in evaluated if r["capability"] == c], lambda r, k=key: r[k])
            b = _bound_of(cc, alpha)
            b["within_tau"] = bool(cc.n) and b["upper"] <= tau
            per[c] = b
        crit_out[name] = {"available": True, "secondary": name != cfg.criterion, "basis": basis,
                          "theta": "acc(reference) - acc(candidate)" if name == "net_change" else
                          "P(cross-stack harmful flip) - P(replicate harmful flip)",
                          "pooled": _bound_of(pooled, alpha), "per_capability": per,
                          "simultaneous_upper_alpha": alpha / max(1, len(caps)),
                          "per_capability_simultaneous_upper": {c: criteria.upper(
                              _paired_counts([r for r in evaluated if r["capability"] == c], lambda r, k=key: r[k]),
                              alpha / max(1, len(caps))) for c in caps if any(r["capability"] == c for r in evaluated)}}
    # noise floor
    st = gate.get("stats") or {"reference": noise.pair_stats(records, "ref", "rep", "ref_grade", "rep_grade", alpha),
                               "candidate": noise.pair_stats(records, "cand", "cand_rep", "cand_grade",
                                                             "cand_rep_grade", alpha)}
    fl = gate.get("floor")
    batching = extra.get("adapters_batching") or {}
    noise_out = {
        "probe": {"items": px.get("probe_items"), "second_runs_in_reverse_order": True,
                  "batch_composition_varied": batching,
                  "protocol": "per chunk: the reference's first run, then the candidate's first run, in the seeded "
                              "order; then each needed second run with its requests in reverse order. A second run "
                              "is sent after both first runs, so it can differ from the first run in timing and, "
                              "for an engine with a prefix or response cache, in cache state."},
        "replicate_rule": px.get("rule"), "same_coupling_replicates": px.get("same_coupling"),
        "per_stack": st, "routing": px.get("routing"), "noisy_on_probe": px.get("noisy"),
        "noisy_on_channel_probe": px.get("channel_noisy"),
        "floor": fl, "gate_floor": {"source": gate.get("floor_source"), "per_capability": gate.get("floors"),
                                    "supplied": cfg.plan_floor,
                                    "supplied_contradicted": gate.get("supplied_floor_contradicted")},
        "errors_per_pass": px.get("pass_errors"),
        "error_rule": "an errored output is retried once; a reference first run still in error leaves the item "
                      "out of the residuals (and the run cannot pass); a candidate first run in error is graded as a "
                      "wrong answer (a loss); an errored second run is left out of the floor and never counts as a "
                      "repeat flip",
    }
    # estimates beside the verdict
    est_add = {}
    fns = _residual_fns()
    rep_items = [r for r in records if "rep" in r]
    rep_basis = "all items" if (px.get("routing") or {}).get("reference_every_item") else "probe items"
    for name, fn in fns.items():
        cnt = _paired_counts(records, fn)
        if cnt.n == 0:
            est_add[name] = None
            continue
        b = _bound_of(cnt, alpha)
        if name == "replicate_net_change":  # two-sided check at alpha (each side at alpha/2)
            b2 = {"lower": criteria.lower(cnt, alpha / 2), "upper": criteria.upper(cnt, alpha / 2)}
            b["two_sided_alpha"] = alpha
            b["two_sided"] = b2
            # residual +1 here means the second run is right and the first wrong (the second run is better)
            b["sign"] = "acc(reference second run) - acc(reference first run)"
            if b2["lower"] > 0 or b2["upper"] < 0:
                lim.append("the reference's second run differs systematically from its first (drift, or a cache in "
                           "front of the engine); the floor may be wrong")
        else:
            b["sign"] = "+ is the loss direction (cross-stack minus replicate)"
        b["basis"] = rep_basis
        est_add[name] = b
    net = _paired_counts([r for r in records if r.get("residual_net") is not None], lambda r: r["residual_net"])
    est_add["net_change"] = (dict(_bound_of(net, alpha), sign="net loss = acc(reference) - acc(candidate)")
                             if net.n else None)
    est_add["replicate_items"] = len(rep_items)
    # power
    need = gate.get("items_needed") or {}
    power_out = {
        "criterion": cfg.criterion, "tau": tau, "alpha": alpha, "power_requested": cfg.power,
        "model": "net loss residuals with symmetric noise flips at the floor (p+ = p- = f/2), no real change"
                 if cfg.criterion == "net_change" else
                 "harmful_excess residuals with replicate harmful and helpful flips at f/2 each, no real excess",
        "planned_items": None,
        "power_at_planned": gate.get("power_at_planned"), "power_per_capability": gate.get("power_per_capability"),
        "underpowered": gate.get("underpowered"), "floor_source": gate.get("floor_source"),
        "floors": gate.get("floors"), "items_needed": need.get("items_needed"),
        "items_needed_per_capability": need.get("per_capability"), "items_needed_exceeds": need.get("exceeds"),
        "items_needed_lower_bound": need.get("lower_bound"), "short_capabilities": gate.get("short_capabilities"),
        "smallest_tau_at_planned": gate.get("smallest_tau_at_planned"),
        "method": "exact pass probability of the bound the verdict uses, summed over the discordant count "
                  "(truncated at 1e-12 of mass); product over capabilities; items_needed is a powered total whose "
                  "one-smaller total is not, with capabilities in the bank's proportions",
        "assumptions": ["items are independent draws; capabilities are disjoint", "the floor is a point estimate "
                        "from this run (or the supplied --plan-floor); per capability where a stack ran twice on "
                        "every item", "the power of exact tests is not monotone in n: a smaller powered size may "
                        "exist below a gap"],
    }
    # deterministic mode
    det = {}
    declared = extra.get("declared") or {}
    for role, key in (("reference", "reference"), ("candidate", "candidate")):
        s = st[role]
        ch = (channels or {}).get(role) or {}
        det[role] = determinism.section((profiles.get(role) or {}).get("kind"), declared.get(role) or {}, {
            "items": s["items"], "byte_identical": s["byte_identical"], "answer_flips": s["answer_flips"],
            "channel_repeat_identity": ch.get("greedy_repeat_identity_rate_without_errors"),
            "batch_composition_varied": batching.get(role)})
        if det[role]["declared_on_but_nondeterministic"]:
            lim.append(f"the {role} declares a deterministic mode, but its outputs differed between runs")
    # limitations of the paired run
    if gate.get("underpowered"):
        if gate.get("floor_source") == "unmeasured":
            lim.append("the noise floor was not measured (no item was run twice): no pass or fail is issued; "
                       "run the probe (--floor-probe-items) or pass --plan-floor")
        else:
            lim.append(f"the planned items have power {gate.get('power_at_planned'):.3f} < {cfg.power} for tau = "
                       f"{tau} at the measured floor: the verdict is INCONCLUSIVE_UNDERPOWERED, never a pass or a "
                       f"fail" + (f"; about {need['items_needed']} items would do" if need.get("items_needed") else ""))
    if gate.get("floor_source") == "supplied" and not (fl or {}).get("measured"):
        lim.append("the power gate used the supplied --plan-floor; nothing in this run measured the floor (no item "
                   "was run twice), so the run could not check it")
    if gate.get("supplied_floor_contradicted"):
        lim.append("the supplied --plan-floor is below what this run measured; the gate used the measured floor")
    if px.get("excluded"):
        lim.append(f"{px['excluded']} item(s) had no valid output from the reference's run(s) after a retry and were "
                   f"left out of the residuals; EQUIVALENT_WITHIN_TAU needs every planned item")
    pe = px.get("pass_errors") or {}
    if any(v.get("errored") for v in pe.values()):
        lim.append("errored outputs per run (before and after one retry): " + ", ".join(
            f"{k} {v['errored']}/{v['errored_after_retry']}" for k, v in pe.items() if v.get("errored")))
    if px.get("stopped_underpowered"):
        lim.append("stopped after the probe because the plan is underpowered (--stop-if-underpowered)")
    if px.get("pre_gate_deviation") and not px["pre_gate_deviation"].get("allowed_by_power_gate"):
        d = px["pre_gate_deviation"]
        lim.append(f"evidence of a deviation that the power gate did not let become a verdict: at look {d['look']} "
                   f"(n = {d['n']}) the lower bound exceeded tau for {', '.join(d['capabilities'])}")
    if fl and fl.get("measured") and fl.get("value", 0) > 0 and st["candidate"]["items"] and st["reference"]["items"]:
        rc = st["candidate"]["repeat_flip_rate"]
        rr = st["reference"]["repeat_flip_rate"]
        if rc["lower"] > (rr["point"] or 0.0):
            lim.append("the candidate is noisier than the reference; under harmful_excess the candidate is charged "
                       "for noise beyond the reference's floor")
    if cfg.criterion == "harmful_excess" and ((px.get("noisy") or {}).get("candidate") or
                                              (px.get("channel_noisy") or {}).get("candidate")):
        lim.append("harmful_excess subtracts the reference's floor only; a noisy candidate is charged for the part "
                   "of its own noise beyond that floor")
    return {"criteria": crit_out, "noise": noise_out, "power": power_out, "deterministic_mode": det,
            "estimates_add": est_add}, lim


def build_report(*, cfg, state, plan, stats, records, looks, verdict, bound, first_verdict, profiles, profile_diff,
                 channels, coupling, meter, gradebook, prepared, skipped, bank_items, phases, started, elapsed,
                 bank_digest_value, extra=None) -> dict:
    extra = extra or {}
    paired = extra.get("paired") is not None
    n = len(records)
    h = sum(1 for r in records if r["flip"] < 0)
    g = sum(1 for r in records if r["flip"] > 0)
    ref_ok = sum(1 for r in records if r["ref_grade"]["correct"])
    cand_ok = sum(1 for r in records if r["cand_grade"]["correct"])
    alpha = cfg.alpha
    est = {
        "n": n, "harmful": h, "helpful": g, "unchanged": n - h - g,
        "identical_outputs": sum(1 for r in records if r["identical"]),
        "stats_backend": stats.name,
        "bounds_note": "each bound is a one-sided (1 - alpha) bound; lower and upper are separate statements",
    }
    if n:
        est["harmful_flip_rate"] = stats.rate_bound(h, n, alpha).to_dict()
        est["helpful_flip_rate"] = stats.rate_bound(g, n, alpha).to_dict()
        est["churn_rate"] = stats.rate_bound(h + g, n, alpha).to_dict()
        est["accuracy"] = {"reference": ref_ok / n, "candidate": cand_ok / n}
        est["accuracy_change"] = stats.accuracy_change(g, n - h - g, h, alpha).to_dict()
    if paired:
        est["accuracy_change_note"] = ("the 0.1.0 unpaired union-of-Clopper-Pearson bound, kept for compatibility; "
                                       "the paired net change is estimates.net_change")
    per_suite = {}
    by_suite = defaultdict(list)
    for r in records:
        by_suite[r["suite"]].append(r)
    for s, rs in sorted(by_suite.items()):
        sh = sum(1 for r in rs if r["flip"] < 0)
        sg = sum(1 for r in rs if r["flip"] > 0)
        per_suite[s] = {"n": len(rs), "harmful": sh, "helpful": sg, "churn": (sh + sg) / len(rs),
                        "identical_outputs": sum(1 for r in rs if r["identical"]),
                        "reference_accuracy": sum(r["ref_grade"]["correct"] for r in rs) / len(rs),
                        "candidate_accuracy": sum(r["cand_grade"]["correct"] for r in rs) / len(rs),
                        "harmful_rate_upper_descriptive": stats.rate_upper(sh, len(rs), alpha),
                        "note": "per-suite bounds are descriptive (not adjusted for multiplicity)"}
        if paired:
            cc = _paired_counts(rs, lambda r: r.get("residual_net"))
            per_suite[s]["net_change"] = _bound_of(cc, alpha) if cc.n else None
    calls = dict(gradebook.calls)
    cand_calls = sum(r["cand_grading_calls"] for r in records)
    inheritance = {
        "rule": "candidate output byte-identical to the reference output inherits the reference grade under a "
                "deterministic grader; judge verdicts are reused for identical (prompt, output) pairs",
        "audit_fraction": cfg.audit_fraction,
        "grading_calls": calls,
        "candidate_outputs": n,
        "baseline_candidate_grading_calls": n,
        "candidate_grading_calls_with_inheritance": cand_calls,
        "coupling_overhead_grading_calls": {
            "audit_regrades": sum(1 for r in records if r["cand_grade_how"] in ("inherited_audited", "audit_mismatch")),
            "channel_probe_gradings": 0,
            "note": "channel probes are generated but never graded; the audit re-grades are the only grading overhead"},
        "reduction_factor": (n / cand_calls) if cand_calls else None,
        "audit_mismatches": gradebook.audit_mismatches,
        "by_reference_output_length": _bucket_stats(records),
        "bucket_rule": "bucketed by the reference output's token count (cumulative: <=64 is inside <=256)",
    }
    fins = {"reference": Counter(r["ref"]["finish"] for r in records),
            "candidate": Counter(r["cand"]["finish"] for r in records)}
    fdiv = [r["first_divergence_token"] for r in records if r.get("first_divergence_token") is not None]
    diagnostics = {
        "finish_reasons": {k: dict(v) for k, v in fins.items()},
        "reference_errors": sum(1 for r in records if r["ref"]["error"]),
        "candidate_errors": sum(1 for r in records if r["cand"]["error"]),
        "truncation": _truncation(records),
        "first_divergence_token": {"pairs_with_token_ids_and_divergence": len(fdiv),
                                   "median": statistics.median(fdiv) if fdiv else None},
    }
    limitations = []
    if skipped:
        limitations.append(f"suites skipped: {skipped}")
    if coupling and not coupling.get("coupled", True):
        limitations.append("sampling is not coupled between the stacks; verdict uses the excess harmful-flip rate "
                           "over a reference replicate with wider union bounds")
    for f in (coupling or {}).get("flags", []):
        limitations.append(f)
    if state.get("stopped_by"):
        limitations.append(f"run stopped by the {state['stopped_by']} cap after {n} items; "
                           "EQUIVALENT_WITHIN_TAU can only be declared at the planned n_max")
    if diagnostics["truncation"]["silent_truncation_suspected"]:
        limitations.append("candidate outputs look silently truncated (prefixes of the reference output ending "
                           "with a normal stop, or a 'length' finish below the cap)")
    sections, lim2 = (_paired_sections(cfg, records, looks, extra, channels, coupling, profiles) if paired
                      else (None, []))
    if paired:
        limitations += lim2
    elif coupling and any("not deterministic under greedy decoding" in f for f in coupling.get("flags", [])):
        limitations.append("harmful_abs counts a stack's own nondeterminism as harm; the default criterion "
                           "(net_change) measures the noise floor and judges against it")
    if not paired and coupling and coupling.get("independent_bounds"):
        limitations.append("harmful_abs keeps the 0.1.0 sampling construction: the candidate shares the reference's "
                           "seed while the reference's replicate has its own, so when both stacks run the same "
                           "sampler the subtracted floor is sampling noise the cross pair does not have, and the "
                           "excess bound can be too optimistic; the default criterion gives every run its own seed")
    cost = meter.snapshot()
    bad_usage = {k: v.get("invalid_usage_requests") for k, v in cost["per_stack"].items()
                 if v.get("invalid_usage_requests")}
    if bad_usage:
        limitations.append(f"usage reports outside their domain (negative, non-integer or malformed) were never used "
                           f"as budget operands; those requests were charged at their worst case: {bad_usage}")
    if cost.get("refusals"):
        limitations.append(f"the cost meter refused {len(cost['refusals'])} operand(s) and stopped the run")
    unpriced = [role for role, attr in (("reference", "ref_price"), ("candidate", "cand_price"))
                if (profiles.get(role) or {}).get("kind") not in (None, "hf") and getattr(cfg, attr, None) is None]
    if unpriced:
        limitations.append(f"no price was given for the endpoint stack(s) {unpriced}: their dollar cost is not "
                           f"metered (absent, counted as $0 in usd_total); token counts are still metered and "
                           f"--max-tokens still applies")
    rep = {
        "schema": SCHEMA,
        "tool": {"name": "inferdrift", "version": __version__},
        "status": state["status"],
        "refusal": state.get("refusal"),
        "error": state.get("error"),
        "settings": cfg.to_dict(),
        "bank": {"spec": cfg.bank, "items_available": len(bank_items), "digest": bank_digest_value,
                 "suites": dict(Counter(it["suite"] for it in bank_items)), "skipped_suites": skipped},
        "plan": plan.to_dict(cfg.criterion, len({it["suite"] for it in _planned(cfg, bank_items, plan)}))
        if paired else plan.to_dict(),
        "verdict": {"verdict": verdict, "tau": cfg.tau, "alpha": cfg.alpha, "criterion": cfg.criterion,
                    "statistic": (criteria.STATISTIC[cfg.criterion] if paired else
                                  "excess_harmful_flip_rate" if coupling and coupling.get("independent_bounds")
                                  else "harmful_flip_rate"),
                    "bound": (bound.to_dict() if hasattr(bound, "to_dict") else bound) if bound else None,
                    "n": n, "n_max": plan.n_max,
                    "final_look_reached": bool(looks) and len(looks) == len(plan.looks) and n == plan.n_max,
                    "stopped_by": state.get("stopped_by"), "first_verdict": first_verdict},
        "looks": looks,
        "estimates": est,
        "per_suite": per_suite,
        "inheritance": inheritance,
        "diagnostics": diagnostics,
        "channels": channels,
        "coupling": coupling,
        "stacks": {"reference": profiles.get("reference"), "candidate": profiles.get("candidate"),
                   "diff": profile_diff},
        "downloads": prepared.get("plan") if isinstance(prepared, dict) else None,
        "work_dir": {"removed": state.get("work_dir_removed"), "bytes_at_end": state.get("work_dir_bytes_at_end")},
        "cost": cost,
        "timing": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_s": round(elapsed, 3), "phases": phases,
                   "first_verdict_elapsed_s": (first_verdict or {}).get("elapsed_s")},
        "limitations": limitations,
    }
    if paired:
        px = extra["paired"]
        gate = px.get("gate") or {}
        need = gate.get("items_needed") or {}
        fin = px.get("final") or {}
        per = {}
        for c, cc in sorted((px.get("run_counts") or {}).items()):
            b = _bound_of(cc, cfg.alpha)
            per[c] = {"planned": _planned_counts(cfg, bank_items, plan).get(c, 0), "evaluated": cc.n, "bad": cc.bad,
                      "good": cc.good, "point": cc.point, "upper": b["upper"], "lower": b["lower"],
                      "within_tau": bool(cc.n) and b["upper"] <= cfg.tau,
                      "power_at_planned": (gate.get("power_per_capability") or {}).get(c)}
        rep["verdict"].update({
            "per_capability": per, "items_needed": need.get("items_needed") if gate.get("underpowered") else None,
            "excluded_items": px.get("excluded", 0),
            "pre_gate_deviation": px.get("pre_gate_deviation"),
            "gates": {"underpowered": gate.get("underpowered"), "power_at_planned": gate.get("power_at_planned"),
                      "power_requested": cfg.power, "floor_source": gate.get("floor_source"),
                      "floor": (gate.get("floor") or {}).get("value"),
                      "deviates_look": px.get("pre_gate_deviation"), "all_within_tau": fin.get("all_within"),
                      "every_planned_item_evaluated": px.get("complete"),
                      "stopped_after_probe_underpowered": px.get("stopped_underpowered")}})
        rep["estimates"].update(sections["estimates_add"])
        rep["criteria"] = sections["criteria"]
        rep["noise"] = sections["noise"]
        rep["power"] = sections["power"]
        rep["power"]["planned_items"] = plan.n_max
        rep["power"]["planned_per_capability"] = _planned_counts(cfg, bank_items, plan)
        rep["deterministic_mode"] = sections["deterministic_mode"]
    else:
        rep["criteria"] = {"gating": "harmful_abs", "note": "the 0.1.0 rule (plan.py); the paired criteria are not "
                                                            "computed under harmful_abs"}
        rr = noise.pair_stats(records, "ref", "rep", "ref_grade", "rep_grade", cfg.alpha) \
            if any("rep" in r for r in records) else None
        rep["noise"] = {"floor": "not measured (harmful_abs keeps the 0.1.0 request pattern; the default criterion "
                                 "measures the floor)" if rr is None else
                        {"reference_replicate": rr, "note": "0.1.0 independent-sampling replicate"}}
        import math as _m
        rep["power"] = {"criterion": "harmful_abs", "gate": None,
                        "zero_flip_items_for_tau": _m.ceil(_m.log(cfg.alpha) / _m.log1p(-cfg.tau)),
                        "note": "harmful_abs has no power gate; with no harmful flip at all the 0.1.0 rule needs "
                                "this many items to reach EQUIVALENT_WITHIN_TAU"}
        rep["deterministic_mode"] = {role: determinism.section(
            (profiles.get(role) or {}).get("kind"), (extra.get("declared") or {}).get(role) or {},
            {"items": 0, "byte_identical": 0, "answer_flips": 0,
             "channel_repeat_identity": ((channels or {}).get(role) or {}).get(
                 "greedy_repeat_identity_rate_without_errors"),
             "batch_composition_varied": (extra.get("adapters_batching") or {}).get(role)})
            for role in ("reference", "candidate")}
    rep["reference_cache"] = extra.get("reference_cache") or {"enabled": False}
    return scrub(rep)


def _planned(cfg, bank_items, plan) -> list[dict]:
    import random as _random

    order = list(bank_items)
    if cfg.order == "random":
        _random.Random(cfg.seed).shuffle(order)
    return order[:plan.n_max]


def _planned_counts(cfg, bank_items, plan) -> dict:
    return dict(sorted(Counter(it["suite"] for it in _planned(cfg, bank_items, plan)).items()))


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x + 0.0:.2f}%"


def _f(x, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _paired_summary(rep: dict) -> list[str]:
    v = rep["verdict"]
    est = rep["estimates"]
    pw = rep.get("power") or {}
    nz = rep.get("noise") or {}
    fl = nz.get("floor") or {}
    lines = []
    crit = v.get("criterion")
    what = ("net accuracy loss (reference minus candidate) per capability; helpful and harmful flips offset each "
            "other inside a capability" if crit == "net_change" else
            "excess harmful-flip rate over the reference's own replicate floor, per capability; helpful flips do "
            "not offset harmful ones")
    lines.append(f"- Criterion: `{crit}`: {what}. Power requested: {pw.get('power_requested')}.")
    if v["verdict"] == "INCONCLUSIVE_UNDERPOWERED":
        if pw.get("items_needed"):
            short = ", ".join(f"{s['capability']} {s['planned']} -> {s['needed']}" for s in
                              (pw.get("short_capabilities") or []))
            lines.append(f"- Underpowered: the planned {pw.get('planned_items')} items have power "
                         f"{_f(pw.get('power_at_planned'), 3)} for tau = {v['tau']}; about {pw['items_needed']} items "
                         f"in the bank's proportions would do" + (f" (capabilities short: {short})" if short else "")
                         + f". These items support tau >= {pw.get('smallest_tau_at_planned')} per capability.")
        elif pw.get("items_needed_exceeds"):
            lines.append(f"- Underpowered: more than {pw['items_needed_exceeds']} items per capability would be "
                         f"needed. These items support tau >= {pw.get('smallest_tau_at_planned')} per capability.")
        elif pw.get("floor_source") == "unmeasured":
            lines.append(f"- No verdict: the noise floor was not measured; at least {pw.get('items_needed_lower_bound')}"
                         f" items would be needed even with no noise.")
    d = v.get("pre_gate_deviation")
    if d and not d.get("allowed_by_power_gate"):
        lines.append(f"- Evidence of a deviation (not a verdict, the plan is underpowered): at look {d['look']} "
                     f"(n = {d['n']}) the lower bound exceeded tau for {', '.join(d['capabilities'])} "
                     f"({', '.join(f'{c}: {_f(b)}' for c, b in d['lower_bounds'].items())}).")
    if isinstance(fl, dict) and fl:
        basis = fl.get("basis") or {}
        items = fl.get("items") or {}
        lines.append(f"- Noise floor (answer-level repeat-flip rate): {_pct(fl.get('value'))}, the larger of the two "
                     f"stacks (reference: {basis.get('reference')}, {items.get('reference')} items run twice; "
                     f"candidate: {basis.get('candidate')}, {items.get('candidate')} items); the power gate used the "
                     f"{(nz.get('gate_floor') or {}).get('source')} floor.")
    lines += ["", "## Against the noise floor (one-sided bounds at 1 - alpha)", "",
              "| quantity | cross-stack | replicate (reference twice) | paired excess | lower | upper |",
              "|---|---:|---:|---:|---:|---:|"]
    st = (nz.get("per_stack") or {}).get("reference") or {}
    rr = st.get("items") or 0
    for label, cross_key, rep_count, key in (
            ("harmful flips", "harmful_flip_rate", st.get("harmful_repeat_flips"), "excess_harmful_flip_rate"),
            ("helpful flips", "helpful_flip_rate", st.get("helpful_repeat_flips"), "excess_helpful_flip_rate"),
            ("churn (any flip)", "churn_rate", st.get("answer_flips"), "excess_churn")):
        cross = (est.get(cross_key) or {}).get("point")
        b = est.get(key) or {}
        lines.append(f"| {label} | {_pct(cross)} | {_pct(rep_count / rr if rr else None)} | {_pct(b.get('point'))} | "
                     f"{_pct(b.get('lower'))} | {_pct(b.get('upper'))} |")
    nb = est.get("net_change") or {}
    rb = est.get("replicate_net_change") or {}
    lines.append(f"| net loss (reference - candidate) | {_pct(nb.get('point'))} | "
                 f"{_pct(0.0 - rb['point'] if rb.get('point') is not None else None)} | | {_pct(nb.get('lower'))} | "
                 f"{_pct(nb.get('upper'))} |")
    lines += ["", f"Replicate columns use the items where the reference ran twice "
                  f"({(est.get('excess_harmful_flip_rate') or {}).get('basis', 'none')}).",
              "", "## Verdict per capability (paired bound on the gating statistic)", "",
              "| capability | planned | evaluated | losses | gains | estimate | lower | upper | within tau | power |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for c, d2 in (v.get("per_capability") or {}).items():
        lines.append(f"| {c} | {d2['planned']} | {d2['evaluated']} | {d2['bad']} | {d2['good']} | {_pct(d2['point'])} | "
                     f"{_pct(d2['lower'])} | {_pct(d2['upper'])} | {'yes' if d2['within_tau'] else 'no'} | "
                     f"{_f(d2.get('power_at_planned'), 3)} |")
    dm = rep.get("deterministic_mode") or {}
    lines += ["", "## Deterministic mode", ""]
    for role, d3 in dm.items():
        lines.append(f"- {role}: {d3['status']} ({d3['basis']}); declared: spec {d3['declared'].get('spec')}, "
                     f"torch deterministic algorithms {d3['declared'].get('torch_deterministic_algorithms')}"
                     + (f". {d3['recommendation']}" if d3.get("recommendation") else ""))
    rc = rep.get("reference_cache") or {}
    if rc.get("enabled"):
        lines += ["", "## Reference cache", "", f"Entry found: {rc.get('entry_found')}; hits {rc.get('hits')}; "
                  f"misses {rc.get('misses')}; reference generation requests this run: "
                  f"{rc.get('reference_generations')}; written: {rc.get('written')}."]
    return lines


def write_summary(path, rep: dict) -> None:
    v = rep["verdict"]
    est = rep["estimates"]
    paired = v.get("criterion") in criteria.PAIRED_CRITERIA
    lines = [f"# inferdrift compare: {v['verdict'] or rep['status'].upper()}", ""]
    st = rep["settings"]
    lines.append(f"- Reference: `{st['ref']}`")
    lines.append(f"- Candidate: `{st['cand']}`")
    lines.append(f"- tau = {100 * v['tau']:.2f} points, alpha = {v['alpha']}; bank `{rep['bank']['spec']}` "
                 f"({rep['bank']['items_available']} items), n = {v['n']} of n_max {v['n_max']}")
    cp = rep.get("coupling") or {}
    lines.append(f"- Decoding: {st['decode']['mode']}; coupling: {cp.get('mode')} ({cp.get('reason')})")
    fv = v.get("first_verdict") or {}
    if fv:
        lines.append(f"- First verdict: {fv.get('verdict')} after {fv.get('elapsed_s')} s at n = {fv.get('n')}")
    if rep.get("refusal"):
        lines.append(f"- REFUSED: {rep['refusal']['message']}")
    if rep.get("error"):
        lines.append(f"- ERROR: {rep['error']['message']}")
    if paired:
        lines += _paired_summary(rep)
    else:
        lines += ["", "## Change estimates (one-sided bounds at 1 - alpha)", "",
                  "| quantity | estimate | lower | upper |", "|---|---:|---:|---:|"]
        for key, label in (("harmful_flip_rate", "harmful flip rate"), ("helpful_flip_rate", "helpful flip rate"),
                           ("churn_rate", "churn (any flip)"), ("accuracy_change", "accuracy change (cand - ref)")):
            b = est.get(key)
            if b:
                lines.append(f"| {label} | {_pct(b['point'])} | {_pct(b['lower'])} | {_pct(b['upper'])} |")
    if est.get("accuracy"):
        lines.append(f"\nAccuracy: reference {_pct(est['accuracy']['reference'])}, "
                     f"candidate {_pct(est['accuracy']['candidate'])}; statistics: {est['stats_backend']}.")
    lines += ["", "## Per-suite churn", "", "| suite | n | harmful | helpful | churn | identical | ref acc | cand acc |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, d in rep["per_suite"].items():
        lines.append(f"| {s} | {d['n']} | {d['harmful']} | {d['helpful']} | {_pct(d['churn'])} | "
                     f"{d['identical_outputs']} | {_pct(d['reference_accuracy'])} | {_pct(d['candidate_accuracy'])} |")
    inh = rep["inheritance"]
    red = "n/a" if inh["reduction_factor"] is None else "%.1fx" % inh["reduction_factor"]
    lines += ["", "## Grading with exact inheritance", "",
              f"Candidate outputs: {inh['candidate_outputs']}; grading calls with inheritance (audits included): "
              f"{inh['candidate_grading_calls_with_inheritance']}; reduction: {red}."]
    ch = rep.get("channels") or {}
    lines += ["", "## Channels", ""]
    for role, c in ch.items():
        lines.append(f"- {role}: greedy repeat identity {c.get('greedy_repeat_identity_rate')}, top-k logprobs "
                     f"{c.get('top_logprobs')} (k={c.get('top_logprobs_k')}), prompt logprobs {c.get('prompt_logprobs')}, "
                     f"seed honoured {c.get('seed_honored')}")
    cost = rep["cost"]
    lines += ["", "## Cost", "", f"${cost['usd_total']:.4f}, {cost['tokens_total']} tokens, {cost['wall_s']} s wall; "
              f"caps {cost['caps']}; stopped by: {cost['stopped_by']}."]
    if rep["limitations"]:
        lines += ["", "## Flags and limitations", ""] + [f"- {x}" for x in rep["limitations"]]
    atomic_write_text(path, "\n".join(lines) + "\n")
