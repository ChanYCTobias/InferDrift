# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""JSON report and one-page summary."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from . import __version__
from .jsonio import atomic_write_text
from .redact import scrub

SCHEMA = "inferdrift/report/1"
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


def build_report(*, cfg, state, plan, stats, records, looks, verdict, bound, first_verdict, profiles, profile_diff,
                 channels, coupling, meter, gradebook, prepared, skipped, bank_items, phases, started, elapsed,
                 bank_digest_value) -> dict:
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
        "plan": plan.to_dict(),
        "verdict": {"verdict": verdict, "tau": cfg.tau, "alpha": cfg.alpha,
                    "statistic": ("excess_harmful_flip_rate" if coupling and coupling.get("independent_bounds")
                                  else "harmful_flip_rate"),
                    "bound": bound.to_dict() if bound else None, "n": n, "n_max": plan.n_max,
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
    return scrub(rep)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def write_summary(path, rep: dict) -> None:
    v = rep["verdict"]
    est = rep["estimates"]
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
