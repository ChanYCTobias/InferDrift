# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Command line: ``inferdrift <command>``.

Exit codes: 0 run completed (the verdict is in the report; with ``--exit-on-verdict``
0 means EQUIVALENT_WITHIN_TAU, 10 DEVIATES, 11 INCONCLUSIVE, 12
INCONCLUSIVE_UNDERPOWERED), 3 refused (before any work started, or during a run
when its integrity cannot be kept: a missing second run, a corrupt reference
cache), 4 error.

Every numeric option is read with a strict grammar and checked against its closed
domain before anything is sent or written: caps and durations are
finite and > 0, prices finite and >= 0, counts plain positive integers, tau and
alpha in (0, 1), the audit fraction in [0, 1].  ``nan``, ``inf``, ``1e999``,
negative or zero caps, booleans and other coercions are refused with exit 3 and a
JSON line ``{"status": "refused", "code": "BAD_INPUT", "field": ..., "message": ...}``.
A malformed command line (unknown option, invalid choice, missing argument) is
refused the same way (code ``BAD_ARGUMENTS``).  Everything printed is strict JSON.

``compare --plan FILE`` (experimental) reads a requalification plan the same way:
strict JSON, closed numeric domains, and no code from the file is executed.  With no
``--ref`` and no ``--cand`` it reproduces the certificate from the recorded grades
beside the plan.  With both stacks it runs the candidate only on the plan's census
and audit items, then writes that certificate.  Omitting ``--plan`` leaves every
other option and the comparison itself unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from . import domains as D
from .adapters.base import MAX_SEED, DecodeConfig
from .errors import BadInputError, FidelityError, RefusedError
from .jsonio import StrictJSONError, dumps_strict, loads_strict
from .meter import Caps, PriceCard
from .redact import scrub_text

EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 3, 4
VERDICT_EXIT = {"EQUIVALENT_WITHIN_TAU": 0, "DEVIATES": 10, "INCONCLUSIVE": 11, "INCONCLUSIVE_UNDERPOWERED": 12}


class _ArgumentError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """argparse that reports a malformed command line as a refusal (exit 3), not exit 2."""

    def error(self, message: str):  # noqa: D401 - argparse hook
        raise _ArgumentError(message)


def _looks(text: str | None) -> list[int] | None:
    if text is None:
        return None
    parts = text.split(",")
    if not parts or any(p == "" for p in parts):
        raise BadInputError("--looks", "must be comma-separated plain integers", text)
    return [D.parse_integer(p, "--looks", ge=1) for p in parts]


def _caps(a, *, tokens: bool = True) -> Caps:
    return Caps(D.parse_optional_number(a.max_usd, "--max-usd", gt=0.0),
                D.parse_optional_integer(a.max_tokens, "--max-tokens", ge=1) if tokens else None,
                D.parse_optional_number(a.max_wall_s, "--max-wall-s", gt=0.0))


def _transient(a) -> float:
    x = D.parse_number(a.transient_cap_gib, "--transient-cap-gib", gt=0.0)
    D.gib_to_bytes(x, "--transient-cap-gib")
    return x


def _missing_stacks(a) -> int | None:
    """The historical refusal when ``compare`` is invoked without ``--ref`` or ``--cand``."""
    if a.plan is not None:
        return None
    missing = [flag for flag, value in (("--ref", a.ref), ("--cand", a.cand)) if value is None]
    if not missing:
        return None
    _refusal("BAD_ARGUMENTS", "the following arguments are required: " + ", ".join(missing))
    return EXIT_REFUSED


def _plan_bank(plan: dict, bank_spec: str) -> tuple[str, str]:
    """Write a bank of only the plan's census and audit items. Returns (directory, bank path)."""
    import tempfile

    from .banks import load_bank
    from .requalify import selected_bank_items

    chosen = selected_bank_items(plan, load_bank(bank_spec))
    directory = tempfile.mkdtemp(prefix="inferdrift-plan-bank-")
    path = str(Path(directory) / "bank.jsonl")
    Path(path).write_text("".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in chosen),
                          encoding="utf-8")
    return directory, path


def cmd_compare(a) -> int:
    plan = None
    if a.plan is not None:
        if not isinstance(a.plan, str) or a.plan.strip() == "":
            raise BadInputError("--plan", "is empty; an empty path is not an absent --plan")
        from .plan_stats import PlanStatsError, ensure_ready

        try:
            ensure_ready()
        except PlanStatsError as exc:
            raise BadInputError("--plan", f"missing dependency: {exc.field}: {exc.detail}") from None
        from .requalify import certify, grades_from_compare_records, load_plan, load_recorded_grades

        plan = load_plan(a.plan)
        if a.ref is None and a.cand is None:
            certificate = certify(plan, load_recorded_grades(a.plan, plan))
            print(dumps_strict(certificate))
            return EXIT_OK
        if a.ref is None or a.cand is None:
            raise BadInputError(
                "plan",
                "running the candidate needs both --ref and --cand; omit both to reproduce the certificate "
                "from the recorded grades beside the plan",
            )
    else:
        missing = _missing_stacks(a)
        if missing is not None:
            return missing
    bank_dir = None
    if a.criterion is None:  # the experimental --plan path keeps the 0.1.0 rule unless a criterion is given
        a.criterion = "harmful_abs" if plan is not None else "net_change"
    if plan is not None:
        # The certificate counts every selected item, so the run does not stop at the first DEVIATES look.
        a.no_early_stop = True
        bank_dir, a.bank = _plan_bank(plan, a.bank)
    try:
        return _cmd_compare_body(a, plan)
    finally:
        if bank_dir is not None:
            import shutil

            shutil.rmtree(bank_dir, ignore_errors=True)


def _cmd_compare_body(a, plan) -> int:
    from .compare import CompareConfig, run_compare

    seed = D.parse_integer(a.seed, "--seed", ge=0, le=MAX_SEED)
    temperature = D.parse_number(a.temperature, "--temperature", gt=0.0, le=1e4)
    top_p = D.parse_number(a.top_p, "--top-p", gt=0.0, le=1.0)
    top_k = D.parse_integer(a.top_k, "--top-k", ge=0, le=1 << 31)
    decode = DecodeConfig("sample", temperature=temperature, top_p=top_p, top_k=top_k, seed=seed) \
        if a.decode == "sample" else DecodeConfig("greedy", seed=seed)
    cfg = CompareConfig(
        ref=a.ref, cand=a.cand, bank=a.bank,
        max_items=D.parse_optional_integer(a.max_items, "--max-items", ge=1),
        tau=D.parse_number(a.tau, "--tau", gt=0.0, lt=1.0),
        alpha=D.parse_number(a.alpha, "--alpha", gt=0.0, lt=1.0),
        seed=seed, decode=decode, out_dir=a.out, caps=_caps(a),
        ref_price=PriceCard.parse(a.ref_price, "--ref-price"), cand_price=PriceCard.parse(a.cand_price, "--cand-price"),
        ref_api_key_env=D.env_name(a.ref_api_key_env, "--ref-api-key-env"),
        cand_api_key_env=D.env_name(a.cand_api_key_env, "--cand-api-key-env"),
        work_dir=a.work_dir, keep_work_dir=a.keep_work_dir, transient_cap_gib=_transient(a),
        threads=D.parse_optional_integer(a.threads, "--threads", ge=1, le=1024),
        batch_size=D.parse_optional_integer(a.batch_size, "--batch-size", ge=1, le=4096),
        early_stop=not a.no_early_stop,
        audit_fraction=D.parse_number(a.audit_fraction, "--audit-fraction", ge=0.0, le=1.0),
        skip_channels=a.skip_channels, stats_mode=a.stats, order=a.order, looks=_looks(a.looks),
        sandbox_weak_ok=a.sandbox_weak_ok,
        template_overhead_tokens=D.parse_integer(a.template_overhead_tokens, "--template-overhead-tokens", ge=0,
                                                 le=1 << 20),
        judge=a.judge, judge_api_key_env=D.env_name(a.judge_api_key_env, "--judge-api-key-env"),
        judge_price=PriceCard.parse(a.judge_price, "--judge-price"), judge_cache=a.judge_cache,
        criterion=a.criterion, power=D.parse_number(a.power, "--power", gt=0.0, lt=1.0),
        floor_probe_items=D.parse_optional_integer(a.floor_probe_items, "--floor-probe-items", ge=0, le=1_000_000),
        replicate=a.replicate, plan_floor=D.parse_optional_number(a.plan_floor, "--plan-floor", ge=0.0, le=1.0),
        ref_cache=a.ref_cache, stop_if_underpowered=a.stop_if_underpowered)
    rep = run_compare(cfg)
    v = rep["verdict"]
    gates = v.get("gates") or {}
    line = {"status": rep["status"], "verdict": v["verdict"], "n": v["n"], "tau": v["tau"], "alpha": v["alpha"],
            "first_verdict_elapsed_s": rep["timing"]["first_verdict_elapsed_s"], "cost_usd": rep["cost"]["usd_total"],
            "stopped_by": rep["cost"]["stopped_by"], "report": str(Path(a.out) / "report.json"),
            "criterion": v.get("criterion"), "items_needed": v.get("items_needed"),
            "items_needed_note": v.get("items_needed_note"),
            "floor": {"value": gates.get("floor"), "source": gates.get("floor_source")} if gates else None,
            "pre_gate_deviation": v.get("pre_gate_deviation")}
    if plan is not None and rep["status"] == "completed":
        from .jsonio import read_jsonl_strict, write_json
        from .requalify import certify, grades_from_compare_records

        rows = [row for _, row in read_jsonl_strict(Path(a.out) / "items.jsonl")]
        certificate = certify(plan, grades_from_compare_records(plan, rows))
        write_json(Path(a.out) / "certificate.json", certificate, redact=False)
        line["requalification_certificate"] = certificate
    if rep.get("refusal"):
        line["refusal"] = rep["refusal"]["message"]
    if rep.get("error"):
        line["error"] = rep["error"]["message"]
    print(json.dumps(line, allow_nan=False))  # a non-finite value here is an error, never printed as NaN
    if rep["status"] == "refused":
        return EXIT_REFUSED
    if rep["status"] == "error":
        return EXIT_ERROR
    if a.exit_on_verdict:
        return VERDICT_EXIT.get(v["verdict"], EXIT_ERROR)
    return EXIT_OK


def cmd_inspect(a) -> int:
    """Passive inspection: never executes repository code or renders templates."""
    from . import hub, passive
    from .jsonio import write_json
    from .spec import parse_spec

    target = a.target
    D.env_name(a.api_key_env, "--api-key-env")
    if not target.startswith(("hf:", "http://", "https://")) and os.path.exists(target):
        target = "hf:" + os.path.abspath(target)
    spec = parse_spec(target)
    if spec.kind == "hf" and spec.local_path:
        p = os.path.expanduser(spec.local_path)
        prof = passive.inspect_local_dir(p) if os.path.isdir(p) else {"gguf": passive.read_gguf_file(p)}
    elif spec.kind == "hf":
        chk = hub.check_remote_code(spec.repo, spec.revision or "main")
        prof = {"source": "hub", "repo": spec.repo, "commit": chk["commit"], "flags": chk["flags"],
                "files": chk["files"],
                "config": passive.summarize_config(chk["config"] if isinstance(chk["config"], dict) else None)}
    else:
        from .adapters.http import HttpClient

        c = HttpClient(spec.base_url[:-3] if spec.base_url.endswith("/v1") else spec.base_url,
                       api_key_env=a.api_key_env, timeout_s=30, retries=0)
        prof = {"source": "endpoint", "routes": {}}
        for path in ("/v1/models", "/props", "/version", "/get_model_info", "/get_server_info"):
            try:
                st, obj, _ = c.get(path, allow_status=(401, 404, 405))
                prof["routes"][path] = {"status": st, "keys": sorted(obj)[:30] if isinstance(obj, dict) else None}
            except FidelityError as e:
                prof["routes"][path] = {"error": scrub_text(str(e))}
    out = {"tool": f"inferdrift {__version__}", "mode": "passive", "target": spec.display(), "profile": prof}
    if a.out:
        write_json(a.out, out)
    print(dumps_strict(out, indent=1)[: 200000])
    return EXIT_OK


def cmd_channels(a) -> int:
    import tempfile

    from .channels import detect_channels
    from .compare import CompareConfig, build_adapter, prepare_stacks
    from .meter import CostMeter
    from .spec import parse_spec

    spec = parse_spec(a.target)
    caps = _caps(a, tokens=False)
    price = PriceCard.parse(a.price, "--price")
    if caps.max_usd is not None and spec.kind == "endpoint" and price is None:
        raise BadInputError("--max-usd", "a dollar cap needs --price for an endpoint: an absent price is not a "
                                         "price of zero")
    threads = D.parse_optional_integer(a.threads, "--threads", ge=1, le=1024)
    transient = _transient(a)
    api_key_env = D.env_name(a.api_key_env, "--api-key-env")
    cfg = CompareConfig(ref=a.target, cand=a.target, threads=threads)
    work = Path(a.work_dir or tempfile.mkdtemp(prefix="inferdrift-work-"))
    try:
        prepared = prepare_stacks([("reference", spec)], work, transient)
        try:
            ad = build_adapter("reference", spec, prepared, cfg, api_key_env)
        except ImportError as e:  # e.g. mode=raw&tokenizer=hf:... without transformers: exit 4, not a traceback
            raise FidelityError(f"this spec needs the hf extra, which is not installed ({type(e).__name__}: {e})",
                                code="MISSING_EXTRA") from None
        meter = CostMeter(caps, {"reference": price})
        rep = {"profile": ad.profile(), "channels": detect_channels(ad, meter), "cost": meter.snapshot()}
    finally:
        if not a.work_dir:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
    print(dumps_strict(rep, indent=1))
    return EXIT_OK


def cmd_plan(a) -> int:
    import tempfile

    from . import hub
    from .errors import AllowanceExceededError
    from .spec import parse_spec

    cap = _transient(a)
    specs = [("reference", parse_spec(a.ref))] + ([("candidate", parse_spec(a.cand))] if a.cand else [])
    reqs = []
    for role, spec in specs:
        if spec.kind == "hf" and spec.repo:
            chk = hub.check_remote_code(spec.repo, spec.revision or "main")
            reqs.append(hub.RepoRequest(spec.repo, chk["commit"], hub.MODEL_PATTERNS, role))
    try:
        plan = hub.plan_downloads(reqs, Path(tempfile.gettempdir()), cap)
    except AllowanceExceededError as e:
        plan_obj = getattr(e, "plan", None)
        print(dumps_strict({"ok": False, "refused": scrub_text(str(e)),
                            "plan": plan_obj.to_dict() if plan_obj else None}, indent=1))
        return EXIT_REFUSED
    print(dumps_strict({"ok": True, "plan": plan.to_dict()}, indent=1))
    return EXIT_OK


def cmd_power(a) -> int:
    """Items needed (and power) for the paired criteria: no stacks, no network."""
    from . import power as pw

    crit = a.criterion
    tau = D.parse_number(a.tau, "--tau", gt=0.0, lt=1.0)
    alpha = D.parse_number(a.alpha, "--alpha", gt=0.0, lt=1.0)
    target = D.parse_number(a.power, "--power", gt=0.0, lt=1.0)
    floor = D.parse_number(a.floor, "--floor", ge=0.0, le=1.0)
    if crit == "net_change" and a.excess is not None:
        raise BadInputError("--excess", "applies to --criterion harmful_excess (use --loss with net_change)")
    if crit == "harmful_excess" and a.loss is not None:
        raise BadInputError("--loss", "applies to --criterion net_change (use --excess with harmful_excess)")
    effect_text = a.loss if crit == "net_change" else a.excess
    effect = D.parse_number(effect_text, "--loss" if crit == "net_change" else "--excess", ge=-1.0, le=1.0) \
        if effect_text is not None else 0.0
    m = D.parse_integer(a.capability_count, "--capability-count", ge=1, le=64)
    items = D.parse_optional_integer(a.items, "--items", ge=1, le=10 ** 7)
    model = pw.model_for(crit, floor, effect)
    weights = {f"capability_{i + 1}": 1 for i in range(m)}
    models = {c: model for c in weights}
    need = pw.items_needed(weights, models, alpha, tau, target)
    out = {"criterion": crit, "tau": tau, "alpha": alpha, "power": target, "floor": floor,
           ("loss" if crit == "net_change" else "excess"): effect, "capability_count": m, "model": model.to_dict(),
           "items_needed": need["items_needed"], "items_needed_per_capability": need["per_capability"],
           "items_needed_exceeds": need["exceeds"], "items_needed_checked_up_to": need["checked_up_to"],
           "method": "exact pass probability of the paired bound (the bound compare uses), product over equal "
                     "capabilities; items_needed is a total that reaches the power while one fewer does not, and "
                     "every total from it to items_needed_checked_up_to was checked to reach it too (5% more, at "
                     "least 50 and at most 200); exact power is not monotone in n, and larger totals were not "
                     "checked"}
    if need["items_needed"] is None:
        out["items_needed_search_stopped_at"] = need["searched_to"]
        out["items_needed_note"] = need["note"]
    if items is not None:
        counts = pw.allocate(items, weights)
        out["items"] = items
        out["power_at_items"] = pw.power_total(counts, models, alpha, tau)
        out["smallest_tau_at_items"] = pw.smallest_tau(counts, models, alpha, target)
    print(dumps_strict(out, indent=1))
    return EXIT_OK


def cmd_standin(a) -> int:
    from .standin import main_serve

    a.port = D.parse_integer(a.port, "--port", ge=0, le=65535)
    a.price_in = D.parse_number(a.price_in, "--price-in", ge=0.0, le=1e6)
    a.price_out = D.parse_number(a.price_out, "--price-out", ge=0.0, le=1e6)
    a.truncate_frac = D.parse_optional_number(a.truncate_frac, "--truncate-frac", gt=0.0, le=1.0)
    a.nondeterministic = D.parse_number(a.nondeterministic, "--nondeterministic", ge=0.0, le=1.0)
    a.fail_every = D.parse_integer(a.fail_every, "--fail-every", ge=0, le=1 << 20)
    a.noise_seed = D.parse_integer(a.noise_seed, "--noise-seed", ge=0, le=MAX_SEED)
    D.env_name(a.api_key_env, "--api-key-env")
    return main_serve(a)


def cmd_banks(a) -> int:
    from .banks import ALL_SUITES, BUNDLES, bank_digest, load_bank

    if a.show:
        items = load_bank(a.show)
        print(dumps_strict({"bank": a.show, "items": len(items), "digest": bank_digest(items),
                            "first": items[0] if items else None}, indent=1))
    else:
        out = {s: len(load_bank(s)) for s in ALL_SUITES}
        out["bundles"] = {k: list(v) for k, v in BUNDLES.items()}
        print(dumps_strict(out, indent=1))
    return EXIT_OK


def _read_json_file(path: str, field: str):
    try:
        return loads_strict(Path(path).read_bytes())
    except StrictJSONError as e:
        raise BadInputError(field, f"{path} is not strict JSON ({e})") from None
    except OSError as e:
        raise BadInputError(field, f"{path} cannot be read ({type(e).__name__})") from None


def cmd_diff(a) -> int:
    from .profile import diff

    ra = _read_json_file(a.a, "profile-diff A")
    rb = _read_json_file(a.b, "profile-diff B")
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        raise BadInputError("profile-diff", "both inputs must be JSON objects (a profile or a report)")
    pa = (ra.get("stacks") or {}).get("reference") or ra
    pb = (rb.get("stacks") or {}).get("candidate") or rb
    print(dumps_strict(diff(pa, pb), indent=1, default=str))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="inferdrift", description="Coupled differential evaluation of two inference "
                "stacks with exact grade inheritance and flip-rate bounds.")
    p.add_argument("--version", action="version", version=f"inferdrift {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Numeric options are read as text and parsed strictly by the command (exit 3 on a bad value).
    c = sub.add_parser("compare", help="compare a candidate stack with a reference stack")
    c.add_argument("--ref", default=None, help="reference stack spec (hf:org/name[@rev][?...] or http(s)://...)")
    c.add_argument("--cand", default=None, help="candidate stack spec")
    c.add_argument("--plan", default=None, help="EXPERIMENTAL (its validation bar was not met; the format may "
                   "change). A requalification plan file. With --ref and --cand, run only that "
                   "plan's census and audit and write its certificate. With neither, reproduce the certificate "
                   "from the recorded grades beside the plan. The file is data and is never executed.")
    c.add_argument("--bank", default="core", help="core | full | smoke | suite[,suite] | path/to/bank.jsonl")
    c.add_argument("--max-items", default=None, help="planned n_max, a positive integer (default: the whole bank)")
    c.add_argument("--tau", default="0.01", help="tolerance in (0, 1) (0.01 = 1 point) on the criterion's statistic: "
                   "the net accuracy loss per capability (net_change, the default), the excess harmful-flip rate "
                   "over the noise floor (harmful_excess) or the harmful-flip rate (harmful_abs)")
    c.add_argument("--alpha", default="0.05", help="error rate, in (0, 1)")
    c.add_argument("--seed", default="0", help="non-negative integer")
    c.add_argument("--decode", choices=("greedy", "sample"), default="greedy")
    c.add_argument("--temperature", default="1.0", help="> 0 (sampling only)")
    c.add_argument("--top-p", default="1.0", help="in (0, 1] (sampling only)")
    c.add_argument("--top-k", default="0", help="non-negative integer; 0 = off (sampling only)")
    c.add_argument("--out", default="inferdrift-out")
    c.add_argument("--max-usd", default=None, help="hard dollar cap for the whole run, finite and > 0 (needs a "
                   "price for every endpoint stack)")
    c.add_argument("--max-tokens", default=None, help="hard token cap (prompt + completion), a positive integer")
    c.add_argument("--max-wall-s", default=None, help="hard wall-time cap in seconds, finite and > 0")
    c.add_argument("--ref-price", default=None, help="USD per 1M tokens 'IN,OUT' for the reference endpoint")
    c.add_argument("--cand-price", default=None, help="USD per 1M tokens 'IN,OUT' for the candidate endpoint")
    c.add_argument("--ref-api-key-env", default=None, help="name of the env var holding the reference API key")
    c.add_argument("--cand-api-key-env", default=None, help="name of the env var holding the candidate API key")
    c.add_argument("--work-dir", default=None, help="transient directory for downloads (removed at the end)")
    c.add_argument("--keep-work-dir", action="store_true")
    c.add_argument("--transient-cap-gib", default="6.0", help="refuse jobs whose downloads exceed this (GiB, > 0)")
    c.add_argument("--threads", default=None, help="CPU threads for the transformers adapter (default 8)")
    c.add_argument("--batch-size", default=None)
    c.add_argument("--no-early-stop", action="store_true", help="run to n_max even after DEVIATES")
    c.add_argument("--audit-fraction", default="0.05", help="share of inherited grades re-graded, in [0, 1]")
    c.add_argument("--skip-channels", action="store_true")
    c.add_argument("--stats", choices=("auto", "builtin"), default="auto",
                   help="statistics back end; both choices select the built-in exact binomial")
    c.add_argument("--order", choices=("random", "bank"), default="random")
    c.add_argument("--looks", default=None, help="comma-separated look sizes ending at n_max")
    c.add_argument("--sandbox-weak-ok", action="store_true", help="allow python-level socket blocking when "
                   "network namespaces are unavailable")
    c.add_argument("--template-overhead-tokens", default="512")
    c.add_argument("--judge", default=None, help="frozen judge stack spec for judge-graded items")
    c.add_argument("--judge-api-key-env", default=None)
    c.add_argument("--judge-price", default=None, help="USD per 1M tokens 'IN,OUT' for the judge endpoint")
    c.add_argument("--judge-cache", default=None, help="JSONL file of recorded judge verdicts (reused across runs)")
    c.add_argument("--exit-on-verdict", action="store_true")
    c.add_argument("--criterion", choices=("net_change", "harmful_excess", "harmful_abs"), default=None,
                   help="what the verdict certifies (default net_change; harmful_abs under --plan): net_change = "
                   "net accuracy loss per capability against the noise floor; harmful_excess = excess harmful-flip "
                   "rate over the reference's replicate floor; harmful_abs = the 0.1.0 rule, for deterministic stacks")
    c.add_argument("--power", default="0.9", help="power the plan must have, in (0, 1); below it the verdict is "
                   "INCONCLUSIVE_UNDERPOWERED with the number of items needed, or, when the search passes 20000 "
                   "items in the largest capability, the size where it stopped")
    c.add_argument("--floor-probe-items", default=None, help="items run twice on both stacks to measure the noise "
                   "floor (default 64; 0 turns the probe off)")
    c.add_argument("--plan-floor", default=None, help="a floor in [0, 1] measured beforehand, used by the power gate "
                   "unless this run measures a significantly higher one")
    c.add_argument("--replicate", choices=("auto", "always"), default="auto",
                   help="auto: run a stack twice on every item only when the floor needs it; always: both stacks")
    c.add_argument("--ref-cache", default=None, help="directory of the reference cache (reuses reference outputs "
                   "across candidates)")
    c.add_argument("--stop-if-underpowered", action="store_true", help="stop after the probe when the plan is "
                   "underpowered")
    c.set_defaults(fn=cmd_compare)

    pw = sub.add_parser("power", help="items needed for a tolerance at a noise floor (no stacks, no network)")
    pw.add_argument("--criterion", choices=("net_change", "harmful_excess"), default="net_change")
    pw.add_argument("--tau", default="0.01", help="tolerance in (0, 1)")
    pw.add_argument("--alpha", default="0.05", help="error rate in (0, 1)")
    pw.add_argument("--power", default="0.9", help="power in (0, 1)")
    pw.add_argument("--floor", default="0", help="answer-level repeat-flip rate in [0, 1]")
    pw.add_argument("--loss", default=None, help="net_change: true net loss to plan for (default 0)")
    pw.add_argument("--excess", default=None, help="harmful_excess: true excess harmful-flip rate (default 0)")
    pw.add_argument("--capability-count", default="1", help="number of capabilities, equal shares (default 1)")
    pw.add_argument("--items", default=None, help="also report the power at this many items and the smallest tau")
    pw.set_defaults(fn=cmd_power)

    i = sub.add_parser("inspect", help="passive inspection of a model repository, file or endpoint")
    i.add_argument("target")
    i.add_argument("--out", default=None)
    i.add_argument("--api-key-env", default=None)
    i.set_defaults(fn=cmd_inspect)

    ch = sub.add_parser("channels", help="detect a stack's observation and control channels")
    ch.add_argument("target")
    ch.add_argument("--api-key-env", default=None)
    ch.add_argument("--price", default=None, help="USD per 1M tokens 'IN,OUT' (needed with --max-usd for an endpoint)")
    ch.add_argument("--max-usd", default=None, help="dollar cap, finite and > 0")
    ch.add_argument("--max-wall-s", default="600", help="wall-time cap in seconds, finite and > 0")
    ch.add_argument("--work-dir", default=None)
    ch.add_argument("--threads", default=None)
    ch.add_argument("--transient-cap-gib", default="6.0")
    ch.set_defaults(fn=cmd_channels)

    pl = sub.add_parser("plan-downloads", help="plan (and possibly refuse) the downloads of a comparison")
    pl.add_argument("--ref", required=True)
    pl.add_argument("--cand", default=None)
    pl.add_argument("--transient-cap-gib", default="6.0")
    pl.set_defaults(fn=cmd_plan)

    s = sub.add_parser("standin", help="run the local metered stand-in endpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", default="8099")
    s.add_argument("--backend", default="synthetic")
    s.add_argument("--price-in", default="0", help="USD per 1M input tokens, finite and >= 0")
    s.add_argument("--price-out", default="0", help="USD per 1M output tokens, finite and >= 0")
    s.add_argument("--api-key-env", default=None)
    s.add_argument("--ledger", default=None)
    s.add_argument("--dialect", choices=("openai", "llamacpp", "vllm", "sglang"), default="openai")
    s.add_argument("--model-name", default="standin-model")
    s.add_argument("--ignore-seed", action="store_true")
    s.add_argument("--no-logprobs", action="store_true")
    s.add_argument("--truncate-frac", default=None)
    s.add_argument("--leak-key-in-errors", action="store_true")
    s.add_argument("--bad-usage", choices=("negative", "float", "string", "bool", "huge", "nan"), default=None,
                   help="fault injection: report malformed token usage (the ledger stays truthful)")
    s.add_argument("--nondeterministic", default="0")
    s.add_argument("--fail-every", default="0")
    s.add_argument("--noise-seed", default="0", help="seed of the replay noise model (lines with alt and flip); "
                   "two stand-ins with the same noise seed are the same engine")
    s.add_argument("--work-dir", default=None)
    s.set_defaults(fn=cmd_standin)

    b = sub.add_parser("banks", help="list the default item banks")
    b.add_argument("--show", default=None)
    b.set_defaults(fn=cmd_banks)

    d = sub.add_parser("profile-diff", help="diff two stack profiles (or two reports)")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(fn=cmd_diff)
    return p


def _refusal(code: str, message: str, field: str | None = None) -> None:
    obj = {"status": "refused", "code": code, "message": scrub_text(message)}
    if field is not None:
        obj["field"] = scrub_text(field)
    print(json.dumps(obj, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except _ArgumentError as e:
        _refusal("BAD_ARGUMENTS", str(e))
        return EXIT_REFUSED
    try:
        return int(args.fn(args) or 0)
    except BadInputError as e:
        _refusal(e.code, str(e), e.field)
        return EXIT_REFUSED
    except RefusedError as e:
        _refusal(e.code, str(e))
        return EXIT_REFUSED
    except FidelityError as e:
        print(json.dumps({"status": "error", "code": e.code, "message": scrub_text(str(e))}, allow_nan=False))
        return EXIT_ERROR
    except ValueError as e:  # e.g. a non-finite value reaching a strict JSON writer: an error, never NaN output
        print(json.dumps({"status": "error", "code": "VALUE_ERROR",
                          "message": scrub_text(f"{type(e).__name__}: {e}")[:500]}, allow_nan=False))
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
