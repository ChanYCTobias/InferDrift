# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The comparison run: prepare both stacks, detect channels, couple the decoders,
evaluate items in a seeded random order with sequential looks, grade with exact
inheritance, bound the harmful-flip rate and the accuracy change, meter the cost,
and write the report.

Ordering of safety checks (nothing below executes repository code):

1. parse both specs (keys are never part of a spec);
2. passive pre-check of every Hugging Face source (local: file scan; Hub: tree
   listing plus config.json/tokenizer_config.json read as JSON); a repository that
   declares custom code, or offers only pickle weights, is refused;
3. the download planner sums every file the job would fetch and refuses the job
   above the transient allowance or the free space, before writing a byte;
4. only then are files downloaded into the job's work directory, and only then is
   transformers imported to load safetensors weights without remote code;
5. the work directory is removed when the job ends (unless ``keep_work_dir``).

Before any of this, every input is checked against its closed domain:
``CompareConfig`` validates itself when it is built and again when the
run starts, the bank is loaded strictly and validated, and the sequential plan is
built.  A malformed value raises ``BadInputError`` before a request is sent or a
file is written (the CLI exits 3).  A dollar cap needs an explicit price for every
endpoint stack: an absent price is not a price of zero, and a cap that cannot see
a stack's cost would never bind.
"""

from __future__ import annotations

import dataclasses
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, domains, hub, passive
from .adapters.base import MAX_SEED, DecodeConfig, GenRequest, request_seed
from .banks import bank_digest, load_bank
from .channels import detect_channels
from .coupling import plan_coupling
from .errors import (BadInputError, CapReached, EndpointError, FidelityError, PartialCapReached, RefusedError,
                     UnsafeRepositoryError)
from .inherit import GradeBook, load_judge_cache
from .jsonio import JsonlWriter, write_json
from .meter import Caps, CostMeter, PriceCard
from .plan import DEVIATES, Plan, final_decision, look_decision, look_schedule
from .profile import diff as profile_diff
from .redact import register_secret, scrub_output_text, scrub_text
from .sandbox import namespaces_available
from .spec import StackSpec, parse_spec
from .stats import StatsBackend

CONSECUTIVE_ERROR_LIMIT = 8


@dataclass
class CompareConfig:
    """Settings of one comparison.  Every field has a closed domain; the
    constructor and ``run_compare`` both call ``validate`` and raise ``BadInputError``
    for a value outside it.  A price is optional: None means "no price given", which is
    not a price of zero (see ``validate``)."""

    ref: str
    cand: str
    bank: str = "core"
    max_items: int | None = None
    tau: float = 0.01
    alpha: float = 0.05
    seed: int = 0
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    out_dir: str = "inferdrift-out"
    caps: Caps = field(default_factory=Caps)
    ref_price: PriceCard | None = None
    cand_price: PriceCard | None = None
    ref_api_key_env: str | None = None
    cand_api_key_env: str | None = None
    work_dir: str | None = None
    keep_work_dir: bool = False
    transient_cap_gib: float = hub.DEFAULT_ALLOWANCE_GIB
    threads: int | None = None
    batch_size: int | None = None
    early_stop: bool = True
    audit_fraction: float = 0.05
    skip_channels: bool = False
    stats_mode: str = "auto"
    order: str = "random"
    looks: list[int] | None = None
    sandbox_weak_ok: bool = False
    template_overhead_tokens: int = 512
    save_items: bool = True
    judge: str | None = None
    judge_api_key_env: str | None = None
    judge_price: PriceCard | None = None
    judge_cache: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check every field against its domain; raise BadInputError on the first violation."""
        d = domains
        specs = {"ref": parse_spec(d.text(self.ref, "ref")), "cand": parse_spec(d.text(self.cand, "cand"))}
        if self.judge is not None:
            specs["judge"] = parse_spec(d.text(self.judge, "judge"))
        d.text(self.bank, "bank")
        d.optional_integer(self.max_items, "max_items", ge=1)
        d.number(self.tau, "tau", gt=0.0, lt=1.0)
        d.number(self.alpha, "alpha", gt=0.0, lt=1.0)
        d.integer(self.seed, "seed", ge=0, le=MAX_SEED)
        if not isinstance(self.decode, DecodeConfig):
            raise BadInputError("decode", "must be a DecodeConfig", self.decode)
        DecodeConfig(**self.decode.to_dict())  # re-check its domains
        d.text(self.out_dir, "out_dir")
        if not isinstance(self.caps, Caps):
            raise BadInputError("caps", "must be a Caps", self.caps)
        Caps(self.caps.max_usd, self.caps.max_tokens, self.caps.max_wall_s)  # re-check its domains
        for name in ("ref_price", "cand_price", "judge_price"):
            card = getattr(self, name)
            if card is not None:
                if not isinstance(card, PriceCard):
                    raise BadInputError(name, "must be a PriceCard or None", card)
                PriceCard(card.usd_per_mtok_in, card.usd_per_mtok_out)  # re-check its domains
        for name in ("ref_api_key_env", "cand_api_key_env", "judge_api_key_env"):
            d.env_name(getattr(self, name), name)
        d.optional_text(self.work_dir, "work_dir")
        d.optional_text(self.judge_cache, "judge_cache")
        for name in ("keep_work_dir", "early_stop", "skip_channels", "sandbox_weak_ok", "save_items"):
            d.boolean(getattr(self, name), name)
        d.gib_to_bytes(self.transient_cap_gib, "transient_cap_gib")
        d.optional_integer(self.threads, "threads", ge=1, le=1024)
        d.optional_integer(self.batch_size, "batch_size", ge=1, le=4096)
        d.number(self.audit_fraction, "audit_fraction", ge=0.0, le=1.0)
        d.choice(self.stats_mode, "stats_mode", ("auto", "builtin"))
        d.choice(self.order, "order", ("random", "bank"))
        if self.looks is not None:
            if not isinstance(self.looks, (list, tuple)) or not self.looks:
                raise BadInputError("looks", "must be a non-empty list of integers or None", self.looks)
            looks = [d.integer(x, "looks", ge=1) for x in self.looks]
            if any(b <= a for a, b in zip(looks, looks[1:])):
                raise BadInputError("looks", "must be strictly increasing", self.looks)
        d.integer(self.template_overhead_tokens, "template_overhead_tokens", ge=0, le=1 << 20)
        if self.caps.max_usd is not None:
            prices = {"ref": self.ref_price, "cand": self.cand_price, "judge": self.judge_price}
            unpriced = [role for role, spec in specs.items() if spec.kind == "endpoint" and prices[role] is None]
            if unpriced:
                flags = ", ".join(f"--{r}-price" for r in unpriced)
                raise BadInputError("max_usd", f"a dollar cap needs a price for every endpoint stack ({flags} "
                                               "missing): an absent price is not a price of zero, and the cap could "
                                               "not see that stack's cost")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["decode"] = self.decode.to_dict()
        d["caps"] = self.caps.to_dict()
        for name in ("ref_price", "cand_price", "judge_price"):
            card = getattr(self, name)
            d[name] = dataclasses.asdict(card) if card is not None else None
        return d


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- stacks
def _refuse_if_unsafe(flags: dict, where: str) -> None:
    if flags.get("malformed_json"):
        raise UnsafeRepositoryError(
            f"{where} has metadata that is not strict JSON ({flags['malformed_json']}): a duplicate key could hide "
            f"an auto_map entry, so the repository is refused rather than guessed at")
    if flags.get("remote_code_declared"):
        raise UnsafeRepositoryError(
            f"{where} declares custom code ({sorted(flags.get('auto_map', {}))}); inferdrift never executes "
            f"repository code (no trust_remote_code)")
    if not flags.get("safetensors_files"):
        raise UnsafeRepositoryError(
            f"{where} has no safetensors weights; pickle weights ({flags.get('pickle_weight_files')}) are never loaded")


def prepare_stacks(specs: list[tuple[str, StackSpec]], work_dir: Path, cap_gib: float) -> dict:
    """Passive checks, download plan (with refusal), downloads.  Returns per-role sources."""
    requests: list[hub.RepoRequest] = []
    sources: dict[str, dict] = {}
    for role, spec in specs:
        if spec.kind == "hf":
            if spec.local_path:
                path = os.path.expanduser(spec.local_path)
                prof = passive.inspect_local_dir(path)
                _refuse_if_unsafe(prof["flags"], f"local repository {path}")
                sources[role] = {"local_dir": path, "passive": prof, "commit": None}
            else:
                chk = hub.check_remote_code(spec.repo, spec.revision or "main")
                _refuse_if_unsafe(chk["flags"], f"Hub repository {spec.repo}")
                requests.append(hub.RepoRequest(spec.repo, chk["commit"], hub.MODEL_PATTERNS, role))
                sources[role] = {"hub": spec.repo, "commit": chk["commit"]}
        elif spec.kind == "endpoint" and spec.params.get("tokenizer"):
            tspec = parse_spec(spec.params["tokenizer"])
            if tspec.kind != "hf":
                raise RefusedError("tokenizer= must be an hf: spec")
            if tspec.local_path:
                path = os.path.expanduser(tspec.local_path)
                prof = passive.inspect_local_dir(path)
                if prof["flags"]["remote_code_declared"] or prof["flags"]["malformed_json"]:
                    raise UnsafeRepositoryError(f"tokenizer source {path} declares custom code or has metadata that "
                                                f"is not strict JSON; refused")
                sources[role + "_tokenizer"] = {"local_dir": path, "passive": prof, "commit": None}
            else:
                chk = hub.check_remote_code(tspec.repo, tspec.revision or "main")
                if chk["flags"]["remote_code_declared"]:
                    raise UnsafeRepositoryError(f"tokenizer source {tspec.repo} declares custom code; refused")
                requests.append(hub.RepoRequest(tspec.repo, chk["commit"], hub.TOKENIZER_PATTERNS,
                                                role + "_tokenizer"))
                sources[role + "_tokenizer"] = {"hub": tspec.repo, "commit": chk["commit"]}
    cap_bytes = domains.gib_to_bytes(cap_gib, "transient_cap_gib")
    plan = hub.plan_downloads(requests, work_dir, cap_gib) if requests else hub.DownloadPlan(cap_bytes=cap_bytes)
    t0 = time.monotonic()
    for entry in plan.entries:
        dest = work_dir / "models" / f"{entry['repo'].replace('/', '--')}@{entry['commit']}"
        hub.download(entry, dest)
        src = sources[entry["role"]]
        src["local_dir"] = str(dest)
        prof = passive.inspect_local_dir(dest)
        if entry["role"].endswith("_tokenizer"):
            if prof["flags"]["remote_code_declared"] or prof["flags"]["malformed_json"]:
                raise UnsafeRepositoryError("downloaded tokenizer declares custom code or has metadata that is not "
                                            "strict JSON; refused")
        else:
            _refuse_if_unsafe(prof["flags"], f"downloaded {entry['repo']}")
        prof["commit"] = entry["commit"]
        src["passive"] = prof
    return {"sources": sources, "plan": plan.to_dict(), "download_s": time.monotonic() - t0}


def detect_dialect(spec: StackSpec, api_key_env: str | None) -> str:
    d = spec.params.get("dialect", "auto")
    if d != "auto":
        return d
    from .adapters.http import HttpClient

    root = spec.base_url[:-3] if spec.base_url.endswith("/v1") else spec.base_url
    c = HttpClient(root, api_key_env=api_key_env, timeout_s=15, retries=0)
    for path, name, key in (("/props", "llamacpp", "default_generation_settings"),
                            ("/get_model_info", "sglang", "model_path"), ("/version", "vllm", "version")):
        try:
            status, obj, _ = c.get(path, allow_status=(404, 405, 401))
            if status == 200 and isinstance(obj, dict) and key in obj:
                return name
        except EndpointError:
            continue
    return "openai"


def build_adapter(role: str, spec: StackSpec, prepared: dict, cfg: CompareConfig, api_key_env: str | None):
    if spec.kind == "hf":
        from .adapters.hf import HFAdapter

        src = prepared["sources"][role]
        prof = dict(src["passive"])
        prof["commit"] = src.get("commit")
        return HFAdapter(spec, role, src["local_dir"], prof, threads=cfg.threads, batch_size=cfg.batch_size)
    renderer = None
    if spec.params.get("tokenizer"):
        from .adapters.render import HFRenderer

        renderer = HFRenderer(prepared["sources"][role + "_tokenizer"]["local_dir"])
    dialect = detect_dialect(spec, api_key_env)
    if dialect == "llamacpp":
        from .adapters.llamacpp import LlamaCppAdapter as cls
    elif dialect == "vllm":
        from .adapters.vllm import VLLMAdapter as cls
    elif dialect == "sglang":
        from .adapters.sglang import SGLangAdapter as cls
    else:
        from .adapters.openai_compat import OpenAICompatAdapter as cls
    return cls(spec, role, api_key_env=api_key_env, renderer=renderer,
               template_overhead_tokens=cfg.template_overhead_tokens)


def make_request(item: dict, decode: DecodeConfig, replicate: int = 0) -> GenRequest:
    seed = request_seed(decode.seed, item["id"], replicate) if decode.mode == "sample" else None
    # the replicate's seed wraps inside the seed domain, so a seed near its top never leaves it mid-run
    dec = decode if replicate == 0 else dataclasses.replace(decode, seed=(decode.seed + 7919 * replicate) % (MAX_SEED + 1))
    return GenRequest(item["id"], item["messages"], item.get("tools"), dict(item.get("template_kwargs") or {}),
                      int(item["max_new_tokens"]), dec, seed=seed, coupling_key=f"{item['id']}#{replicate}")


def _first_divergence(text_a: str, text_b: str, a: list | None, b: list | None) -> int | None:
    """Index of the first differing output token, for outputs whose texts differ.

    Only meaningful when both stacks report token ids from the same vocabulary;
    identical texts give None (engines may append different end tokens)."""
    if text_a == text_b or not isinstance(a, list) or not isinstance(b, list):
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _side(res) -> dict:
    return {"text": scrub_output_text(res.text), "finish": res.finish_reason, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "usage_source": res.usage_source,
            "prompt_sha256": res.prompt_sha256, "error": scrub_text(res.error) if res.error else None,
            "elapsed_s": round(res.elapsed_s, 4)}


# --------------------------------------------------------------------------- run
def run_compare(cfg: CompareConfig) -> dict:
    from .report import build_report, write_summary

    started = _utc()
    mono0 = time.monotonic()
    # ---- closed domains first: nothing is sent or written until every input,
    # the bank and the plan are valid; a violation raises BadInputError (CLI exit 3).
    if not isinstance(cfg, CompareConfig):
        raise BadInputError("cfg", "must be a CompareConfig", cfg)
    for env_name in (cfg.ref_api_key_env, cfg.cand_api_key_env, cfg.judge_api_key_env):
        if isinstance(env_name, str) and env_name:
            register_secret(os.environ.get(env_name))  # before any message could carry a key
    cfg.validate()
    items_all = load_bank(cfg.bank)
    skipped = {}
    if any(it["grader"]["type"] == "unit_tests" for it in items_all) and not namespaces_available() \
            and not cfg.sandbox_weak_ok:
        skipped = {s: "network-less sandbox unavailable (no user/network namespaces); pass --sandbox-weak-ok to "
                      "accept python-level socket blocking" for s in {it["suite"] for it in items_all
                                                                       if it["grader"]["type"] == "unit_tests"}}
        items_all = [it for it in items_all if it["grader"]["type"] != "unit_tests"]
    order = list(items_all)
    if cfg.order == "random":
        random.Random(cfg.seed).shuffle(order)
    n_max = len(order) if cfg.max_items is None else min(len(order), cfg.max_items)
    if n_max < 1:
        raise BadInputError("bank", "has no runnable items (every suite was skipped)", cfg.bank)
    plan = Plan(n_max, cfg.tau, cfg.alpha, looks=list(cfg.looks) if cfg.looks else look_schedule(n_max),
                early_stop=cfg.early_stop)
    meter = CostMeter(cfg.caps, {"reference": cfg.ref_price, "candidate": cfg.cand_price, "judge": cfg.judge_price})
    stats = StatsBackend(cfg.stats_mode)
    if cfg.judge is not None:
        load_judge_cache(cfg.judge_cache)  # a malformed cache is refused before any request
    gb = GradeBook(audit_fraction=cfg.audit_fraction, seed=cfg.seed, sandbox_weak_ok=cfg.sandbox_weak_ok)

    # ---- outputs
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("events.jsonl", "items.jsonl", "report.json", "summary.md"):  # files of an earlier run
        (out / stale).unlink(missing_ok=True)
    events = JsonlWriter(out / "events.jsonl")

    def event(kind: str, **kw) -> None:
        events.write({"event": kind, "utc": _utc(), "elapsed_s": round(time.monotonic() - mono0, 3), **kw})

    event("start", tool=f"inferdrift {__version__}")
    state: dict = {"status": "running", "error": None, "refusal": None, "stopped_by": None}
    created_work = False
    if cfg.work_dir:
        work = Path(cfg.work_dir)
        created_work = not work.exists()
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="inferdrift-work-"))
        created_work = True
    phases = {}
    records: list[dict] = []
    looks_out: list[dict] = []
    first_verdict = None
    profiles = {}
    channels = {}
    coupling = None
    prepared = {"plan": None, "sources": {}}
    adapters = {}
    counts = {"harmful": 0, "helpful": 0, "rr_harmful": 0, "n": 0}
    try:
        ref_spec, cand_spec = parse_spec(cfg.ref), parse_spec(cfg.cand)
        judge_spec = parse_spec(cfg.judge) if cfg.judge else None
        if any(it["grader"]["type"] == "judge" for it in items_all) and judge_spec is None:
            raise RefusedError("the bank has judge-graded items; pass --judge SPEC (a frozen judge model)")
        t = time.monotonic()
        stack_list = [("reference", ref_spec), ("candidate", cand_spec)] + ([("judge", judge_spec)] if judge_spec else [])
        prepared = prepare_stacks(stack_list, work, cfg.transient_cap_gib)
        phases["prepare_s"] = round(time.monotonic() - t, 3)
        phases["download_s"] = round(prepared["download_s"], 3)
        event("prepared", plan_total_bytes=prepared["plan"]["total_bytes"])
        t = time.monotonic()
        adapters["reference"] = build_adapter("reference", ref_spec, prepared, cfg, cfg.ref_api_key_env)
        adapters["candidate"] = build_adapter("candidate", cand_spec, prepared, cfg, cfg.cand_api_key_env)
        phases["load_s"] = round(time.monotonic() - t, 3)
        for role, a in adapters.items():
            profiles[role] = a.profile()
        if judge_spec is not None:
            from .inherit import JudgeGrader

            adapters["judge"] = build_adapter("judge", judge_spec, prepared, cfg, cfg.judge_api_key_env)
            profiles["judge"] = adapters["judge"].profile()
            gb.judge = JudgeGrader(adapters["judge"], meter, judge_digest=profiles["judge"]["digest"],
                                   cache_path=cfg.judge_cache)
        event("loaded")
        t = time.monotonic()
        if not cfg.skip_channels:
            for role in ("reference", "candidate"):
                channels[role] = detect_channels(adapters[role], meter)
        phases["channels_s"] = round(time.monotonic() - t, 3)
        coupling = plan_coupling(cfg.decode, adapters["reference"], adapters["candidate"],
                                 channels.get("reference"), channels.get("candidate"))
        event("channels", coupling=coupling["mode"])
        independent = coupling["independent_bounds"]
        consecutive = {"reference": 0, "candidate": 0}
        t_eval = time.monotonic()
        any_dev = False
        n_done = 0
        stop = False
        for li, look_n in enumerate(plan.looks):
            chunk = order[n_done:look_n]
            results = {}
            try:
                results["ref"] = adapters["reference"].generate([make_request(it, cfg.decode) for it in chunk], meter)
                results["cand"] = adapters["candidate"].generate([make_request(it, cfg.decode) for it in chunk], meter)
                if independent:
                    results["rep"] = adapters["reference"].generate(
                        [make_request(it, cfg.decode, replicate=1) for it in chunk], meter)
            except PartialCapReached as e:
                state["stopped_by"] = e.cap
                key = "ref" if "ref" not in results else ("cand" if "cand" not in results else "rep")
                results[key] = e.done
                stop = True
            except CapReached as e:
                state["stopped_by"] = e.cap
                stop = True
            k = min(len(results.get("ref", [])), len(results.get("cand", [])),
                    len(results.get("rep", [])) if independent else 10 ** 9)
            for j in range(k):
                it = chunk[j]
                r, c = results["ref"][j], results["cand"][j]
                for role, res in (("reference", r), ("candidate", c)):
                    consecutive[role] = consecutive[role] + 1 if res.finish_reason == "error" else 0
                    if consecutive[role] >= CONSECUTIVE_ERROR_LIMIT:
                        raise FidelityError(f"{role} failed {CONSECUTIVE_ERROR_LIMIT} consecutive requests; last error: "
                                            f"{scrub_text(res.error or '')}", code="ENDPOINT_FAILING")
                rg = gb.grade_reference(it, r.text)
                cg, how, spent = gb.grade_follower(it, r.text, rg, c.text, "candidate")
                flip = int(cg.correct) - int(rg.correct)
                rec = {"idx": n_done + j, "id": it["id"], "suite": it["suite"], "variant": it.get("variant"),
                       "max_new_tokens": it["max_new_tokens"], "ref": _side(r), "cand": _side(c),
                       "identical": r.text == c.text, "ref_grade": rg.to_dict(), "cand_grade": cg.to_dict(),
                       "cand_grade_how": how, "cand_grading_calls": spent, "flip": flip,
                       "first_divergence_token": _first_divergence(r.text, c.text, r.meta.get("tokens"),
                                                                   c.meta.get("tokens"))}
                counts["harmful"] += flip < 0
                counts["helpful"] += flip > 0
                if independent:
                    p = results["rep"][j]
                    pg, phow, pspent = gb.grade_follower(it, r.text, rg, p.text, "replicate")
                    rflip = int(pg.correct) - int(rg.correct)
                    counts["rr_harmful"] += rflip < 0
                    rec["rep"] = _side(p)
                    rec["rep_grade"] = pg.to_dict()
                    rec["rep_flip"] = rflip
                records.append(rec)
            n_done += k
            counts["n"] = n_done
            if stop or n_done < look_n:
                break
            ex = (counts["harmful"], n_done, counts["rr_harmful"], n_done) if independent else None
            dev, lb, exd = look_decision(stats, plan, li, n_done, counts["harmful"], excess_counts=ex)
            el = round(time.monotonic() - mono0, 3)
            looks_out.append({"look": li, "n": n_done, "harmful": counts["harmful"], "helpful": counts["helpful"],
                              "alpha_look": plan.alpha_per_look, "lower_bound_at_look_alpha": lb, "deviates": dev,
                              "elapsed_s": el, "excess": exd})
            event("look", look=li, n=n_done, harmful=counts["harmful"], deviates=dev)
            if dev and not any_dev:
                any_dev = True
                first_verdict = {"verdict": DEVIATES, "look": li, "n": n_done, "elapsed_s": el, "utc": _utc()}
                event("first_verdict", **first_verdict)
                if plan.early_stop:
                    break
            meter.check_wall()
        phases["eval_s"] = round(time.monotonic() - t_eval, 3)
        state["status"] = "completed"
    except RefusedError as e:
        state["status"] = "refused"
        state["refusal"] = {"code": e.code, "message": scrub_text(str(e)),
                            "plan": getattr(e, "plan", None).to_dict() if getattr(e, "plan", None) else None}
        event("refused", code=e.code)
    except CapReached as e:
        state["status"] = "completed"
        state["stopped_by"] = e.cap
    except FidelityError as e:
        state["status"] = "error"
        state["error"] = {"code": e.code, "message": scrub_text(str(e))}
        event("error", code=e.code)
    except Exception as e:  # never lose the report: record the failure, scrubbed
        import traceback

        state["status"] = "error"
        state["error"] = {"code": "INTERNAL_ERROR", "message": scrub_text(f"{type(e).__name__}: {e}"),
                          "traceback_tail": scrub_text(traceback.format_exc()[-1500:])}
        event("error", code="INTERNAL_ERROR")
    finally:
        for a in adapters.values():
            try:
                a.close()
            except Exception:
                pass
        leftover = hub.dir_bytes(work) if work.exists() else 0
        state["work_dir_bytes_at_end"] = leftover
        if created_work and not cfg.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)
        state["work_dir_removed"] = not work.exists()
    n = counts["n"]
    verdict = None
    bound = None
    if state["status"] == "completed":
        ex = (counts["harmful"], n, counts["rr_harmful"], n) if (coupling and coupling["independent_bounds"]) else None
        reached = n == plan.n_max and len(looks_out) == len(plan.looks)
        verdict, bound = final_decision(stats, plan, n, counts["harmful"], reached_n_max=reached,
                                        any_deviates=any(lk["deviates"] for lk in looks_out), excess_counts=ex)
        if first_verdict is None:
            first_verdict = {"verdict": verdict, "look": len(looks_out) - 1 if looks_out else None, "n": n,
                             "elapsed_s": round(time.monotonic() - mono0, 3), "utc": _utc(),
                             "final_look_reached": reached}
            event("first_verdict", **first_verdict)
    event("finish", status=state["status"], verdict=verdict)
    events.close()
    report = build_report(cfg=cfg, state=state, plan=plan, stats=stats, records=records, looks=looks_out,
                          verdict=verdict, bound=bound, first_verdict=first_verdict, profiles=profiles,
                          profile_diff=profile_diff(profiles["reference"], profiles["candidate"])
                          if "reference" in profiles and "candidate" in profiles else None,
                          channels=channels, coupling=coupling, meter=meter, gradebook=gb, prepared=prepared,
                          skipped=skipped, bank_items=items_all, phases=phases, started=started,
                          elapsed=time.monotonic() - mono0, bank_digest_value=bank_digest(items_all))
    if cfg.save_items:
        with JsonlWriter(out / "items.jsonl") as w:
            for rec in records:
                w.write(rec)
    write_json(out / "report.json", report)
    write_summary(out / "summary.md", report)
    return report
