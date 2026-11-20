# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""The comparison run: prepare both stacks, detect channels, couple the decoders,
evaluate items in a seeded random order with sequential looks, grade with exact
inheritance, measure each stack's noise floor, bound the chosen criterion, check
the power of the plan, meter the cost, and write the report.

Criteria (``criteria.py``): ``net_change`` (default) and ``harmful_excess`` use
the exact paired bound per capability; ``harmful_abs`` is the 0.1.0 rule and runs
the 0.1.0 loop unchanged (``_loop_harmful_abs``).

Run protocol of the paired criteria, per chunk of the seeded order: the
reference's first run R1 and the candidate's first run C1 in forward order, then
the reference's second run R2 and the candidate's second run C2 (where needed)
with the requests in reverse order.  The first chunk is the probe (the first
``floor_probe_items`` planned items), where both stacks run twice.  A second run
is the same request as the first under greedy decoding and under shared Gumbel
noise; without verified shared randomness every run gets its own derived seed,
so a replicate pair and a cross-stack pair are exchangeable when nothing
changed.  An errored output is retried once; an item whose reference output is
still an error is left out of the residuals (and the run cannot pass), a
candidate error counts as a wrong answer, and an errored second run is left out
of the floor.

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
from collections import Counter
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, criteria, determinism, domains, hub, noise, passive
from . import power as pw
from .adapters.base import MAX_SEED, DecodeConfig, GenRequest, GenResult, request_seed
from .banks import bank_digest, load_bank
from .channels import detect_channels
from .coupling import plan_coupling
from .errors import (BadInputError, CapReached, EndpointError, FidelityError, PartialCapReached, RefusedError,
                     UnsafeRepositoryError)
from .graders import Grade
from .inherit import GradeBook, load_judge_cache
from .jsonio import JsonlWriter, canonical, write_json
from .meter import Caps, CostMeter, PriceCard
from .plan import DEVIATES, Plan, final_decision, look_decision, look_schedule
from .refcache import RefCache
from .profile import diff as profile_diff
from .redact import register_secret, scrub_output_text, scrub_text
from .sandbox import namespaces_available
from .spec import StackSpec, parse_spec
from .stats import StatsBackend

CONSECUTIVE_ERROR_LIMIT = 8
DEFAULT_PROBE_ITEMS = 64


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
    criterion: str = "net_change"
    power: float = 0.9
    floor_probe_items: int | None = None
    replicate: str = "auto"
    plan_floor: float | None = None
    ref_cache: str | None = None
    stop_if_underpowered: bool = False

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
        d.choice(self.criterion, "criterion", criteria.CRITERIA)
        d.number(self.power, "power", gt=0.0, lt=1.0)
        d.optional_integer(self.floor_probe_items, "floor_probe_items", ge=0, le=1_000_000)
        d.choice(self.replicate, "replicate", ("auto", "always"))
        d.optional_number(self.plan_floor, "plan_floor", ge=0.0, le=1.0)
        d.optional_text(self.ref_cache, "ref_cache")
        d.boolean(self.stop_if_underpowered, "stop_if_underpowered")
        if self.criterion == criteria.HARMFUL_ABS:
            extra = [name for name, on in (("floor_probe_items", self.floor_probe_items not in (None, 0)),
                                           ("replicate", self.replicate != "auto"),
                                           ("plan_floor", self.plan_floor is not None),
                                           ("ref_cache", self.ref_cache is not None),
                                           ("stop_if_underpowered", self.stop_if_underpowered)) if on]
            if extra:
                raise BadInputError("criterion", f"harmful_abs keeps the 0.1.0 request pattern and rule; "
                                                 f"{', '.join(extra)} apply to net_change and harmful_excess only")
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
class _Counting:
    """Forwards to an adapter and counts the generation requests sent to it."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.requests = 0

    def generate(self, reqs, meter):
        self.requests += len(reqs)
        return self._inner.generate(reqs, meter)

    def prompt_logprobs(self, req, meter):
        self.requests += 1
        return self._inner.prompt_logprobs(req, meter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_indices(criterion: str, decode: DecodeConfig, coupling_mode: str | None) -> tuple[dict, bool, str]:
    """Replicate index (see ``make_request``) of each run, whether second runs share the
    first run's coupling, and the rule in words.

    Chosen so that, when nothing changed, a replicate pair (R1, R2) is exchangeable
    with the cross-stack pair (R1, C1): the same request under greedy decoding and
    under shared Gumbel noise; without verified shared randomness, a distinct derived
    seed for every run.  ``harmful_abs`` keeps the 0.1.0 rule."""
    if criterion == criteria.HARMFUL_ABS:
        return {"R1": 0, "C1": 0, "R2": 1, "C2": 1}, decode.mode == "greedy", \
            "0.1.0 rule: the candidate shares the reference's seed; the replicate has its own"
    if decode.mode == "greedy" or coupling_mode == "shared_gumbel":
        return {"R1": 0, "C1": 0, "R2": 0, "C2": 0}, True, \
            "a second run repeats the first run's request exactly (greedy decoding or shared Gumbel noise)"
    return {"R1": 0, "R2": 1, "C1": 2, "C2": 3}, False, \
        "every run has its own derived seed (no verified shared randomness between the stacks)"


def replicate_request_rule(decode: DecodeConfig, ref_adapter, cand_adapter) -> str:
    """'same' or 'fresh_seed': how the reference's second run is built (known before channels)."""
    if decode.mode == "greedy" or (ref_adapter.supports_shared_noise and cand_adapter.supports_shared_noise):
        return "same"
    return "fresh_seed"


def _from_cache(item_id: str, rec: dict) -> GenResult:
    s = rec["side"]
    return GenResult(item_id, s["text"], s["finish"], s["prompt_tokens"], s["completion_tokens"], s["usage_source"],
                     elapsed_s=float(s["elapsed_s"] or 0.0), prompt_sha256=s["prompt_sha256"],
                     meta={"tokens": rec["tokens"], "from_cache": True})


def _ok(res) -> bool:
    return res is not None and res.finish_reason != "error"


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
    crit = cfg.criterion
    paired = crit in criteria.PAIRED_CRITERIA
    items_all = load_bank(cfg.bank)
    skipped = {}
    ns_ok = namespaces_available() if any(it["grader"]["type"] == "unit_tests" for it in items_all) else True
    if any(it["grader"]["type"] == "unit_tests" for it in items_all) and not ns_ok and not cfg.sandbox_weak_ok:
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
    planned_caps = Counter(it["suite"] for it in order[:n_max])
    bank_caps = Counter(it["suite"] for it in items_all)
    probe_n = 0
    if paired:
        probe_n = min(n_max, DEFAULT_PROBE_ITEMS if cfg.floor_probe_items is None else cfg.floor_probe_items)

    # ---- outputs
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("events.jsonl", "items.jsonl", "report.json", "summary.md"):  # files of an earlier run
        (out / stale).unlink(missing_ok=True)
    events = JsonlWriter(out / "events.jsonl")

    def event(kind: str, **kw) -> None:
        events.write({"event": kind, "utc": _utc(), "elapsed_s": round(time.monotonic() - mono0, 3), **kw})

    event("start", tool=f"inferdrift {__version__}", criterion=crit)
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
    declared = {}
    cache = None
    channels_from_cache = False
    px: dict = {"criterion": crit, "probe_items": probe_n, "routing": None, "pass_errors": {}, "pre_gate_deviation":
                None, "gate": None, "excluded": 0, "complete": False, "run_counts": None, "stopped_underpowered": False,
                "rule": None, "same_coupling": None, "ref_all": False, "cand_all": False}
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
        adapters["reference"] = _Counting(build_adapter("reference", ref_spec, prepared, cfg, cfg.ref_api_key_env))
        adapters["candidate"] = _Counting(build_adapter("candidate", cand_spec, prepared, cfg, cfg.cand_api_key_env))
        phases["load_s"] = round(time.monotonic() - t, 3)
        for role, a in adapters.items():
            profiles[role] = a.profile()
        if judge_spec is not None:
            from .inherit import JudgeGrader

            adapters["judge"] = build_adapter("judge", judge_spec, prepared, cfg, cfg.judge_api_key_env)
            profiles["judge"] = adapters["judge"].profile()
            gb.judge = JudgeGrader(adapters["judge"], meter, judge_digest=profiles["judge"]["digest"],
                                   cache_path=cfg.judge_cache)
        for role in ("reference", "candidate"):
            declared[role] = determinism.declared(adapters[role], profiles[role])
        event("loaded")
        if paired and cfg.ref_cache:
            cache = RefCache(cfg.ref_cache, {
                "tool_version": __version__, "reference_profile_digest": profiles["reference"]["digest"],
                "reference_spec": ref_spec.display(),
                "decode": {"mode": "greedy"} if cfg.decode.mode == "greedy" else cfg.decode.to_dict(),
                "replicate_request": replicate_request_rule(cfg.decode, adapters["reference"], adapters["candidate"]),
                "bank_digest": bank_digest(items_all)}, items_all)
            cache.load()  # a corrupt entry raises CacheCorrupt (refused, exit 3)
        t = time.monotonic()
        if not cfg.skip_channels:
            for role in ("reference", "candidate"):
                if role == "reference" and cache is not None and cache.channels() is not None:
                    channels[role] = dict(cache.channels(), from_reference_cache=True)
                    channels_from_cache = True
                    continue
                channels[role] = detect_channels(adapters[role], meter)
        phases["channels_s"] = round(time.monotonic() - t, 3)
        coupling = plan_coupling(cfg.decode, adapters["reference"], adapters["candidate"],
                                 channels.get("reference"), channels.get("candidate"), criterion=crit)
        event("channels", coupling=coupling["mode"])
        independent = coupling["independent_bounds"]
        t_eval = time.monotonic()
        if not paired:
            # ---------------------------------------------------------- harmful_abs: the 0.1.0 loop, unchanged
            consecutive = {"reference": 0, "candidate": 0}
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
                            raise FidelityError(f"{role} failed {CONSECUTIVE_ERROR_LIMIT} consecutive requests; last "
                                                f"error: {scrub_text(res.error or '')}", code="ENDPOINT_FAILING")
                    rg = gb.grade_reference(it, r.text)
                    cg, how, spent = gb.grade_follower(it, r.text, rg, c.text, "candidate")
                    flip = int(cg.correct) - int(rg.correct)
                    rec = {"idx": n_done + j, "id": it["id"], "suite": it["suite"], "variant": it.get("variant"),
                           "max_new_tokens": it["max_new_tokens"], "ref": _side(r), "cand": _side(c),
                           "identical": r.text == c.text, "ref_grade": rg.to_dict(), "cand_grade": cg.to_dict(),
                           "cand_grade_how": how, "cand_grading_calls": spent, "flip": flip,
                           "first_divergence_token": _first_divergence(r.text, c.text, r.meta.get("tokens"),
                                                                       c.meta.get("tokens")),
                           "capability": it["suite"]}
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
        else:
            state["meter"] = meter
            first_verdict = _loop_paired(cfg, plan, order, n_max, probe_n, planned_caps, bank_caps, adapters, meter,
                                         gb, cache, channels, coupling, records, looks_out, state, px, event, mono0,
                                         ns_ok)
            counts["n"] = len(records)
            counts["harmful"] = sum(1 for r in records if r["flip"] < 0)
            counts["helpful"] = sum(1 for r in records if r["flip"] > 0)
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
        state.pop("meter", None)
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
    if state["status"] == "completed" and not paired:
        ex = (counts["harmful"], n, counts["rr_harmful"], n) if (coupling and coupling["independent_bounds"]) else None
        reached = n == plan.n_max and len(looks_out) == len(plan.looks)
        verdict, bound = final_decision(stats, plan, n, counts["harmful"], reached_n_max=reached,
                                        any_deviates=any(lk["deviates"] for lk in looks_out), excess_counts=ex)
        if first_verdict is None:
            first_verdict = {"verdict": verdict, "look": len(looks_out) - 1 if looks_out else None, "n": n,
                             "elapsed_s": round(time.monotonic() - mono0, 3), "utc": _utc(),
                             "final_look_reached": reached}
            event("first_verdict", **first_verdict)
    elif state["status"] == "completed":
        try:
            verdict, bound = _final_paired(cfg, plan, n_max, planned_caps, bank_caps, records, looks_out, state, px)
        except RefusedError as e:  # a missing second run found at the end: refused, never evaluated
            state["status"] = "refused"
            state["refusal"] = {"code": e.code, "message": scrub_text(str(e)), "plan": None}
            event("refused", code=e.code)
            verdict, bound = None, None
        if verdict is not None and first_verdict is None:
            first_verdict = {"verdict": verdict, "look": len(looks_out) - 1 if looks_out else None, "n": n,
                             "elapsed_s": round(time.monotonic() - mono0, 3), "utc": _utc(),
                             "final_look_reached": bool(looks_out) and looks_out[-1]["n"] == plan.n_max}
            event("first_verdict", **first_verdict)
    if cache is not None and state["status"] == "completed":
        cache.write(None if channels_from_cache else channels.get("reference"), _utc())
    event("finish", status=state["status"], verdict=verdict)
    events.close()
    ref_requests = adapters["reference"].requests if isinstance(adapters.get("reference"), _Counting) else 0
    extra = {"paired": px if paired else None, "declared": declared, "adapters_batching": {
        role: bool(getattr(a, "max_concurrency", 1) > 1) for role, a in adapters.items() if role != "judge"},
        "reference_cache": cache.report(ref_requests) if cache is not None else
        {"enabled": False, "reference_generations": ref_requests},
        "reference_requests": ref_requests}
    report = build_report(cfg=cfg, state=state, plan=plan, stats=stats, records=records, looks=looks_out,
                          verdict=verdict, bound=bound, first_verdict=first_verdict, profiles=profiles,
                          profile_diff=profile_diff(profiles["reference"], profiles["candidate"])
                          if "reference" in profiles and "candidate" in profiles else None,
                          channels=channels, coupling=coupling, meter=meter, gradebook=gb, prepared=prepared,
                          skipped=skipped, bank_items=items_all, phases=phases, started=started,
                          elapsed=time.monotonic() - mono0, bank_digest_value=bank_digest(items_all), extra=extra)
    if cfg.save_items:
        with JsonlWriter(out / "items.jsonl") as w:
            for rec in records:
                w.write(rec)
    write_json(out / "report.json", report)
    write_summary(out / "summary.md", report)
    return report


# --------------------------------------------------------------------------- paired criteria
def _loop_paired(cfg, plan, order, n_max, probe_n, planned_caps, bank_caps, adapters, meter, gb, cache, channels,
                 coupling, records, looks_out, state, px, event, mono0, ns_ok) -> dict | None:
    """The evaluation loop of ``net_change`` and ``harmful_excess``.  Fills ``records``,
    ``looks_out`` and ``px``; returns the first verdict the power gate allowed (or None)."""
    crit = cfg.criterion
    alpha = cfg.alpha
    idx, same_coupling, rule = run_indices(crit, cfg.decode, coupling["mode"])
    independent = coupling["independent_bounds"]
    px["rule"] = rule
    px["same_coupling"] = same_coupling
    reasons = {"reference": [], "candidate": []}
    ref_all = cand_all = False
    if crit == criteria.HARMFUL_EXCESS:
        ref_all = True
        reasons["reference"].append("harmful_excess needs the reference's second run on every item")
    if independent:
        ref_all = True
        reasons["reference"].append("sampling without verified shared randomness (as 0.1.0)")
    if cfg.replicate == "always":
        ref_all = cand_all = True
        reasons["reference"].append("--replicate always")
        reasons["candidate"].append("--replicate always")
    ch_noisy = {}
    for role in ("reference", "candidate"):
        rate = (channels.get(role) or {}).get("greedy_repeat_identity_rate_without_errors")
        ch_noisy[role] = rate is not None and rate < 1.0
    if same_coupling:
        if ch_noisy["reference"] or ch_noisy["candidate"]:
            which = [r for r in ("reference", "candidate") if ch_noisy[r]]
            ref_all = True
            reasons["reference"].append(f"the channel probe saw repeated greedy outputs differ on: {', '.join(which)}")
        if ch_noisy["candidate"]:
            cand_all = True
            reasons["candidate"].append("the channel probe saw the candidate's repeated greedy outputs differ")
    pass_err = {k: {"requests": 0, "errored": 0, "retried": 0, "errored_after_retry": 0, "cached": 0}
                for k in ("R1", "C1", "R2", "C2")}
    px["pass_errors"] = pass_err
    noisy = {"reference": None, "candidate": None}
    identity_cache: dict = {}

    def grade_identity(it: dict) -> dict:
        t = it["grader"]["type"]
        if t == "judge":
            return {"judge": gb.judge.judge_digest, "template": gb.judge.template_digest} if gb.judge else {"judge": None}
        if t == "unit_tests":
            if "sandbox" not in identity_cache:
                identity_cache["sandbox"] = "namespaces" if ns_ok else (
                    "python_socket_blocking" if cfg.sandbox_weak_ok else "none")
            return {"sandbox": identity_cache["sandbox"]}
        return {}

    def run_pass(role: str, chunk: list, kind: str, reverse: bool):
        n = len(chunk)
        res: list = [None] * n
        cached: dict = {}
        if cache is not None and role == "reference":
            crep = 0 if kind == "R1" else 1
            for j, it in enumerate(chunk):
                rec = cache.get(it["id"], crep)
                if rec is not None:
                    res[j] = _from_cache(it["id"], rec)
                    cached[j] = rec
            pass_err[kind]["cached"] += len(cached)
        positions = [j for j in range(n) if res[j] is None]
        if reverse:
            positions = positions[::-1]
        cap = None
        what = {"R1": "first", "C1": "first", "R2": "second", "C2": "second"}[kind]
        code = "RESULTS_INCOMPLETE" if kind in ("R1", "C1") else "REPLICATE_INCOMPLETE"

        def send(pos: list) -> list:
            nonlocal cap
            reqs = [make_request(chunk[j], cfg.decode, idx[kind]) for j in pos]
            try:
                got = adapters[role].generate(reqs, meter)
            except PartialCapReached as e:
                got, cap = list(e.done), e.cap
            except CapReached as e:
                got, cap = [], e.cap
            if cap is None and len(got) != len(reqs):
                raise RefusedError(f"the {role} adapter returned {len(got)} outputs for {len(reqs)} requests in its "
                                   f"{what} run, without a cap; the run is refused rather than evaluated on a silent "
                                   f"subset", code=code)
            return got

        if positions:
            got = send(positions)
            pass_err[kind]["requests"] += len(got)
            for j, r in zip(positions, got):
                res[j] = r
            bad = [j for j in positions[:len(got)] if res[j].finish_reason == "error"]
            pass_err[kind]["errored"] += len(bad)
            if bad and cap is None:  # an errored output is retried once
                got2 = send(bad)
                pass_err[kind]["retried"] += len(got2)
                for j, r in zip(bad, got2):
                    res[j] = r
            pass_err[kind]["errored_after_retry"] += sum(1 for j in positions if res[j] is not None
                                                         and res[j].finish_reason == "error")
        return res, cached, cap

    consec = {"R1": 0, "C1": 0, "R2": 0, "C2": 0}
    role_of = {"R1": "reference", "C1": "candidate", "R2": "reference", "C2": "candidate"}
    m = len(planned_caps)
    K = len(plan.looks)
    alpha_look = alpha / (m * K)
    resid_key = "residual_net" if crit == criteria.NET_CHANGE else "residual_excess"
    run_counts = {c: criteria.Counts() for c in planned_caps}
    folded = {"n": 0, "excluded": 0}
    px["run_counts"] = run_counts

    def fold_to(limit: int) -> None:
        while folded["n"] < limit:
            rec = records[folded["n"]]
            r = rec[resid_key]
            if r is None:
                folded["excluded"] += 1
            else:
                run_counts[rec["capability"]].add(r)
            folded["n"] += 1

    def gate_now(final: bool) -> dict:
        return evaluate_gate(cfg, records, planned_caps, bank_caps, px["ref_all"], px["cand_all"], final=final)

    first_verdict = None
    boundaries = ([probe_n] if probe_n > 0 else []) + [lk for lk in plan.looks if lk > probe_n]
    pending = list(enumerate(plan.looks))
    px["ref_all"], px["cand_all"] = ref_all, cand_all
    px["noisy"] = noisy
    px["channel_noisy"] = ch_noisy
    px["routing"] = {"reference_every_item": ref_all, "candidate_every_item": cand_all, "reasons": reasons,
                     "same_coupling_replicates": same_coupling, "rule": rule}
    px["n_done"] = 0
    try:
        first_verdict = _paired_chunks(cfg, plan, order, probe_n, boundaries, pending, run_pass, need_state=px,
                                       records=records, looks_out=looks_out, state=state, event=event, mono0=mono0,
                                       gb=gb, cache=cache, grade_identity=grade_identity, consec=consec,
                                       role_of=role_of, run_counts=run_counts, fold_to=fold_to, folded=folded,
                                       alpha_look=alpha_look, same_coupling=same_coupling, noisy=noisy,
                                       reasons=reasons, gate_now=gate_now)
    finally:
        fold_to(px["n_done"])
        px["excluded"] = folded["excluded"]
        px["routing"]["reference_every_item"] = px["ref_all"]
        px["routing"]["candidate_every_item"] = px["cand_all"]
    return first_verdict


def _paired_chunks(cfg, plan, order, probe_n, boundaries, pending, run_pass, *, need_state, records, looks_out, state,
                   event, mono0, gb, cache, grade_identity, consec, role_of, run_counts, fold_to, folded, alpha_look,
                   same_coupling, noisy, reasons, gate_now) -> dict | None:
    px = need_state
    alpha, tau = cfg.alpha, cfg.tau
    first_verdict = None
    n_done = 0
    stop = False
    for b_end in boundaries:
        chunk = order[n_done:b_end]
        is_probe = n_done == 0 and probe_n > 0
        need = {"R1": True, "C1": True, "R2": is_probe or px["ref_all"], "C2": is_probe or px["cand_all"]}
        outs = {}
        cached = {}
        for kind in ("R1", "C1", "R2", "C2"):
            if not need[kind]:
                continue
            res, cset, cap = run_pass(role_of[kind], chunk, kind, reverse=kind in ("R2", "C2"))
            outs[kind], cached[kind] = res, cset
            if cap is not None:
                state["stopped_by"] = cap
                stop = True
                break
        k = 0
        while k < len(chunk) and all(kind in outs and outs[kind][k] is not None for kind in need if need[kind]):
            k += 1
        for kind in ("R1", "C1", "R2", "C2"):
            if kind not in outs:
                continue
            for j in range(k):
                r = outs[kind][j]
                consec[kind] = consec[kind] + 1 if r.finish_reason == "error" else 0
                if consec[kind] >= CONSECUTIVE_ERROR_LIMIT:
                    raise FidelityError(f"{role_of[kind]} failed {CONSECUTIVE_ERROR_LIMIT} consecutive requests "
                                        f"(after one retry each) in its {'first' if kind[1] == '1' else 'second'} "
                                        f"run; last error: {scrub_text(r.error or '')}", code="ENDPOINT_FAILING")
        for j in range(k):
            it = chunk[j]
            r1, c1 = outs["R1"][j], outs["C1"][j]
            from_cache = j in cached["R1"]
            if from_cache:
                crec = cached["R1"][j]
                ident = grade_identity(it)
                if canonical(crec["grade_identity"]) == canonical(ident):
                    rg = gb.grade_reference_cached(it, r1.text, Grade(**crec["grade"]))
                else:
                    gb.calls["reference_cache_regraded"] += 1
                    rg = gb.grade_reference(it, r1.text)
            else:
                rg = gb.grade_reference(it, r1.text)
            cg, how, spent = gb.grade_follower(it, r1.text, rg, c1.text, "candidate")
            flip = int(cg.correct) - int(rg.correct)
            rec = {"idx": n_done + j, "id": it["id"], "suite": it["suite"], "variant": it.get("variant"),
                   "max_new_tokens": it["max_new_tokens"], "ref": _side(r1), "cand": _side(c1),
                   "identical": r1.text == c1.text, "ref_grade": rg.to_dict(), "cand_grade": cg.to_dict(),
                   "cand_grade_how": how, "cand_grading_calls": spent, "flip": flip,
                   "first_divergence_token": _first_divergence(r1.text, c1.text, r1.meta.get("tokens"),
                                                               c1.meta.get("tokens")),
                   "capability": it["suite"], "probe": is_probe, "ref_source": "cache" if from_cache else "run"}
            pg = None
            if "R2" in outs:
                p = outs["R2"][j]
                pg, _phow, _pspent = gb.grade_follower(it, r1.text, rg, p.text, "replicate")
                rec["rep"] = _side(p)
                rec["rep_grade"] = pg.to_dict()
                rec["rep_flip"] = int(pg.correct) - int(rg.correct)
                rec["rep_source"] = "cache" if j in cached["R2"] else "run"
            if "C2" in outs:
                q = outs["C2"][j]
                qg, _qhow, _qspent = gb.grade_follower(it, c1.text, cg, q.text, "candidate_replicate")
                rec["cand_rep"] = _side(q)
                rec["cand_rep_grade"] = qg.to_dict()
                rec["cand_rep_flip"] = int(qg.correct) - int(cg.correct)
            r1_ok = _ok(r1)
            rec["residual_net"] = criteria.residual_net(rg.correct, cg.correct) if r1_ok else None
            rec["residual_excess"] = criteria.residual_excess(rg.correct, cg.correct, pg.correct) \
                if (r1_ok and pg is not None and _ok(outs["R2"][j])) else None
            rec["rep_required"] = need["R2"]
            rec["cand_rep_required"] = need["C2"]
            records.append(rec)
            if cache is not None:
                ident = grade_identity(it)
                if not from_cache:
                    cache.put(it, 0, rec["ref"], rec["ref_grade"], ident, r1.meta.get("tokens"))
                if "R2" in outs and j not in cached["R2"]:
                    cache.put(it, 1, rec["rep"], rec["rep_grade"], ident, outs["R2"][j].meta.get("tokens"))
        n_done += k
        px["n_done"] = n_done
        if is_probe and k == len(chunk):
            st_r = noise.pair_stats(records, "ref", "rep", "ref_grade", "rep_grade", alpha)
            st_c = noise.pair_stats(records, "cand", "cand_rep", "cand_grade", "cand_rep_grade", alpha)
            noisy["reference"] = st_r["items"] > 0 and st_r["byte_identical"] < st_r["items"]
            noisy["candidate"] = st_c["items"] > 0 and st_c["byte_identical"] < st_c["items"]
            if same_coupling:
                if noisy["reference"] or noisy["candidate"]:
                    which = [r for r in ("reference", "candidate") if noisy[r]]
                    if not px["ref_all"]:
                        px["ref_all"] = True
                    reasons["reference"].append(f"the probe saw outputs of {', '.join(which)} differ between two runs "
                                                f"of the same requests")
                if noisy["candidate"] and not px["cand_all"]:
                    px["cand_all"] = True
                    reasons["candidate"].append("the probe saw the candidate's outputs differ between two runs")
            event("probe", items=k, noisy=noisy, reference_every_item=px["ref_all"],
                  candidate_every_item=px["cand_all"])
            if cfg.stop_if_underpowered and not stop:
                g = gate_now(False)
                if g["underpowered"]:
                    px["stopped_underpowered"] = True
                    stop = True
        # ---- looks reached by this chunk
        while pending and pending[0][1] <= n_done and not px["stopped_underpowered"]:
            li, look_n = pending.pop(0)
            fold_to(look_n)
            dev, lbs = criteria.look_check(run_counts, alpha_look, tau)
            seen = records[:look_n]
            el = round(time.monotonic() - mono0, 3)
            allowed = False
            gate_info = None
            if dev:
                gate_info = gate_now(False)
                allowed = not gate_info["underpowered"]
                if px["pre_gate_deviation"] is None:
                    px["pre_gate_deviation"] = {
                        "look": li, "n": look_n, "alpha_look": alpha_look,
                        "capabilities": sorted(c for c, v in lbs.items() if v is not None and v > tau),
                        "lower_bounds": {c: v for c, v in lbs.items() if v is not None and v > tau},
                        "allowed_by_power_gate": allowed}
            lbv = [v for v in lbs.values() if v is not None]
            looks_out.append({"look": li, "n": look_n, "harmful": sum(1 for r in seen if r["flip"] < 0),
                              "helpful": sum(1 for r in seen if r["flip"] > 0), "alpha_look": alpha_look,
                              "lower_bound_at_look_alpha": max(lbv) if lbv else None, "deviates": dev,
                              "deviates_allowed_by_power_gate": allowed if dev else None, "elapsed_s": el,
                              "excess": None,
                              "per_capability": {c: dict(run_counts[c].to_dict(), lower=lbs.get(c))
                                                 for c in sorted(run_counts)},
                              "excluded_items": folded["excluded"],
                              "power_at_planned": gate_info["power_at_planned"] if gate_info else None})
            event("look", look=li, n=look_n, deviates=dev, allowed=allowed)
            if allowed and first_verdict is None:
                first_verdict = {"verdict": criteria.DEVIATES, "look": li, "n": look_n, "elapsed_s": el, "utc": _utc()}
                event("first_verdict", **first_verdict)
                if plan.early_stop:
                    stop = True
                    break
        if stop or n_done < b_end:
            break
        state["meter"].check_wall()
    return first_verdict


def evaluate_gate(cfg, records: list[dict], planned_caps: Counter, bank_caps: Counter, ref_all: bool,
                  cand_all: bool, *, final: bool) -> dict:
    """The power gate: is the plan powered at the floor known so far?  With ``final``,
    also the items needed and the smallest tolerance the planned items support."""
    crit, alpha, tau, target = cfg.criterion, cfg.alpha, cfg.tau, cfg.power
    st_r = noise.pair_stats(records, "ref", "rep", "ref_grade", "rep_grade", alpha)
    st_c = noise.pair_stats(records, "cand", "cand_rep", "cand_grade", "cand_rep_grade", alpha)
    caps_all = sorted(set(bank_caps) | set(planned_caps))
    fl = noise.comparison_floor(st_r, st_c, ref_all, cand_all, caps_all, alpha)
    contradicted = False
    if cfg.plan_floor is not None:
        floors = {c: cfg.plan_floor for c in caps_all}
        source = "supplied"
        if fl["measured"] and fl["bounds"]["lower"] > cfg.plan_floor:
            floors = dict(fl["per_capability"])
            source = "measured"
            contradicted = True
    elif fl["measured"]:
        floors = dict(fl["per_capability"])
        source = "measured"
    else:
        floors = {c: 0.0 for c in caps_all}
        source = "unmeasured"
    models = {c: pw.model_for(crit, floors[c]) for c in caps_all}
    planned = {c: planned_caps[c] for c in sorted(planned_caps)}
    p_planned = pw.power_total(planned, {c: models[c] for c in planned}, alpha, tau)
    per_cap_power = {c: pw.pass_probability(planned[c], models[c], alpha, tau) for c in planned}
    underpowered = source == "unmeasured" or p_planned < target
    out = {"floor_source": source, "floors": floors, "supplied_floor_contradicted": contradicted,
           "power_at_planned": p_planned, "power_per_capability": per_cap_power, "underpowered": underpowered,
           "floor": fl, "stats": {"reference": st_r, "candidate": st_c}}
    if not final:
        return out
    weights = {c: bank_caps[c] for c in caps_all if bank_caps[c] > 0} or dict(planned)
    n_planned = sum(planned.values())
    if underpowered and source != "unmeasured":
        need = pw.items_needed(weights, {c: models[c] for c in weights}, alpha, tau, target, above=n_planned,
                               cap_per_capability=pw.COMPARE_SEARCH_CAP)
    elif source == "unmeasured":
        zero = {c: pw.model_for(crit, 0.0) for c in weights}
        lb = pw.items_needed(weights, zero, alpha, tau, target, above=0, cap_per_capability=pw.COMPARE_SEARCH_CAP)
        need = {"items_needed": None, "per_capability": None, "exceeds": None,
                "lower_bound": lb["items_needed"],
                "note": "the floor was not measured; the zero-floor size is only a lower bound and no verdict is "
                        "issued on it"}
    else:
        need = {"items_needed": None, "per_capability": None, "exceeds": None}
    short = []
    if need.get("per_capability"):
        short = [{"capability": c, "planned": planned.get(c, 0), "needed": need["per_capability"][c]}
                 for c in sorted(need["per_capability"]) if need["per_capability"][c] > planned.get(c, 0)]
    out["items_needed"] = need
    out["short_capabilities"] = short
    out["smallest_tau_at_planned"] = pw.smallest_tau(planned, {c: models[c] for c in planned}, alpha, target)
    return out


def _final_paired(cfg, plan, n_max, planned_caps, bank_caps, records, looks_out, state, px) -> tuple:
    """Verdict of a completed paired run, and the binding capability's bound."""
    missing = [r["id"] for r in records if (r.get("rep_required") and "rep" not in r)
               or (r.get("cand_rep_required") and "cand_rep" not in r)]
    if missing:
        raise RefusedError(f"{len(missing)} item(s) lack a second run that the replication plan required "
                           f"(first: {missing[0]}); the run is refused", code="REPLICATE_INCOMPLETE")
    run_counts = px["run_counts"] or {c: criteria.Counts() for c in planned_caps}
    px["run_counts"] = run_counts
    px.setdefault("n_done", 0)
    complete = px.get("n_done") == n_max and px["excluded"] == 0 and not state.get("stopped_by") \
        and not px["stopped_underpowered"]
    px["complete"] = complete
    all_within, ubs = criteria.final_check(run_counts, cfg.alpha, cfg.tau)
    gate = evaluate_gate(cfg, records, planned_caps, bank_caps, px["ref_all"], px["cand_all"], final=True)
    px["gate"] = gate
    any_dev = any(lk["deviates"] for lk in looks_out)
    verdict = criteria.decide(underpowered=gate["underpowered"], deviates=any_dev, complete=complete,
                              all_within=all_within)
    if px["n_done"] == 0 and state.get("stopped_by"):
        verdict = criteria.INCONCLUSIVE  # a cap stopped the run before any item: no statement about power either
    if verdict == criteria.DEVIATES and px["pre_gate_deviation"]:
        cap = max(px["pre_gate_deviation"]["lower_bounds"], key=px["pre_gate_deviation"]["lower_bounds"].get)
    else:
        cap = max(ubs, key=lambda c: (ubs[c] if ubs[c] is not None else 2.0)) if ubs else None
    bound = None
    if cap is not None:
        bd = criteria.bound_dict(run_counts[cap], cfg.alpha)
        bd["capability"] = cap
        bound = bd
    px["final"] = {"all_within": all_within, "upper_bounds": ubs, "binding_capability": cap}
    return verdict, bound
