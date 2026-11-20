# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Local metered stand-in endpoint (OpenAI-compatible, with dialect emulation).

A small HTTP server for testing the cost meter and the adapters without provider
spend.  It prices every request per token, keeps a server-side ledger (the
independent record a cap test compares the client meter with), and can emulate
the metadata and native routes of llama.cpp, vLLM and SGLang.

Backends
    synthetic       deterministic text derived from a hash of the request
    replay:FILE     JSONL lines {"key": request_key(...), "text": ...}; misses fall
                    back to synthetic text and are counted.  A line may also carry
                    "alt" and "flip" (a number in [0, 1]): each request for that key
                    then serves "alt" with probability "flip" (a noisy engine)
    hf:SPEC         a transformers model (needs the hf extra), e.g. hf:Qwen/Qwen3-0.6B

Fault injection (for adversarial tests): ``ignore_seed``, ``no_logprobs``,
``truncate_frac`` (cut outputs and still report a normal stop),
``leak_key_in_errors`` (echo the caller's credential in error bodies),
``nondeterministic`` (perturb greedy outputs with this probability),
``fail_every`` (every k-th generation request returns HTTP 500),
``bad_usage`` (report malformed token usage - negative, float, string, boolean,
beyond the exact range or NaN - while the server-side ledger stays truthful).

Noise model of replay lines with "alt" and "flip": the j-th greedy request for a
key serves "alt" when u < flip, with u = (first 8 bytes of
sha256(f"{noise_seed}|{key}|{j}")) / 2**64.  A sampled request that carries a
seed (and the seed is honoured) draws u from sha256(f"{noise_seed}|{key}|s{seed}")
instead, so the same seed replays the same answer and another seed draws again.
The draws do not depend on thread interleaving (requests of one key are never
concurrent in a comparison).  Two stand-ins with different noise seeds are
independent engines; with the same noise seed they are the same engine, byte for
byte.

Closed domains: prices are finite and >= 0, fractions lie in their
intervals, counts are non-negative integers; request bodies are parsed strictly
and a request whose numeric fields fall outside their domain gets HTTP 400 and is
not billed.  Responses are strict JSON: a non-finite log-probability is sent as
null (as llama.cpp does), never as NaN or Infinity.

Usage (stand-in counts tokens as ceil(UTF-8 bytes / 4) unless the backend has
a tokenizer)::

    inferdrift standin --port 8099 --price-in 10 --price-out 30 --api-key-env MY_KEY
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import domains
from .errors import BadInputError, FidelityError
from .jsonio import StrictJSONError, canonical, loads_strict, read_jsonl_strict, sha256_text

BAD_USAGE = ("negative", "float", "string", "bool", "huge", "nan")
_MAX_REQ_TOKENS = 1 << 20


def _finite_or_none(obj):
    """Strict-JSON form of a response: non-finite floats become null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    return obj


class _BadRequest(Exception):
    pass


def _req_int(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _BadRequest(f"{key} must be an integer in [{lo}, {hi}]")
    return v


def _req_temperature(body: dict, key: str = "temperature") -> float:
    v = body.get(key)
    if v is None:
        return 0.0
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1e4:
        raise _BadRequest(f"{key} must be a finite number in [0, 10000]")
    return float(v)


def _req_seed(v):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= domains.MAX_EXACT_INT:
        raise _BadRequest(f"seed must be an integer in [0, {domains.MAX_EXACT_INT}]")
    return v

WORDS = ("alpha bravo cedar delta ember fjord grove harbor island jade kelp lumen maple nectar onyx pebble quartz "
         "raven sable tundra umber velvet willow xenon yarrow zephyr").split()


def request_key(messages=None, tools=None, prompt=None) -> str:
    if prompt is not None:
        return sha256_text("prompt|" + (prompt if isinstance(prompt, str) else canonical(prompt)))
    return sha256_text(canonical({"m": messages, "t": tools or None}))


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


class StandinState:
    def __init__(self, *, backend: str = "synthetic", price_in: float = 0.0, price_out: float = 0.0,
                 api_key: str | None = None, ledger_path: str | None = None, dialect: str = "openai",
                 model_name: str = "standin-model", ignore_seed: bool = False, no_logprobs: bool = False,
                 truncate_frac: float | None = None, leak_key_in_errors: bool = False, nondeterministic: float = 0.0,
                 fail_every: int = 0, work_dir: str | None = None, bad_usage: str | None = None,
                 noise_seed: int = 0, extra_routes: dict | None = None) -> None:
        self.backend = domains.text(backend, "backend")
        self.price_in = domains.number(price_in, "price_in", ge=0.0, le=1e6)
        self.price_out = domains.number(price_out, "price_out", ge=0.0, le=1e6)
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise BadInputError("api_key", "must be a non-empty string or None")
        self.api_key = api_key
        self.dialect = domains.choice(dialect, "dialect", ("openai", "llamacpp", "vllm", "sglang"))
        self.model_name = domains.text(model_name, "model_name", max_len=256)
        self.ignore_seed = domains.boolean(ignore_seed, "ignore_seed")
        self.no_logprobs = domains.boolean(no_logprobs, "no_logprobs")
        self.truncate_frac = domains.optional_number(truncate_frac, "truncate_frac", gt=0.0, le=1.0)
        self.leak = domains.boolean(leak_key_in_errors, "leak_key_in_errors")
        self.nondet = domains.number(nondeterministic, "nondeterministic", ge=0.0, le=1.0)
        self.fail_every = domains.integer(fail_every, "fail_every", ge=0, le=1 << 20)
        self.bad_usage = None if bad_usage is None else domains.choice(bad_usage, "bad_usage", BAD_USAGE)
        self.noise_seed = domains.integer(noise_seed, "noise_seed", ge=0, le=domains.MAX_EXACT_INT)
        if extra_routes is not None and (not isinstance(extra_routes, dict) or not all(
                isinstance(k, str) and k.startswith("/") and isinstance(v, dict) for k, v in extra_routes.items())):
            raise BadInputError("extra_routes", "must map paths starting with / to JSON objects")
        self.extra_routes = dict(extra_routes or {})
        self.noise: dict[str, tuple[str, float]] = {}
        self.key_requests: dict[str, int] = {}
        self.lock = threading.Lock()
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0, "errors": 0,
                       "replay_misses": 0}
        self.ledger_path = ledger_path
        self.counter = 0
        self.replay: dict[str, str] = {}
        self.hf = None
        self.hf_lock = threading.Lock()
        if backend.startswith("replay:"):
            path = backend[len("replay:"):]
            try:
                rows = read_jsonl_strict(path)
            except StrictJSONError as e:
                raise BadInputError(f"replay file {path}", f"is not strict JSON ({e})") from None
            for no, e in rows:
                where = f"replay file {path}:{no}"
                if not isinstance(e, dict) or set(e) not in ({"key", "text"}, {"key", "text", "alt", "flip"}) \
                        or not isinstance(e["key"], str) or not isinstance(e["text"], str):
                    raise BadInputError(where, "each line must be {\"key\": str, \"text\": str} or also carry "
                                               "\"alt\": str and \"flip\": a number in [0, 1]")
                self.replay[e["key"]] = e["text"]
                if "alt" in e:
                    if not isinstance(e["alt"], str):
                        raise BadInputError(f"{where}.alt", "must be a string", e["alt"])
                    self.noise[e["key"]] = (e["alt"], domains.number(e["flip"], f"{where}.flip", ge=0.0, le=1.0))
        elif backend.startswith("hf:"):
            self._load_hf(backend, work_dir)

    # ---------------------------------------------------------------- hf backend
    def _load_hf(self, spec_text: str, work_dir: str | None) -> None:
        import tempfile
        from pathlib import Path

        from .adapters.hf import HFAdapter
        from .compare import prepare_stacks
        from .spec import parse_spec

        spec = parse_spec(spec_text)
        wd = Path(work_dir or tempfile.mkdtemp(prefix="inferdrift-standin-"))
        prepared = prepare_stacks([("candidate", spec)], wd, 6.0)
        src = prepared["sources"]["candidate"]
        prof = dict(src["passive"])
        prof["commit"] = src.get("commit")
        self.hf = HFAdapter(spec, "candidate", src["local_dir"], prof)
        self.model_name = spec.repo or os.path.basename(str(spec.local_path))

    # ---------------------------------------------------------------- generation
    def _synthetic(self, key: str, max_tokens: int, sample_salt: str | None) -> str:
        h = hashlib.sha256((key + "|" + (sample_salt or "")).encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        n = 3 + rng.randrange(40)
        text = " ".join(rng.choice(WORDS) for _ in range(n)) + "."
        return text

    def generate(self, key: str, *, messages=None, tools=None, template_kwargs=None, max_tokens: int,
                 temperature: float, seed, prompt=None) -> tuple[str, int, int, str]:
        """Return (text, prompt_tokens, completion_tokens, finish)."""
        sampled = temperature and temperature > 0
        if self.hf is not None:
            from .adapters.base import DecodeConfig, GenRequest
            from .meter import Caps, CostMeter

            dec = DecodeConfig("sample", temperature=float(temperature), seed=int(seed or 0)) if sampled \
                else DecodeConfig("greedy")
            if sampled and (self.ignore_seed or seed is None):
                dec = DecodeConfig("sample", temperature=float(temperature), seed=random.randrange(1 << 30))
            req = GenRequest("standin", messages or [{"role": "user", "content": str(prompt)}], tools,
                             dict(template_kwargs or {}), int(max_tokens), dec, coupling_key=key)
            with self.hf_lock:  # one model, many request threads
                res = self.hf.generate([req], CostMeter(Caps()))[0]
            text, pt, ct, fin = res.text, res.prompt_tokens, res.completion_tokens, res.finish_reason
        else:
            if key in self.replay:
                text = self.replay[key]
                if key in self.noise:
                    alt, flip = self.noise[key]
                    if self._noise_draw(key, sampled, seed) < flip:
                        text = alt
            else:
                with self.lock:
                    if self.replay:
                        self.totals["replay_misses"] += 1
                salt = None
                if sampled:
                    salt = f"r{random.random()}" if (self.ignore_seed or seed is None) else f"s{seed}"
                text = self._synthetic(key, max_tokens, salt)
            if prompt is not None:
                pt = approx_tokens(prompt if isinstance(prompt, str) else canonical(prompt))
            else:
                pt = approx_tokens("".join(m.get("content") or "" for m in messages or []) +
                                   (canonical(tools) if tools else "")) + 8 * len(messages or [])
            fin = "stop"
            if approx_tokens(text) > max_tokens:
                b = text.encode("utf-8")[: max_tokens * 4]
                text = b.decode("utf-8", errors="ignore")
                fin = "length"
            ct = approx_tokens(text) if text else 0
        if self.nondet and not sampled and random.random() < self.nondet:
            text = text + " " + random.choice(WORDS)
            ct += 1
            fin = "stop"
        if self.truncate_frac is not None and text:
            text = text[: max(1, int(len(text) * self.truncate_frac))]
            if self.hf is None:
                ct = approx_tokens(text)
            fin = "stop"
        return text, pt, ct, fin

    def _noise_draw(self, key: str, sampled: bool, seed) -> float:
        if sampled:
            if self.ignore_seed or seed is None:
                return random.random()
            tag = f"{self.noise_seed}|{key}|s{seed}"
        else:
            with self.lock:
                j = self.key_requests.get(key, 0)
                self.key_requests[key] = j + 1
            tag = f"{self.noise_seed}|{key}|{j}"
        return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big") / 2.0 ** 64

    def bill(self, route: str, pt: int, ct: int, status: int = 200) -> float:
        usd = (pt * self.price_in + ct * self.price_out) / 1e6
        with self.lock:
            self.counter += 1
            self.totals["requests"] += 1
            self.totals["prompt_tokens"] += pt
            self.totals["completion_tokens"] += ct
            self.totals["usd"] += usd
            if self.ledger_path:
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(canonical({"n": self.counter, "route": route, "prompt_tokens": pt,
                                       "completion_tokens": ct, "usd": usd, "status": status}) + "\n")
        return usd

    def usage_value(self, n: int):
        """The token count as reported to the client (malformed under ``bad_usage``)."""
        bad = self.bad_usage
        if bad is None:
            return n
        return {"negative": -1000 * (n + 1), "float": float(n) + 0.5, "string": str(n), "bool": True,
                "huge": (1 << 60) + n, "nan": float("nan")}[bad]

    def fake_logprobs(self, text: str, k: int) -> list:
        out = []
        for i, w in enumerate(text.split()):
            h = int(hashlib.sha256(f"{i}|{w}".encode()).hexdigest()[:8], 16)
            lp = -((h % 1000) / 1000.0)
            out.append({"token": w, "logprob": lp,
                        "top_logprobs": [{"token": w if j == 0 else WORDS[(h + j) % len(WORDS)],
                                          "logprob": lp - j * 0.5} for j in range(max(1, k))]})
        return out


def make_handler(state: StandinState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "inferdrift-standin/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet (and key-free)
            return

        def _send(self, status: int, obj, *, raw_nan: bool = False) -> None:
            if raw_nan:  # fault injection only: emit a bare NaN constant, which strict clients refuse
                data = json.dumps(obj, allow_nan=True).encode("utf-8")
            else:
                data = json.dumps(_finite_or_none(obj), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth_ok(self) -> bool:
            if not state.api_key:
                return True
            hdr = self.headers.get("Authorization", "")
            if hdr == f"Bearer {state.api_key}":
                return True
            msg = "Incorrect API key provided"
            if state.leak:
                msg += f": {hdr[7:] if hdr.startswith('Bearer ') else hdr}"
            with state.lock:
                state.totals["errors"] += 1
            self._send(401, {"error": {"message": msg, "type": "invalid_request_error"}})
            return False

        def _body(self) -> dict:
            length = self.headers.get("Content-Length") or "0"
            if not length.isdigit() or int(length) > (64 << 20):
                raise _BadRequest("Content-Length must be a non-negative integer of at most 64 MiB")
            n = int(length)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                obj = loads_strict(raw)
            except (StrictJSONError, UnicodeDecodeError) as e:
                raise _BadRequest(f"request body is not strict JSON ({e})") from None
            if not isinstance(obj, dict):
                raise _BadRequest("request body must be a JSON object")
            return obj

        def _maybe_fail(self) -> bool:
            if state.fail_every and (state.counter + 1) % state.fail_every == 0:
                with state.lock:
                    state.counter += 1
                    state.totals["errors"] += 1
                msg = "internal error"
                if state.leak:
                    msg += f" while serving key {state.api_key}"
                self._send(500, {"error": {"message": msg}})
                return True
            return False

        def do_GET(self):  # noqa: N802
            if not self._auth_ok():
                return
            p = self.path.split("?")[0]
            if p in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok"})
            if p == "/v1/models":
                m = {"id": state.model_name, "object": "model", "owned_by": state.dialect}
                if state.dialect == "vllm":
                    m.update(root=state.model_name, max_model_len=32768)
                if state.dialect == "llamacpp":
                    m["meta"] = {"n_params": 596049920, "size": 0}
                return self._send(200, {"object": "list", "data": [m]})
            if p == "/ledger":
                with state.lock:
                    return self._send(200, dict(state.totals))
            if p in state.extra_routes:
                return self._send(200, state.extra_routes[p])
            if state.dialect == "llamacpp" and p == "/props":
                return self._send(200, {"default_generation_settings": {"n_ctx": 32768}, "total_slots": 1,
                                        "model_path": f"/models/{state.model_name}.gguf",
                                        "chat_template": "{# stand-in #}", "build_info": "standin-emulation"})
            if state.dialect == "vllm" and p == "/version":
                return self._send(200, {"version": "standin-emulation"})
            if state.dialect == "sglang" and p == "/get_model_info":
                return self._send(200, {"model_path": state.model_name, "is_generation": True})
            if state.dialect == "sglang" and p == "/get_server_info":
                return self._send(200, {"version": "standin-emulation", "tp_size": 1, "dp_size": 1,
                                        "quantization": None, "kv_cache_dtype": "auto",
                                        "attention_backend": "standin", "dtype": "auto",
                                        "speculative_algorithm": None})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            if not self._auth_ok():
                return
            p = self.path.split("?")[0]
            try:
                body = self._body()
                if p in ("/v1/chat/completions", "/v1/completions", "/completion", "/generate") and self._maybe_fail():
                    return
                if p == "/v1/chat/completions":
                    return self._chat(body)
                if p == "/v1/completions":
                    return self._completions(body)
                if p == "/completion" and state.dialect == "llamacpp":
                    return self._llama_completion(body)
                if p == "/tokenize" and state.dialect == "llamacpp":
                    txt = str(body.get("content", ""))
                    return self._send(200, {"tokens": list(range(approx_tokens(txt)))})
                if p == "/generate" and state.dialect == "sglang":
                    return self._sglang_generate(body)
            except _BadRequest as e:  # out-of-domain request: refused, not billed
                with state.lock:
                    state.totals["errors"] += 1
                return self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            self._send(404, {"error": {"message": "not found"}})

        def _chat(self, b: dict) -> None:
            messages = b.get("messages") or []
            tools = b.get("tools")
            key = request_key(messages=messages, tools=tools)
            if not isinstance(messages, list) or (tools is not None and not isinstance(tools, list)):
                raise _BadRequest("messages must be a list and tools a list or null")
            if b.get("chat_template_kwargs") is not None and not isinstance(b.get("chat_template_kwargs"), dict):
                raise _BadRequest("chat_template_kwargs must be an object or null")
            mt = _req_int(b, "max_tokens", _req_int(b, "max_completion_tokens", 256, 1, _MAX_REQ_TOKENS), 1,
                          _MAX_REQ_TOKENS)
            temperature, seed = _req_temperature(b), _req_seed(b.get("seed"))
            k = _req_int(b, "top_logprobs", 1, 0, 20)
            text, pt, ct, fin = state.generate(key, messages=messages, tools=tools,
                                               template_kwargs=b.get("chat_template_kwargs"), max_tokens=mt,
                                               temperature=temperature, seed=seed)
            state.bill("chat", pt, ct)
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}
            if b.get("logprobs") and not state.no_logprobs:
                choice["logprobs"] = {"content": state.fake_logprobs(text, k or 1)}
            u_pt, u_ct = state.usage_value(pt), state.usage_value(ct)
            self._send(200, {"id": "standin", "object": "chat.completion", "model": state.model_name,
                             "choices": [choice], "system_fingerprint": "standin",
                             "usage": {"prompt_tokens": u_pt, "completion_tokens": u_ct,
                                       "total_tokens": pt + ct if state.bad_usage is None else u_ct}},
                       raw_nan=state.bad_usage == "nan")

        def _completions(self, b: dict) -> None:
            prompt = b.get("prompt", "")
            key = request_key(prompt=prompt)
            mt = _req_int(b, "max_tokens", 16, 1, _MAX_REQ_TOKENS)
            temperature, seed = _req_temperature(b), _req_seed(b.get("seed"))
            nlp = b.get("logprobs")
            if nlp is False:
                nlp = None
            if nlp is not None and (isinstance(nlp, bool) or not isinstance(nlp, int) or not 0 <= nlp <= 20):
                raise _BadRequest("logprobs must be an integer in [0, 20]")
            text, pt, ct, fin = state.generate(key, prompt=prompt, max_tokens=mt, temperature=temperature, seed=seed)
            state.bill("completions", pt, ct)
            choice = {"index": 0, "text": text, "finish_reason": fin}
            if nlp and not state.no_logprobs:
                lp = state.fake_logprobs(text, int(nlp))
                choice["logprobs"] = {"tokens": [e["token"] for e in lp],
                                      "token_logprobs": [e["logprob"] for e in lp],
                                      "top_logprobs": [{t["token"]: t["logprob"] for t in e["top_logprobs"]}
                                                       for e in lp]}
            if b.get("prompt_logprobs") and state.dialect == "vllm" and not state.no_logprobs:
                n = pt
                choice["prompt_logprobs"] = [None] + [{"1": {"logprob": -1.0, "rank": 1}} for _ in range(n - 1)]
            u_pt, u_ct = state.usage_value(pt), state.usage_value(ct)
            self._send(200, {"id": "standin", "object": "text_completion", "model": state.model_name,
                             "choices": [choice],
                             "usage": {"prompt_tokens": u_pt, "completion_tokens": u_ct,
                                       "total_tokens": pt + ct if state.bad_usage is None else u_ct}},
                       raw_nan=state.bad_usage == "nan")

        def _llama_completion(self, b: dict) -> None:
            prompt = b.get("prompt", "")
            key = request_key(prompt=prompt)
            mt = _req_int(b, "n_predict", 128, 1, _MAX_REQ_TOKENS)
            temperature, seed = _req_temperature(b), _req_seed(b.get("seed"))
            n_probs = _req_int(b, "n_probs", 0, 0, 20)
            text, pt, ct, fin = state.generate(key, prompt=prompt, max_tokens=mt, temperature=temperature, seed=seed)
            if isinstance(prompt, list):
                pt = len(prompt)
            state.bill("completion", pt, ct)
            out = {"content": text, "stop": True, "stop_type": "eos" if fin == "stop" else "limit",
                   "tokens_predicted": state.usage_value(ct), "tokens_evaluated": state.usage_value(pt),
                   "model": state.model_name}
            if b.get("return_tokens"):
                out["tokens"] = [int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) for w in text.split()]
            if n_probs and not state.no_logprobs:
                out["completion_probabilities"] = state.fake_logprobs(text, n_probs)
            self._send(200, out, raw_nan=state.bad_usage == "nan")

        def _sglang_generate(self, b: dict) -> None:
            prompt = b.get("input_ids") if b.get("input_ids") is not None else b.get("text", "")
            key = request_key(prompt=prompt)
            sp = b.get("sampling_params") or {}
            if not isinstance(sp, dict):
                raise _BadRequest("sampling_params must be an object")
            mt = _req_int(sp, "max_new_tokens", 128, 1, _MAX_REQ_TOKENS)
            temperature, seed = _req_temperature(sp), _req_seed(sp.get("sampling_seed"))
            text, pt, ct, fin = state.generate(key, prompt=prompt, max_tokens=mt, temperature=temperature, seed=seed)
            if isinstance(prompt, list):
                pt = len(prompt)
            state.bill("generate", pt, ct)
            mi = {"prompt_tokens": state.usage_value(pt), "completion_tokens": state.usage_value(ct),
                  "finish_reason": {"type": fin}}
            if b.get("return_logprob") and not state.no_logprobs:
                if b.get("logprob_start_len", -1) == 0:
                    mi["input_token_logprobs"] = [[None, 0, None]] + [[-1.0, i, None] for i in range(1, pt)]
                lp = state.fake_logprobs(text, _req_int(b, "top_logprobs_num", 1, 0, 20) or 1)
                mi["output_token_logprobs"] = [[e["logprob"], i, None] for i, e in enumerate(lp)]
                mi["output_top_logprobs"] = [[[t["logprob"], j, None] for j, t in enumerate(e["top_logprobs"])]
                                             for e in lp]
            self._send(200, {"text": text, "meta_info": mi}, raw_nan=state.bad_usage == "nan")

    return Handler


def serve(state: StandinState, host: str = "127.0.0.1", port: int = 0):
    """Start the server in a daemon thread; return (server, bound_port)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, httpd.server_address[1]


def main_serve(args) -> int:
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    state = StandinState(backend=args.backend, price_in=args.price_in, price_out=args.price_out, api_key=key,
                         ledger_path=args.ledger, dialect=args.dialect, model_name=args.model_name,
                         ignore_seed=args.ignore_seed, no_logprobs=args.no_logprobs, truncate_frac=args.truncate_frac,
                         leak_key_in_errors=args.leak_key_in_errors, nondeterministic=args.nondeterministic,
                         fail_every=args.fail_every, work_dir=args.work_dir, bad_usage=getattr(args, "bad_usage", None),
                         noise_seed=getattr(args, "noise_seed", 0))
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    except OSError as e:  # e.g. the port is already in use: a clean error (exit 4), not a traceback
        raise FidelityError(f"the stand-in cannot listen on {args.host}:{args.port}: "
                            f"{e.strerror or type(e).__name__}", code="LISTEN_FAILED") from None
    print(json.dumps({"standin": "listening", "host": args.host, "port": httpd.server_address[1],
                      "dialect": args.dialect, "price_usd_per_mtok": [state.price_in, state.price_out]},
                     allow_nan=False), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    sys.stdout.flush()
    return 0
