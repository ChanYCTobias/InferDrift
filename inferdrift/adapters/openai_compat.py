# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Any OpenAI-compatible endpoint (chat completions, or completions in raw mode).

Dialect subclasses (llama.cpp, vLLM, SGLang) reuse the request plumbing and add
their native profile and logprob channels."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

from ..errors import CapReached, EndpointError, FidelityError, PartialCapReached
from ..profile import field_value, finalize, host_hardware, new_profile, unobservable
from ..redact import scrub_text
from .base import Adapter, GenRequest, GenResult, canonical_tool_text, chat_bytes_upper, provider_usage
from .http import HttpClient


class OpenAICompatAdapter(Adapter):
    kind = "openai"
    dialect = "openai"
    sends_template_kwargs = False  # plain OpenAI rejects unknown fields

    def __init__(self, spec, role: str, *, api_key_env: str | None = None, renderer=None,
                 template_overhead_tokens: int = 512) -> None:
        super().__init__(spec, role)
        base = spec.base_url
        self.root = base[:-3] if base.endswith("/v1") else base
        self.v1 = self.root + "/v1"
        self.mode = spec.params.get("mode", "chat")
        self.client = HttpClient(self.root, api_key_env=api_key_env,
                                 timeout_s=float(spec.params.get("timeout_s", 300)))
        self.max_concurrency = max(1, int(spec.params.get("max_concurrency", 1)))
        self.renderer = renderer
        self.overhead = template_overhead_tokens
        if self.mode == "raw" and renderer is None:
            raise FidelityError("mode=raw needs tokenizer=hf:<repo> for client-side rendering", code="BAD_SPEC")
        self.model = spec.params.get("model")
        self._models_cache = None
        flag = spec.params.get("send_template_kwargs")
        if flag is not None:  # opt in (or out): many OpenAI-compatible servers accept chat_template_kwargs
            self.sends_template_kwargs = flag in ("1", "true", "yes")

    # ---------------------------------------------------------------- metadata
    def models(self) -> list:
        if self._models_cache is None:
            try:
                _, obj, _ = self.client.get("/v1/models", timeout=30)
                self._models_cache = (obj or {}).get("data", []) if isinstance(obj, dict) else []
            except EndpointError as e:
                if e.status in (401, 403):
                    raise  # an authentication failure is reported, never mistaken for an empty model list
                self._models_cache = []
        return self._models_cache

    def model_id(self) -> str:
        if not self.model:
            ms = self.models()
            if not ms:
                raise FidelityError("endpoint lists no models; pass model=<id> in the spec", code="BAD_SPEC")
            self.model = ms[0].get("id")
        return self.model

    def profile(self) -> dict:
        p = new_profile(self.kind, self.spec.display())
        f = p["fields"]
        f["model_id"] = field_value(self.model_id(), "endpoint:/v1/models")
        f["engine"] = field_value(self.dialect, "declared:spec")
        f["prompt_rendering"] = field_value(
            "client:" + (self.renderer.template_sha256 if self.renderer else "") if self.mode == "raw"
            else "server", "declared:spec")
        if self.renderer is not None:
            f["chat_template_sha256"] = field_value(self.renderer.template_sha256, "runtime:client_renderer")
        else:
            f["chat_template_sha256"] = unobservable("rendered by the server")
        f["hardware"] = unobservable("remote endpoint")
        f["decode_parameters"] = field_value("set per request by inferdrift", "declared:tool")
        return finalize(p)

    # ---------------------------------------------------------------- requests
    def _decode_fields(self, req: GenRequest) -> dict:
        d = {}
        if req.decode.mode == "greedy":
            d["temperature"] = 0.0
        else:
            d["temperature"] = req.decode.temperature
            d["top_p"] = req.decode.top_p
        if req.seed is not None:
            d["seed"] = req.seed
        return d

    def _chat_body(self, req: GenRequest) -> dict:
        body = {"model": self.model_id(), "messages": req.messages, "max_tokens": req.max_new_tokens,
                "stream": False}
        if req.tools:
            body["tools"] = req.tools
        body.update(self._decode_fields(req))
        if req.top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = req.top_logprobs
        if self.sends_template_kwargs and req.template_kwargs:
            body["chat_template_kwargs"] = req.template_kwargs
        return body

    def _raw_prompt(self, req: GenRequest) -> tuple[str, list[int]]:
        text = self.renderer.render(req.messages, req.tools, req.template_kwargs)
        return text, self.renderer.encode(text)

    def _completion_body(self, req: GenRequest, prompt) -> dict:
        body = {"model": self.model_id(), "prompt": prompt, "max_tokens": req.max_new_tokens, "stream": False}
        body.update(self._decode_fields(req))
        if req.top_logprobs:
            body["logprobs"] = req.top_logprobs
        return body

    def _parse_chat(self, obj: dict) -> tuple[str, str, dict, list | None]:
        ch = (obj.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = canonical_tool_text(msg.get("content"), msg.get("tool_calls")) if msg.get("tool_calls") \
            else (msg.get("content") or "")
        fr = ch.get("finish_reason") or "unknown"
        fr = "stop" if fr in ("stop", "tool_calls", "eos") else ("length" if fr == "length" else fr)
        lp = None
        if isinstance(ch.get("logprobs"), dict) and ch["logprobs"].get("content") is not None:
            lp = ch["logprobs"]["content"]
        return text, fr, obj.get("usage") or {}, lp

    def _parse_completion(self, obj: dict) -> tuple[str, str, dict, list | None]:
        ch = (obj.get("choices") or [{}])[0]
        text = ch.get("text") or ""
        fr = ch.get("finish_reason") or "unknown"
        fr = "stop" if fr in ("stop", "eos") else ("length" if fr == "length" else fr)
        lp = None
        if isinstance(ch.get("logprobs"), dict) and ch["logprobs"].get("top_logprobs") is not None:
            lp = ch["logprobs"]["top_logprobs"]
        return text, fr, obj.get("usage") or {}, lp

    def _one(self, req: GenRequest, meter) -> GenResult:
        t0 = time.monotonic()
        if self.mode == "raw":
            text_prompt, ids = self._raw_prompt(req)
            upper = len(ids)
            psha = hashlib.sha256(text_prompt.encode()).hexdigest()
        else:
            upper = chat_bytes_upper(req.messages, req.tools, self.overhead)
            psha = None
        res = meter.reserve(self.role, upper, req.max_new_tokens)
        try:
            rem = meter.remaining_wall()
            if self.mode == "raw":
                _, obj, _ = self.client.post("/v1/completions", self._completion_body(req, text_prompt), timeout=rem)
                if obj is not None and not isinstance(obj, dict):
                    raise EndpointError("response is not a JSON object")
                text, fr, usage, lp = self._parse_completion(obj or {})
            else:
                _, obj, _ = self.client.post("/v1/chat/completions", self._chat_body(req), timeout=rem)
                if obj is not None and not isinstance(obj, dict):
                    raise EndpointError("response is not a JSON object")
                text, fr, usage, lp = self._parse_chat(obj or {})
        except EndpointError as e:
            # The outcome (and the bill) of a failed request is unknown: charge the worst case.
            meter.settle(res, upper, req.max_new_tokens, usage_source="estimated_worst_case")
            return GenResult(req.item_id, "", "error", None, None, "estimated_worst_case", error=scrub_text(str(e)),
                             elapsed_s=time.monotonic() - t0, prompt_sha256=psha)
        if not isinstance(usage, dict):
            usage = {"prompt_tokens": usage, "completion_tokens": usage}  # malformed: tagged below, never used
        pt, ct, src = provider_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), upper,
                                     req.max_new_tokens)
        meter.settle(res, pt, ct, usage_source=src)
        return GenResult(req.item_id, text, fr, pt, ct, src, elapsed_s=time.monotonic() - t0, prompt_sha256=psha,
                         logprobs=lp)

    def generate(self, reqs: list[GenRequest], meter) -> list[GenResult]:
        if self.max_concurrency <= 1 or len(reqs) <= 1:
            out = []
            for r in reqs:
                try:
                    out.append(self._one(r, meter))
                except CapReached as e:
                    raise PartialCapReached(out, e) from None
            return out
        results: dict[int, GenResult] = {}
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {ex.submit(self._one, r, meter): i for i, r in enumerate(reqs)}
            for fut, i in futs.items():
                try:
                    results[i] = fut.result()
                except CapReached as e:
                    errors.append(e)
        if errors:
            # Keep the completed prefix only, so look accounting stays a prefix of the order.
            done = []
            for i in range(len(reqs)):
                if i not in results:
                    break
                done.append(results[i])
            raise PartialCapReached(done, errors[0])
        return [results[i] for i in range(len(reqs))]

