# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""llama.cpp server (``llama-server``).

``mode=chat`` uses the OpenAI-compatible route with the server's own chat
template.  ``mode=raw`` renders the prompt client-side (``tokenizer=hf:...``) and
sends token ids to the native ``/completion`` route, which also exposes top-k
token probabilities (``n_probs``) and the generated token ids.  The profile
comes from ``/props`` and ``/v1/models``."""

from __future__ import annotations

import hashlib
import os
import time

from ..errors import CapReached, EndpointError, PartialCapReached
from ..profile import field_value, finalize, unobservable
from ..redact import scrub_text
from .base import GenRequest, GenResult, provider_usage
from .openai_compat import OpenAICompatAdapter


class LlamaCppAdapter(OpenAICompatAdapter):
    kind = "llamacpp"
    dialect = "llamacpp"
    sends_template_kwargs = True

    @property
    def cache_prompt(self) -> bool:
        """Prompt-cache reuse is off unless the spec says cache_prompt=1: reusing a cached prefix changes the
        batch shape of the prompt evaluation, which with some quantized kernels changes greedy outputs between
        identical requests (seen as a greedy repeat-identity rate below 1)."""
        return self.spec.params.get("cache_prompt", "0") in ("1", "true", "yes")

    def _chat_body(self, req: GenRequest) -> dict:
        body = super()._chat_body(req)
        body["cache_prompt"] = self.cache_prompt
        return body

    def props(self) -> dict:
        try:
            _, obj, _ = self.client.get("/props", timeout=30)
            return obj if isinstance(obj, dict) else {}
        except EndpointError:
            return {}

    def profile(self) -> dict:
        p = super().profile()
        f = p["fields"]
        props = self.props()
        f["engine"] = field_value("llama.cpp", "endpoint:/props")
        bi = props.get("build_info")
        f["engine_version"] = field_value(bi, "endpoint:/props") if bi else unobservable("no build_info in /props")
        mp = props.get("model_path")
        if mp:
            f["model_id"] = field_value(os.path.basename(str(mp)), "endpoint:/props")
        tmpl = props.get("chat_template")
        if self.mode == "chat" and isinstance(tmpl, str):
            f["chat_template_sha256"] = field_value(hashlib.sha256(tmpl.encode()).hexdigest(), "endpoint:/props")
        dgs = props.get("default_generation_settings") or {}
        n_ctx = dgs.get("n_ctx")
        f["output_caps"] = field_value({"n_ctx": n_ctx, "per_request_max_tokens": "set by inferdrift"},
                                       "endpoint:/props")
        f["parallelism"] = field_value({"slots": props.get("total_slots")}, "endpoint:/props")
        f["decode_parameters"] = field_value({"set_per_request_by": "inferdrift", "cache_prompt": self.cache_prompt},
                                             "declared:spec")
        meta = {}
        for m in self.models():
            if isinstance(m, dict) and isinstance(m.get("meta"), dict):
                meta = m["meta"]
                break
        if meta.get("n_params"):
            f["parameters_stored"] = field_value(meta.get("n_params"), "endpoint:/v1/models")
        f["quantization"] = field_value({"model_file": os.path.basename(str(mp)) if mp else None,
                                         "size_bytes": meta.get("size")}, "endpoint:/v1/models")
        f["attention_backend"] = unobservable("llama.cpp does not report kernel selection over HTTP")
        f["gemm_backend"] = unobservable("llama.cpp does not report kernel selection over HTTP")
        f["speculative_decoding"] = field_value(bool(props.get("speculative")) if "speculative" in props else None,
                                                "endpoint:/props")
        p["flags"]["props_keys"] = sorted(props)[:40]
        return finalize(p)

    def _native_body(self, req: GenRequest, ids: list[int]) -> dict:
        body = {"prompt": ids, "n_predict": req.max_new_tokens, "cache_prompt": self.cache_prompt, "return_tokens": True,
                "stream": False, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "min_p": 0.0, "top_p": 1.0}
        if req.decode.mode == "greedy":
            body["temperature"] = 0.0
            body["top_k"] = 1
        else:
            body["temperature"] = req.decode.temperature
            body["top_k"] = req.decode.top_k or 0
            body["top_p"] = req.decode.top_p
        if req.seed is not None:
            body["seed"] = req.seed
        if req.top_logprobs:
            body["n_probs"] = req.top_logprobs
        return body

    def _one(self, req: GenRequest, meter) -> GenResult:
        if self.mode != "raw":
            return super()._one(req, meter)
        t0 = time.monotonic()
        text_prompt, ids = self._raw_prompt(req)
        psha = hashlib.sha256(text_prompt.encode()).hexdigest()
        res = meter.reserve(self.role, len(ids), req.max_new_tokens)
        try:
            _, obj, _ = self.client.post("/completion", self._native_body(req, ids), timeout=meter.remaining_wall())
            if obj is not None and not isinstance(obj, dict):
                raise EndpointError("response is not a JSON object")
        except EndpointError as e:
            meter.settle(res, len(ids), req.max_new_tokens, usage_source="estimated_worst_case")
            return GenResult(req.item_id, "", "error", None, None, "estimated_worst_case", error=scrub_text(str(e)),
                             elapsed_s=time.monotonic() - t0, prompt_sha256=psha)
        obj = obj or {}
        text = obj.get("content") or ""
        st = obj.get("stop_type")
        if st is None:
            st = "eos" if obj.get("stopped_eos") else ("limit" if obj.get("stopped_limit") else
                                                         ("word" if obj.get("stopped_word") else None))
        fr = "stop" if st in ("eos", "word") else ("length" if st == "limit" else str(st))
        _, ct, src = provider_usage(len(ids), obj.get("tokens_predicted"), len(ids), req.max_new_tokens)
        meter.settle(res, len(ids), ct, usage_source=src)
        lp = obj.get("completion_probabilities") if req.top_logprobs else None
        return GenResult(req.item_id, text, fr, len(ids), ct, src, elapsed_s=time.monotonic() - t0,
                         prompt_sha256=psha, logprobs=lp,
                         meta={"tokens": obj.get("tokens"), "tokens_evaluated": obj.get("tokens_evaluated"),
                               "truncated": obj.get("truncated")})

    def generate(self, reqs, meter):
        if self.mode != "raw":
            return super().generate(reqs, meter)
        out = []
        for r in reqs:
            try:
                out.append(self._one(r, meter))
            except CapReached as e:
                raise PartialCapReached(out, e) from None
        return out
