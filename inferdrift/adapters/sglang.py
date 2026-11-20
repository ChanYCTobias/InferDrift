# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""SGLang server.

Chat mode uses the OpenAI-compatible route (with ``chat_template_kwargs``).  Raw
mode sends token ids to the native ``/generate`` route, which returns input and
output token log-probabilities.  The profile reads ``/get_model_info`` and
``/get_server_info`` (server arguments such as quantization, KV-cache dtype,
attention backend, tensor parallelism and speculative decoding).

Validation status: exercised against the protocol emulation in
``inferdrift.standin`` only; not run against a live SGLang engine by this
release (see the support matrix in the README)."""

from __future__ import annotations

import hashlib
import time

from ..errors import CapReached, EndpointError, PartialCapReached
from ..profile import field_value, finalize, unobservable
from ..redact import scrub_text
from .base import GenRequest, GenResult, provider_usage
from .openai_compat import OpenAICompatAdapter

VALIDATION = "protocol_emulation_only"


class SGLangAdapter(OpenAICompatAdapter):
    kind = "sglang"
    dialect = "sglang"
    sends_template_kwargs = True

    def profile(self) -> dict:
        p = super().profile()
        f = p["fields"]
        try:
            _, info, _ = self.client.get("/get_model_info", timeout=30)
        except EndpointError:
            info = None
        try:
            _, srv, _ = self.client.get("/get_server_info", timeout=30)
        except EndpointError:
            srv = None
        info = info if isinstance(info, dict) else {}
        srv = srv if isinstance(srv, dict) else {}
        f["engine"] = field_value("sglang", "endpoint:/get_server_info" if srv else "declared:spec")
        f["engine_version"] = field_value(srv.get("version"), "endpoint:/get_server_info") if srv.get("version") \
            else unobservable()
        if info.get("model_path"):
            f["model_id"] = field_value(info["model_path"], "endpoint:/get_model_info")
        if srv:
            f["quantization"] = field_value(srv.get("quantization"), "endpoint:/get_server_info")
            f["kv_dtype"] = field_value(srv.get("kv_cache_dtype"), "endpoint:/get_server_info")
            f["attention_backend"] = field_value(srv.get("attention_backend"), "endpoint:/get_server_info")
            f["weight_dtype"] = field_value(srv.get("dtype"), "endpoint:/get_server_info")
            f["parallelism"] = field_value({"tp": srv.get("tp_size"), "dp": srv.get("dp_size")},
                                           "endpoint:/get_server_info")
            f["speculative_decoding"] = field_value(srv.get("speculative_algorithm"), "endpoint:/get_server_info")
        p["flags"]["validation"] = VALIDATION
        return finalize(p)

    def _native_body(self, req: GenRequest, ids: list[int], *, logprob_start: int = -1) -> dict:
        sp = {"max_new_tokens": req.max_new_tokens}
        if req.decode.mode == "greedy":
            sp["temperature"] = 0.0
        else:
            sp.update(temperature=req.decode.temperature, top_p=req.decode.top_p, top_k=req.decode.top_k or -1)
        if req.seed is not None:
            sp["sampling_seed"] = req.seed
        body = {"input_ids": ids, "sampling_params": sp}
        if req.top_logprobs or logprob_start >= 0:
            body.update(return_logprob=True, top_logprobs_num=max(1, req.top_logprobs),
                        logprob_start_len=logprob_start)
        return body

    def _one(self, req: GenRequest, meter) -> GenResult:
        if self.mode != "raw":
            return super()._one(req, meter)
        t0 = time.monotonic()
        text_prompt, ids = self._raw_prompt(req)
        psha = hashlib.sha256(text_prompt.encode()).hexdigest()
        res = meter.reserve(self.role, len(ids), req.max_new_tokens)
        try:
            _, obj, _ = self.client.post("/generate", self._native_body(req, ids), timeout=meter.remaining_wall())
            if obj is not None and not isinstance(obj, dict):
                raise EndpointError("response is not a JSON object")
        except EndpointError as e:
            meter.settle(res, len(ids), req.max_new_tokens, usage_source="estimated_worst_case")
            return GenResult(req.item_id, "", "error", None, None, "estimated_worst_case", error=scrub_text(str(e)),
                             elapsed_s=time.monotonic() - t0, prompt_sha256=psha)
        obj = obj or {}
        mi = obj.get("meta_info") or {}
        if not isinstance(mi, dict):
            mi = {"prompt_tokens": mi, "completion_tokens": mi}  # malformed: tagged below, never used
        fr_raw = mi.get("finish_reason")
        fr_type = fr_raw.get("type") if isinstance(fr_raw, dict) else fr_raw
        fr = "stop" if fr_type in ("stop", "eos") else ("length" if fr_type == "length" else str(fr_type))
        pt, ct, src = provider_usage(mi.get("prompt_tokens"), mi.get("completion_tokens"), len(ids), req.max_new_tokens)
        meter.settle(res, pt, ct, usage_source=src)
        return GenResult(req.item_id, obj.get("text") or "", fr, pt, ct, src, elapsed_s=time.monotonic() - t0,
                         prompt_sha256=psha, logprobs=mi.get("output_top_logprobs") if req.top_logprobs else None)

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

    def prompt_logprobs(self, req: GenRequest, meter) -> list | None:
        if self.renderer is None:
            return None
        text = self.renderer.render(req.messages, req.tools, req.template_kwargs)
        ids = self.renderer.encode(text)
        res = meter.reserve(self.role, len(ids), 1, kind="probe")
        probe = GenRequest(req.item_id, req.messages, req.tools, req.template_kwargs, 1, req.decode)
        try:
            _, obj, _ = self.client.post("/generate", self._native_body(probe, ids, logprob_start=0))
        except EndpointError:
            meter.settle(res, len(ids), 1, usage_source="estimated_worst_case")
            return None
        mi = (obj if isinstance(obj, dict) else {}).get("meta_info") or {}
        if not isinstance(mi, dict):
            mi = {"prompt_tokens": mi, "completion_tokens": mi}  # malformed: tagged below, never used
        pt, ct, src = provider_usage(mi.get("prompt_tokens"), mi.get("completion_tokens"), len(ids), 1)
        meter.settle(res, pt, ct, usage_source=src)
        return mi.get("input_token_logprobs")
