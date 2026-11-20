# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""vLLM OpenAI-compatible server.

Adds ``chat_template_kwargs`` on chat requests, token-id prompts on
``/v1/completions`` in raw mode, prompt log-probabilities through the
``prompt_logprobs`` extension, and the ``/version`` route for the profile.

Validation status: exercised against the protocol emulation in
``inferdrift.standin`` only; not run against a live vLLM engine by this
release (see the support matrix in the README)."""

from __future__ import annotations

from ..errors import EndpointError
from ..profile import field_value, finalize, unobservable
from .base import GenRequest, provider_usage
from .openai_compat import OpenAICompatAdapter

VALIDATION = "protocol_emulation_only"


class VLLMAdapter(OpenAICompatAdapter):
    kind = "vllm"
    dialect = "vllm"
    sends_template_kwargs = True

    def profile(self) -> dict:
        p = super().profile()
        f = p["fields"]
        try:
            _, v, _ = self.client.get("/version", timeout=30)
            ver = (v or {}).get("version") if isinstance(v, dict) else None
        except EndpointError:
            ver = None
        f["engine"] = field_value("vllm", "endpoint:/version" if ver else "declared:spec")
        f["engine_version"] = field_value(ver, "endpoint:/version") if ver else unobservable()
        for m in self.models():
            if isinstance(m, dict) and m.get("id") == self.model_id():
                f["output_caps"] = field_value({"max_model_len": m.get("max_model_len")}, "endpoint:/v1/models")
                if m.get("root"):
                    f["model_revision"] = field_value(m.get("root"), "endpoint:/v1/models")
        f["attention_backend"] = unobservable("chosen at server start; not reported over HTTP")
        p["flags"]["validation"] = VALIDATION
        return finalize(p)

    def _completion_body(self, req: GenRequest, prompt) -> dict:
        body = super()._completion_body(req, prompt)
        if self.mode == "raw" and self.renderer is not None and isinstance(prompt, str):
            body["prompt"] = self.renderer.encode(prompt)
        return body

    def prompt_logprobs(self, req: GenRequest, meter) -> list | None:
        if self.renderer is None:
            text = None
            prompt = None
        else:
            text = self.renderer.render(req.messages, req.tools, req.template_kwargs)
            prompt = self.renderer.encode(text)
        if prompt is None:
            return None
        res = meter.reserve(self.role, len(prompt), 1, kind="probe")
        try:
            _, obj, _ = self.client.post("/v1/completions", {"model": self.model_id(), "prompt": prompt,
                                                             "max_tokens": 1, "temperature": 0.0,
                                                             "prompt_logprobs": 1})
        except EndpointError:
            meter.settle(res, len(prompt), 1, usage_source="estimated_worst_case")
            return None
        body = obj if isinstance(obj, dict) else {}
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {"prompt_tokens": usage, "completion_tokens": usage}  # malformed: tagged below, never used
        pt, ct, src = provider_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), len(prompt), 1)
        meter.settle(res, pt, ct, usage_source=src)
        choices = body.get("choices")
        ch = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        return ch.get("prompt_logprobs")
