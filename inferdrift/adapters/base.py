# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapter interface shared by the reference (transformers) and endpoint adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .. import domains
from ..errors import BadInputError

MAX_SEED = domains.MAX_EXACT_INT  # seeds live in the exact wire range; derived seeds wrap inside it
MAX_NEW_TOKENS = 1 << 20


@dataclass(frozen=True)
class DecodeConfig:
    """Decoding parameters, each inside its domain: temperature finite and
    > 0, top_p in (0, 1], top_k and seed non-negative integers (never booleans)."""

    mode: str = "greedy"  # greedy | sample
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        domains.choice(self.mode, "decode", ("greedy", "sample"))
        object.__setattr__(self, "temperature", domains.number(self.temperature, "temperature", gt=0.0, le=1e4))
        object.__setattr__(self, "top_p", domains.number(self.top_p, "top_p", gt=0.0, le=1.0))
        domains.integer(self.top_k, "top_k", ge=0, le=1 << 31)
        domains.integer(self.seed, "seed", ge=0, le=MAX_SEED)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenRequest:
    item_id: str
    messages: list
    tools: list | None
    template_kwargs: dict
    max_new_tokens: int
    decode: DecodeConfig
    seed: int | None = None
    coupling_key: str = ""
    top_logprobs: int = 0

    def __post_init__(self) -> None:
        domains.integer(self.max_new_tokens, "max_new_tokens", ge=1, le=MAX_NEW_TOKENS)
        domains.integer(self.top_logprobs, "top_logprobs", ge=0, le=64)
        domains.optional_integer(self.seed, "request seed", ge=0, le=MAX_SEED)
        if not isinstance(self.decode, DecodeConfig):
            raise BadInputError("decode", "must be a DecodeConfig", self.decode)


def provider_usage(prompt_tokens, completion_tokens, upper_in: int, upper_out: int) -> tuple[int, int, str]:
    """(tokens_in, tokens_out, usage_source) for the meter from a provider's usage report.

    Both counts present as non-negative integers in the exact range -> the provider's
    numbers.  Either count absent (None) -> the request's worst case, ``estimated``.
    Either count present but malformed (negative, boolean, float, string, beyond the exact
    range) -> the worst case, tagged ``invalid_provider_usage``: a provider's malformed
    usage is never a budget operand."""
    def ok(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= domains.MAX_EXACT_INT

    if ok(prompt_tokens) and ok(completion_tokens):
        return int(prompt_tokens), int(completion_tokens), "provider"
    if all(v is None or ok(v) for v in (prompt_tokens, completion_tokens)):
        return upper_in, upper_out, "estimated"
    return upper_in, upper_out, "invalid_provider_usage"


@dataclass
class GenResult:
    item_id: str
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str
    error: str | None = None
    elapsed_s: float = 0.0
    prompt_sha256: str | None = None
    logprobs: list | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def request_seed(base_seed: int, item_id: str, replicate: int = 0) -> int:
    """Per-item sampling seed derived from the run seed (31-bit, positive)."""
    h = hashlib.sha256(f"{base_seed}|{item_id}|{replicate}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def chat_bytes_upper(messages: list, tools: list | None, overhead_tokens: int) -> int:
    """Upper bound on prompt tokens for a chat request.

    Every token of a byte-level or byte-fallback tokenizer covers at least one
    byte of the text it encodes, so the UTF-8 length of the message and tool JSON
    bounds the tokens they produce; ``overhead_tokens`` covers what the chat
    template adds (role markers and tool-use instructions)."""
    text = json.dumps({"m": messages, "t": tools}, ensure_ascii=False)
    return len(text.encode("utf-8")) + overhead_tokens


def canonical_tool_text(content: str | None, tool_calls: list | None) -> str:
    """Serialise an endpoint's structured tool calls into the text form graders read.

    The arguments are model output (data under test, graded as text), never a control
    or budget operand, so they are kept as the model produced them."""
    parts = []
    if content:
        parts.append(content)
    for tc in tool_calls or []:
        fn = (tc or {}).get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parts.append("<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": args},
                                                   ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(parts)


class Adapter:
    """Base class.  Subclasses implement ``generate`` and ``profile``."""

    kind = "base"
    supports_shared_noise = False

    def __init__(self, spec, role: str) -> None:
        self.spec = spec
        self.role = role

    def profile(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(self, reqs: list[GenRequest], meter) -> list[GenResult]:  # pragma: no cover - abstract
        raise NotImplementedError

    def prompt_logprobs(self, req: GenRequest, meter) -> list | None:
        """Log-probabilities of the prompt tokens, or None when the channel lacks them."""
        return None

    def close(self) -> None:
        pass
