# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Stack specifications.

    hf:<org/name>[@revision][?attn=sdpa&dtype=float32&threads=8&batch=16&quant=rtn-int4-g128&template=FILE]
    hf:/path/to/local/dir[?...]
    http(s)://host:port[/v1][?dialect=openai|llamacpp|vllm|sglang&mode=chat|raw&model=ID&tokenizer=hf:...
                              &send_template_kwargs=1&cache_prompt=1]

API keys never appear in a spec; ``--ref-api-key-env`` / ``--cand-api-key-env``
name the environment variable that holds one.

Every parameter has a closed domain: counts are plain positive
integers (``threads`` 1..1024, ``batch`` 1..4096, ``max_batch_tokens`` 1..2**31,
``max_concurrency`` 1..256), ``timeout_s`` is a plain decimal in (0, 86400],
enumerations take only their listed values, and a parameter given twice is refused.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import asdict, dataclass, field

from . import domains
from .errors import BadInputError, SpecError

HF_PARAMS = {"attn", "dtype", "threads", "batch", "quant", "template", "name", "max_batch_tokens"}
EP_PARAMS = {"dialect", "mode", "model", "tokenizer", "name", "max_concurrency", "timeout_s", "cache_prompt",
             "send_template_kwargs"}
DIALECTS = ("auto", "openai", "llamacpp", "vllm", "sglang")
DTYPES = ("float32", "bfloat16", "float16")
ATTN = ("sdpa", "eager", "flash_attention_2", "flash_attention_3", "flex_attention")
FLAGS = ("0", "1", "true", "false", "yes", "no")
_INT_PARAMS = {"threads": 1024, "batch": 4096, "max_batch_tokens": 1 << 31, "max_concurrency": 256}
_QUANT = re.compile(r"rtn-int([0-9]{1,2})-g([0-9]{1,6})")


def _params(query: str, allowed: set, kind: str) -> dict:
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    params: dict = {}
    for k, v in pairs:
        if k in params:
            raise SpecError(f"{kind} spec parameter {k!r} given more than once")
        params[k] = v
    bad = set(params) - allowed
    if bad:
        raise SpecError(f"unknown {kind} spec parameter(s): {sorted(bad)}")
    for k, v in params.items():
        field_name = f"{kind} spec parameter {k}"
        if v == "":
            raise BadInputError(field_name, "must not be empty")
        if k in _INT_PARAMS:
            domains.parse_integer(v, field_name, ge=1, le=_INT_PARAMS[k])
        elif k == "timeout_s":
            domains.parse_number(v, field_name, gt=0.0, le=86400.0)
        elif k == "dtype":
            domains.choice(v, field_name, DTYPES)
        elif k == "attn":
            domains.choice(v, field_name, ATTN)
        elif k in ("cache_prompt", "send_template_kwargs"):
            domains.choice(v, field_name, FLAGS)
        elif k == "quant":
            m = _QUANT.fullmatch(v)
            if not m or not 2 <= int(m.group(1)) <= 8 or not 1 <= int(m.group(2)) <= 65536:
                raise BadInputError(field_name, "must be rtn-int<bits>-g<group> with bits 2..8 and group 1..65536", v)
    return params


@dataclass
class StackSpec:
    kind: str
    raw: str
    repo: str | None = None
    local_path: str | None = None
    revision: str | None = None
    base_url: str | None = None
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.params.get("name") or self.display()

    def display(self) -> str:
        if self.kind == "hf":
            src = self.repo or self.local_path
            rev = f"@{self.revision}" if self.revision else ""
            q = "&".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            return f"hf:{src}{rev}" + (f"?{q}" if q else "")
        q = "&".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.base_url}" + (f"?{q}" if q else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["display"] = self.display()
        return d


def parse_spec(text: str) -> StackSpec:
    if not isinstance(text, str) or not text.strip():
        raise SpecError("empty stack spec")
    text = text.strip()
    if text.startswith("hf:"):
        body = text[3:]
        query = ""
        if "?" in body:
            body, query = body.split("?", 1)
        params = _params(query, HF_PARAMS, "hf")
        revision = None
        if body.startswith(("/", "./", "../", "~")):
            local = body
            if "@" in local.rsplit("/", 1)[-1]:
                local, revision = local.rsplit("@", 1)
            return StackSpec("hf", text, local_path=local, revision=revision, params=params)
        if "@" in body:
            body, revision = body.split("@", 1)
        if body.count("/") != 1 or not all(body.split("/")):
            raise SpecError(f"hf repo must look like org/name, got {body!r}")
        return StackSpec("hf", text, repo=body, revision=revision or "main", params=params)
    if text.startswith(("http://", "https://")):
        u = urllib.parse.urlsplit(text)
        if u.username or u.password:
            raise SpecError("credentials in endpoint URLs are not accepted; use --*-api-key-env")
        params = _params(u.query, EP_PARAMS, "endpoint")
        try:
            u.port  # noqa: B018 - raises ValueError for a port outside 0..65535
        except ValueError:
            raise BadInputError("endpoint spec port", "must be an integer in 0..65535", u.netloc) from None
        if not u.hostname:
            raise SpecError("endpoint spec has no host")
        if params.get("dialect", "auto") not in DIALECTS:
            raise SpecError(f"unknown dialect {params.get('dialect')!r}")
        if params.get("mode", "chat") not in ("chat", "raw"):
            raise SpecError("mode must be chat or raw")
        if "tokenizer" in params and parse_spec(params["tokenizer"]).kind != "hf":
            raise SpecError("tokenizer= must be an hf: spec")
        base = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.rstrip("/"), "", ""))
        return StackSpec("endpoint", text, base_url=base, params=params)
    raise SpecError(f"unrecognised stack spec {text!r} (expected hf:... or http(s)://...)")
