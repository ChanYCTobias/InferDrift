# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Credential redaction.

API keys are read from environment variables named on the command line and are
registered here the moment they are read.  Every string the package writes
(reports, item records, summaries, error messages, log lines) passes
``scrub_text``/``scrub`` first, which replaces

* every registered secret, and its URL-encoded and base64 forms, and
* anything that looks like a bearer token or a common provider key format

with ``[REDACTED]``.  Keys never appear on the command line, in stack specs or in
profiles; only the *name* of the environment variable is recorded.
"""

from __future__ import annotations

import base64
import re
import threading
import urllib.parse
from typing import Any

REDACTED = "[REDACTED]"
_lock = threading.Lock()
_secrets: set[str] = set()

# Generic shapes: bearer headers, sk-/hf_/pk- style keys, long hex or base64 runs
# following key-like field names.
_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}"),
    re.compile(r"\b(?:pk|rk|ak)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)((?:api[_-]?key|x-api-key|authorization|token|secret)\"?\s*[:=]\s*\"?)[^\s\",;]{8,}"),
]


def register_secret(value: str | None) -> None:
    """Register a secret value so that it is scrubbed from every output."""
    if not value or len(value) < 4:
        return
    forms = {value, urllib.parse.quote(value, safe=""), urllib.parse.quote_plus(value)}
    try:
        forms.add(base64.b64encode(value.encode()).decode().rstrip("="))
    except Exception:  # pragma: no cover - str always encodes
        pass
    with _lock:
        _secrets.update(f for f in forms if len(f) >= 4)


def clear_secrets() -> None:
    with _lock:
        _secrets.clear()


def scrub_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    with _lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, REDACTED)
    for pat in _PATTERNS:
        if pat.groups:
            text = pat.sub(lambda m: m.group(1) + REDACTED, text)
        else:
            text = pat.sub(REDACTED, text)
    return text


def scrub(obj: Any) -> Any:
    """Recursively scrub strings inside JSON-like data (dict keys too)."""
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {scrub_text(k) if isinstance(k, str) else k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def scrub_output_text(text: str) -> str:
    """Scrub only registered secrets (not generic patterns).

    Model outputs are compared byte for byte, so generic pattern scrubbing would
    alter legitimate text (for example a model writing 'token: abcdefgh').  Item
    records therefore get exact-secret scrubbing only; reports and messages get
    the full ``scrub_text``."""
    if not isinstance(text, str) or not text:
        return text
    with _lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, REDACTED)
    return text
