# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Minimal JSON-over-HTTP client (standard library) with key redaction.

Responses are parsed strictly: a duplicate key or a NaN/Infinity
constant makes the response an ``EndpointError`` (the request is then charged at
its worst case), never a silently accepted value."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .. import domains
from ..errors import EndpointError
from ..jsonio import StrictJSONError, loads_strict
from ..redact import register_secret, scrub_text

USER_AGENT = "inferdrift/0.1"


class HttpClient:
    def __init__(self, base_url: str, *, api_key_env: str | None = None, timeout_s: float = 300.0,
                 retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = domains.number(timeout_s, "timeout_s", gt=0.0, le=86400.0)
        self.retries = domains.integer(retries, "retries", ge=0, le=10)
        self._key = None
        self.api_key_env = api_key_env
        if api_key_env:
            key = os.environ.get(api_key_env)
            if key:
                register_secret(key)
                self._key = key

    @property
    def has_key(self) -> bool:
        return self._key is not None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if self._key:
            h["Authorization"] = "Bearer " + self._key
        return h

    def request(self, method: str, path: str, body: dict | None = None, *, timeout: float | None = None,
                allow_status: tuple[int, ...] = ()) -> tuple[int, dict | list | None, dict]:
        url = self.base_url + path
        data = json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None  # never send NaN
        t = self.timeout_s if timeout is None else max(1.0, min(self.timeout_s, timeout))
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
            try:
                with urllib.request.urlopen(req, timeout=t) as r:  # noqa: S310 - user-named endpoint
                    raw = r.read(256 << 20)
                    hdrs = {k.lower(): v for k, v in r.headers.items()}
                    status = r.status
            except urllib.error.HTTPError as e:
                raw = e.read(1 << 16) if e.fp else b""
                status = e.code
                hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
                if status in allow_status:
                    try:
                        return status, loads_strict(raw.decode("utf-8", errors="replace")), hdrs
                    except StrictJSONError:
                        return status, None, hdrs
                if status in (429, 502, 503, 504) and attempt <= self.retries:
                    time.sleep(min(8.0, 0.5 * 2 ** attempt))
                    continue
                msg = raw.decode("utf-8", errors="replace")[:600]
                raise EndpointError(f"HTTP {status} from {method} {path}: {scrub_text(msg)}", status=status) from None
            except urllib.error.URLError as e:
                if attempt <= self.retries:
                    time.sleep(min(8.0, 0.5 * 2 ** attempt))
                    continue
                raise EndpointError(f"endpoint unreachable ({method} {path}): {scrub_text(str(e.reason))}") from None
            except TimeoutError:
                raise EndpointError(f"timeout after {t:.0f} s ({method} {path})") from None
            if not raw:
                return status, None, hdrs
            try:
                return status, loads_strict(raw), hdrs
            except (StrictJSONError, UnicodeDecodeError) as e:
                raise EndpointError(f"response from {method} {path} is not strict JSON "
                                    f"({scrub_text(str(e))[:120]})") from None

    def get(self, path: str, **kw):
        return self.request("GET", path, None, **kw)

    def post(self, path: str, body: dict, **kw):
        return self.request("POST", path, body, **kw)
