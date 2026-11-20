# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON, strict parsing, digests and atomic writes.

Reading: every JSON input inferdrift acts on (item banks, judge
caches, profiles, Hub metadata, endpoint responses, repository config files) is
parsed by ``loads_strict``: a duplicate key, a NaN/Infinity constant or a number
that overflows to infinity is an error, never a silent last-wins or a non-finite
operand.

Writing: output is strict JSON (``allow_nan=False``).  A non-finite float that
reaches a writer is not turned into ``null`` silently; it is written as the
explicit tag ``{"nonfinite": "nan" | "inf" | "-inf"}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .redact import scrub


class StrictJSONError(ValueError):
    """Duplicate key, non-finite constant or overflowing number in a JSON input."""


def _no_duplicates(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise StrictJSONError(f"duplicate key {k[:80]!r}")
        out[k] = v
    return out


def _no_constant(name: str):
    raise StrictJSONError(f"non-finite constant {name}")


def _finite_float(text: str) -> float:
    x = float(text)
    if not math.isfinite(x):
        raise StrictJSONError(f"number {text[:40]} overflows to infinity")
    return x


def loads_strict(data: str | bytes | bytearray) -> Any:
    """``json.loads`` with duplicate-key detection and rejection of non-finite numbers."""
    try:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError:
                raise StrictJSONError("invalid UTF-8") from None
        return json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_no_constant,
                          parse_float=_finite_float)
    except StrictJSONError:
        raise
    except (ValueError, RecursionError) as e:  # JSONDecodeError, int digit limit, nesting
        raise StrictJSONError(str(e)[:200]) from None


def read_jsonl_strict(path: str | os.PathLike) -> list[tuple[int, Any]]:
    """[(line_number, value)] for every non-blank line, each parsed by ``loads_strict``.
    Errors carry the line number."""
    out = []
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append((no, loads_strict(line)))
            except StrictJSONError as e:
                raise StrictJSONError(f"line {no}: {e}") from None
    return out


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _tag_nonfinite(obj: Any) -> Any:
    """Replace a non-finite float by an explicit tag (never by a silent null)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"nonfinite": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: _tag_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tag_nonfinite(v) for v in obj]
    if hasattr(obj, "to_dict") and not isinstance(obj, type):
        return _tag_nonfinite(obj.to_dict())
    return obj


def dumps_strict(obj: Any, **kw) -> str:
    """Strict JSON text for printing (non-finite floats tagged explicitly)."""
    kw.setdefault("default", _default)
    return json.dumps(_tag_nonfinite(obj), allow_nan=False, **kw)


def canonical(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, strict (non-finite floats tagged)."""
    return json.dumps(_tag_nonfinite(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_default, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical(obj))


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: str | os.PathLike, obj: Any, *, redact: bool = True) -> None:
    data = scrub(obj) if redact else obj
    atomic_write_text(path, json.dumps(_tag_nonfinite(data), indent=1, sort_keys=True, ensure_ascii=False,
                                       default=_default, allow_nan=False) + "\n")


class JsonlWriter:
    """Append-only JSONL writer (one canonical object per line, flushed per line)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, obj: Any) -> None:
        self._f.write(canonical(obj) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: str | os.PathLike) -> list:
    """Every non-blank line, parsed strictly (see ``loads_strict``)."""
    return [v for _, v in read_jsonl_strict(path)]
