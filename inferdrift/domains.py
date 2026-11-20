# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Closed domains at inferdrift's executable boundaries.

Every resource, control and statistical input is accepted only inside its declared
domain; anything else raises ``BadInputError`` naming the input and the rule it
breaks, before a request is sent or an output is written (the CLI exits 3).

* Numbers are ``int`` or ``float``; a boolean is never a number.  They must be
  finite and inside their interval.
* Counts and byte sizes are integers (never booleans) in the exact wire range
  ``|n| <= 2**53 - 1``.  Narrowing conversions are checked against that range.
* Command-line text is parsed with a strict grammar: a plain decimal number or a
  plain integer.  ``nan``, ``inf``, ``1e999`` (which overflows to infinity),
  hexadecimal, underscores, blanks and non-ASCII digits are refused, not coerced.
* Absence is kept distinct from zero: an absent cap (``None``) means "no cap";
  a cap of zero or below is refused.
"""

from __future__ import annotations

import math
import re

from .errors import BadInputError

MAX_EXACT_INT = 2 ** 53 - 1
GIB = 1 << 30

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_MAX_NUMBER_TEXT = 64


def _fmt(b) -> str:
    return str(b) if isinstance(b, int) else f"{b:g}"


def _bounds(x, field, value, gt, ge, lt, le) -> None:
    if gt is not None and not x > gt:
        raise BadInputError(field, f"must be > {_fmt(gt)}", value)
    if ge is not None and not x >= ge:
        raise BadInputError(field, f"must be >= {_fmt(ge)}", value)
    if lt is not None and not x < lt:
        raise BadInputError(field, f"must be < {_fmt(lt)}", value)
    if le is not None and not x <= le:
        raise BadInputError(field, f"must be <= {_fmt(le)}", value)


# ----------------------------------------------------------------------- typed values
def number(value, field: str, *, gt=None, ge=None, lt=None, le=None) -> float:
    """A finite real number inside the given bounds, returned as float."""
    if isinstance(value, bool):
        raise BadInputError(field, "must be a number; a boolean is not accepted", value)
    if isinstance(value, int):
        if abs(value) > MAX_EXACT_INT:
            raise BadInputError(field, f"must lie within the exact range +-{MAX_EXACT_INT}", value)
        x = float(value)
    elif isinstance(value, float):
        x = float(value)
    else:
        raise BadInputError(field, "must be a number", value)
    if not math.isfinite(x):
        raise BadInputError(field, "must be finite (NaN and infinity are not accepted)", value)
    _bounds(x, field, value, gt, ge, lt, le)
    return x


def integer(value, field: str, *, ge=None, le=None) -> int:
    """An integer (never a boolean) in the exact wire range and the given bounds."""
    if isinstance(value, bool):
        raise BadInputError(field, "must be an integer; a boolean is not accepted", value)
    if not isinstance(value, int):
        raise BadInputError(field, "must be an integer", value)
    v = int(value)
    if abs(v) > MAX_EXACT_INT:
        raise BadInputError(field, f"must lie within the exact range +-{MAX_EXACT_INT}", value)
    _bounds(v, field, value, None, ge, None, le)
    return v


def optional_number(value, field: str, **bounds) -> float | None:
    return None if value is None else number(value, field, **bounds)


def optional_integer(value, field: str, **bounds) -> int | None:
    return None if value is None else integer(value, field, **bounds)


def boolean(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise BadInputError(field, "must be true or false", value)
    return value


def choice(value, field: str, options) -> str:
    if not isinstance(value, str) or value not in options:
        raise BadInputError(field, f"must be one of {', '.join(options)}", value)
    return value


def text(value, field: str, *, max_len: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadInputError(field, "must be a non-empty string", value)
    if len(value) > max_len or "\x00" in value:
        raise BadInputError(field, f"must be at most {max_len} characters without NUL bytes")
    return value


def optional_text(value, field: str, **kw) -> str | None:
    return None if value is None else text(value, field, **kw)


def env_name(value, field: str) -> str | None:
    """The NAME of an environment variable.  The value is never echoed: a caller who
    passed a key here by mistake must not see it printed back."""
    if value is None:
        return None
    if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
        raise BadInputError(field, "must be the name of an environment variable ([A-Za-z_][A-Za-z0-9_]*, at "
                                   "most 128 characters); the key itself never goes on the command line")
    return value


def gib_to_bytes(cap_gib, field: str) -> int:
    """A positive, finite GiB allowance as an exact byte count (checked before narrowing)."""
    x = number(cap_gib, field, gt=0)
    nbytes = x * GIB
    if not nbytes <= MAX_EXACT_INT:
        raise BadInputError(field, f"must be at most {MAX_EXACT_INT // GIB} GiB (exact byte range)", cap_gib)
    out = int(nbytes)
    if out < 1:
        raise BadInputError(field, "must be at least one byte", cap_gib)
    return out


def all_finite(obj, field: str, _depth: int = 0) -> None:
    """Every number anywhere inside a JSON-like value is finite and in the exact range
    (for values that did not come through the strict JSON parser)."""
    if _depth > 64:
        raise BadInputError(field, "nesting deeper than 64 levels")
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        number(obj, field)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise BadInputError(field, "object keys must be strings", k)
            all_finite(v, field, _depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            all_finite(v, field, _depth + 1)
        return
    raise BadInputError(field, f"unsupported value type {type(obj).__name__}")


# --------------------------------------------------------------- command-line text
def parse_number(value: str, field: str, **bounds) -> float:
    """Strict decimal text -> finite float inside the bounds."""
    if not isinstance(value, str) or len(value) > _MAX_NUMBER_TEXT or not _DECIMAL.fullmatch(value):
        raise BadInputError(field, "must be a plain decimal number (nan, inf, hexadecimal, underscores and blanks "
                                   "are not accepted)", value)
    return number(float(value), field, **bounds)  # '1e999' overflows to inf and is refused there


def parse_integer(value: str, field: str, **bounds) -> int:
    """Strict integer text -> int inside the bounds and the exact range."""
    if not isinstance(value, str) or len(value) > 20 or not _INTEGER.fullmatch(value):
        raise BadInputError(field, "must be a plain integer", value)
    return integer(int(value), field, **bounds)


def parse_optional_number(value: str | None, field: str, **bounds) -> float | None:
    return None if value is None else parse_number(value, field, **bounds)


def parse_optional_integer(value: str | None, field: str, **bounds) -> int | None:
    return None if value is None else parse_integer(value, field, **bounds)
