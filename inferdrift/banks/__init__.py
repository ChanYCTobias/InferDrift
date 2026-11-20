# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable item banks.

A bank is a list of items.  ``load_bank(spec)`` accepts

* a bundle name: ``core`` (short math, multiple choice, tool calling, instruction
  following, unit-tested code; 500 items), ``full`` (core plus the 16k-token
  needle suite), ``smoke`` (the first 8 items of each core suite);
* a comma-separated list of suite names (``math_short,tool_calling``);
* the path of a JSONL file with one item per line (your own bank).

Item schema (closed: unknown keys, wrong types and out-of-range numbers are refused)::

    {"id": str, "suite": str, "variant": str (optional),
     "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str}...],
     "tools": [{..}...] | null (optional), "template_kwargs": {str: scalar} | null (optional),
     "max_new_tokens": int in [1, 1048576], "grader": {"type": ..., <type's own fields>}}

Bank files are read with the strict JSON parser: a duplicate key, a
NaN/Infinity constant or a number that overflows to infinity is refused with the
file and line, never coerced.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from .. import domains
from ..errors import BadInputError
from ..graders import DETERMINISTIC_TYPES
from ..jsonio import StrictJSONError, canonical, loads_strict, read_jsonl_strict, sha256_text

CORE_SUITES = ("math_short", "multiple_choice", "tool_calling", "instruction_following", "code_unit")
ALL_SUITES = CORE_SUITES + ("needle_16k",)
BUNDLES = {"core": CORE_SUITES, "full": ALL_SUITES}
REQUIRED = ("id", "suite", "messages", "max_new_tokens", "grader")
OPTIONAL = ("variant", "tools", "template_kwargs")
MAX_NEW_TOKENS = 1 << 20
ROLES = ("system", "user", "assistant", "tool")
GRADER_FIELDS = {  # type -> (required fields, optional fields)
    "exact": (("answers",), ("normalize",)),
    "choice": (("answer",), ("letters",)),
    "numeric": (("answer",), ("tol",)),
    "tool_call": (("name", "arguments"), ("parameters",)),
    "json_schema": (("schema",), ()),
    "instruction": (("checks",), ()),
    "unit_tests": (("tests",), ("timeout_s",)),
    "judge": ((), ("rubric",)),
}
assert set(GRADER_FIELDS) == set(DETERMINISTIC_TYPES) | {"judge"}, "grader schema out of step with graders.py"
CHECK_FIELDS = {  # instruction check type -> its parameter (or None)
    "lowercase": None, "uppercase": None, "no_commas": None, "wrap_quotes": None,
    "max_words": "n", "min_words": "n", "bullet_count": "n", "max_sentences": "n",
    "include_keywords": "keywords", "exclude_keywords": "keywords", "json_keys": "keys",
    "start_with": "prefix", "end_with": "suffix",
}


def _items_dir():
    return resources.files(__package__).joinpath("items")


def load_suite(name: str) -> list[dict]:
    if name not in ALL_SUITES:
        raise BadInputError("bank", f"unknown suite; known: {', '.join(ALL_SUITES)}", name)
    text = _items_dir().joinpath(f"{name}.jsonl").read_text(encoding="utf-8")
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                out.append(loads_strict(line))
            except StrictJSONError as e:
                raise BadInputError(f"bundled suite {name}:{no}", f"is not strict JSON ({e})") from None
    return out


def _strs(value, where: str, *, non_empty: bool = True) -> list:
    if not isinstance(value, list) or (non_empty and not value) or not all(isinstance(x, str) for x in value):
        raise BadInputError(where, "must be a non-empty list of strings" if non_empty else "must be a list of strings")
    return value


def _closed(obj: dict, required, optional, where: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise BadInputError(where, f"missing required field(s) {missing}")
    unknown = sorted(set(obj) - set(required) - set(optional))
    if unknown:
        raise BadInputError(where, f"unknown field(s) {unknown[:5]}; allowed: {sorted(set(required) | set(optional))}")


def validate_grader(g, where: str) -> None:
    if not isinstance(g, dict):
        raise BadInputError(where, "must be an object")
    gtype = g.get("type")
    if gtype not in GRADER_FIELDS:
        raise BadInputError(f"{where}.type", f"unknown grader type; known: {', '.join(GRADER_FIELDS)}", gtype)
    req, opt = GRADER_FIELDS[gtype]
    _closed(g, ("type",) + req, opt, where)
    if gtype == "exact":
        _strs(g["answers"], f"{where}.answers")
        if "normalize" in g:
            domains.choice(g["normalize"], f"{where}.normalize", ("strip", "casefold", "alnum"))
    elif gtype == "choice":
        letters = g.get("letters", "ABCD")
        if not isinstance(letters, str) or not 1 <= len(letters) <= 26 or len(set(letters)) != len(letters):
            raise BadInputError(f"{where}.letters", "must be 1 to 26 distinct characters", letters)
        if not isinstance(g["answer"], str) or len(g["answer"]) != 1 or g["answer"] not in letters:
            raise BadInputError(f"{where}.answer", "must be one of the letters", g["answer"])
    elif gtype == "numeric":
        domains.number(g["answer"], f"{where}.answer")
        if "tol" in g:
            domains.number(g["tol"], f"{where}.tol", ge=0.0)
    elif gtype == "tool_call":
        domains.text(g["name"], f"{where}.name", max_len=256)
        if not isinstance(g["arguments"], dict):
            raise BadInputError(f"{where}.arguments", "must be an object")
        if g.get("parameters") is not None and not isinstance(g["parameters"], dict):
            raise BadInputError(f"{where}.parameters", "must be an object or null")
    elif gtype == "json_schema":
        if not isinstance(g["schema"], dict):
            raise BadInputError(f"{where}.schema", "must be an object")
    elif gtype == "instruction":
        checks = g["checks"]
        if not isinstance(checks, list) or not checks:
            raise BadInputError(f"{where}.checks", "must be a non-empty list")
        for i, chk in enumerate(checks):
            cw = f"{where}.checks[{i}]"
            if not isinstance(chk, dict) or chk.get("type") not in CHECK_FIELDS:
                raise BadInputError(f"{cw}.type", f"unknown check; known: {', '.join(CHECK_FIELDS)}",
                                    chk.get("type") if isinstance(chk, dict) else chk)
            param = CHECK_FIELDS[chk["type"]]
            _closed(chk, ("type",) + ((param,) if param else ()), (), cw)
            if param == "n":
                domains.integer(chk["n"], f"{cw}.n", ge=0 if chk["type"] == "bullet_count" else 1)
            elif param in ("keywords", "keys"):
                _strs(chk[param], f"{cw}.{param}")
            elif param in ("prefix", "suffix"):
                domains.text(chk[param], f"{cw}.{param}")
    elif gtype == "unit_tests":
        domains.text(g["tests"], f"{where}.tests", max_len=1 << 20)
        if "timeout_s" in g:
            domains.number(g["timeout_s"], f"{where}.timeout_s", gt=0.0, le=600.0)
    elif gtype == "judge":
        if "rubric" in g and not isinstance(g["rubric"], str):
            raise BadInputError(f"{where}.rubric", "must be a string", g["rubric"])


def validate_item(it, where: str) -> None:
    if not isinstance(it, dict):
        raise BadInputError(where, "must be an object")
    _closed(it, REQUIRED, OPTIONAL, where)
    domains.text(it["id"], f"{where}.id", max_len=512)
    domains.text(it["suite"], f"{where}.suite", max_len=256)
    if "variant" in it and not isinstance(it["variant"], str):
        raise BadInputError(f"{where}.variant", "must be a string", it["variant"])
    msgs = it["messages"]
    if not isinstance(msgs, list) or not msgs:
        raise BadInputError(f"{where}.messages", "must be a non-empty list")
    for i, m in enumerate(msgs):
        mw = f"{where}.messages[{i}]"
        if not isinstance(m, dict):
            raise BadInputError(mw, "must be an object")
        _closed(m, ("role", "content"), (), mw)
        domains.choice(m["role"], f"{mw}.role", ROLES)
        if not isinstance(m["content"], str):
            raise BadInputError(f"{mw}.content", "must be a string")
    tools = it.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
        raise BadInputError(f"{where}.tools", "must be a list of objects or null")
    tk = it.get("template_kwargs")
    if tk is not None:
        if not isinstance(tk, dict) or not all(isinstance(v, (str, bool, int, float)) or v is None
                                               for v in tk.values()):
            raise BadInputError(f"{where}.template_kwargs", "must be an object of scalar values or null")
    domains.integer(it["max_new_tokens"], f"{where}.max_new_tokens", ge=1, le=MAX_NEW_TOKENS)
    validate_grader(it["grader"], f"{where}.grader")
    domains.all_finite(it, where)  # numbers anywhere (schemas, arguments) are finite


def validate_items(items: list[dict]) -> None:
    if not isinstance(items, list):
        raise BadInputError("bank", "must be a list of items")
    seen = set()
    for i, it in enumerate(items):
        where = f"bank item {i}" + (f" ({it.get('id')!r})" if isinstance(it, dict) and isinstance(it.get("id"), str)
                                    else "")
        validate_item(it, where)
        if it["id"] in seen:
            raise BadInputError(where, "duplicate item id", it["id"])
        seen.add(it["id"])


def load_bank_file(path: str | os.PathLike) -> list[dict]:
    """Items of a user bank file (JSONL), parsed strictly and validated."""
    try:
        rows = read_jsonl_strict(path)
    except StrictJSONError as e:
        raise BadInputError(f"bank file {path}", f"is not strict JSON ({e})") from None
    except FileNotFoundError:
        raise BadInputError(f"bank file {path}", "does not exist") from None
    except (OSError, UnicodeDecodeError) as e:
        raise BadInputError(f"bank file {path}", f"cannot be read ({type(e).__name__})") from None
    items = []
    for no, it in rows:
        validate_item(it, f"bank file {path}:{no}")
        items.append(it)
    return items


def load_bank(spec: str) -> list[dict]:
    if not isinstance(spec, str) or not spec.strip():
        raise BadInputError("bank", "must be a bundle, suite list or JSONL path", spec)
    if spec in BUNDLES:
        items = [it for s in BUNDLES[spec] for it in load_suite(s)]
    elif spec == "smoke":
        items = [it for s in CORE_SUITES for it in load_suite(s)[:8]]
    elif os.path.sep in spec or spec.endswith(".jsonl"):
        items = load_bank_file(Path(spec))
    else:
        items = [it for s in spec.split(",") for it in load_suite(s.strip())]
    if not items:
        raise BadInputError("bank", "has no items", spec)
    validate_items(items)
    return items


def bank_digest(items: list[dict]) -> str:
    return sha256_text("\n".join(canonical(it) for it in items))


def regenerate(out_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """Write every default suite with the generators; return {suite: sha256}."""
    from .generators import generate_suite

    out = Path(out_dir) if out_dir else Path(str(_items_dir()))
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ALL_SUITES:
        text = "".join(canonical(it) + "\n" for it in generate_suite(name))
        (out / f"{name}.jsonl").write_text(text, encoding="utf-8")
        digests[name] = sha256_text(text)
    return digests
