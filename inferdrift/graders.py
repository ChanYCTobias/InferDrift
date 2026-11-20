# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic graders.

A grade is a pure function of (item, output text): no stack-specific input
(prompt rendering, token ids, finish reason, timing) reaches a grader.  That is
what makes exact grade inheritance sound: an output byte-identical to the
reference output must receive the reference output's grade.

Grader types (``item['grader']['type']``):

* ``exact``        - normalised string equality against one or more answers
* ``choice``       - multiple choice letter extraction
* ``numeric``      - final number (after 'Answer:' when present) within a tolerance
* ``tool_call``    - JSON tool call: name, arguments, and the tool's parameter schema
* ``json_schema``  - a JSON object that validates against a schema subset
* ``instruction``  - verifiable instruction checks, all of which must hold
* ``unit_tests``   - code extracted from the output runs against tests in the
                     network-less sandbox (``sandbox.py``)
* ``judge``        - a frozen judge model's verdict, cached per (prompt, output);
                     see ``inherit.JudgeGrader``

Model outputs are the data under test, not inputs that control the tool: the
JSON and numbers a grader extracts from an output are read the way a reader of
that output would read them, and only ever decide that output's grade.  Grader
*parameters* come from the item bank, whose closed schema (``banks``) keeps
them finite and in range.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

GRADER_VERSION = "inferdrift-graders/1"
DETERMINISTIC_TYPES = ("exact", "choice", "numeric", "tool_call", "json_schema", "instruction", "unit_tests")


@dataclass(frozen=True)
class Grade:
    correct: bool
    reason: str
    grader: str

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- helpers
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _to_number(s: str) -> float | None:
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Number after the last 'Answer:' (or 'answer is'), else the last number in the text."""
    m = list(re.finditer(r"(?i)\banswer\s*(?:is|:)\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", text))
    if m:
        return _to_number(m[-1].group(1))
    nums = _NUM.findall(text)
    if not nums:
        return None
    return _to_number(nums[-1])


def extract_choice(text: str, letters: str = "ABCD") -> str | None:
    t = text.strip()
    cls = "[" + re.escape(letters) + "]"
    m = re.search(r"(?i)\banswer\s*(?:is|:)?\s*\(?\**(" + cls + r")\**\)?(?![A-Za-z])", t)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"\(?\**(" + cls + r")\**\)?[.):]?", t)
    if m:
        return m.group(1)
    m = re.search(r"(?<![A-Za-z])(" + cls + r")(?![A-Za-z])", t)
    if m:
        return m.group(1)
    return None


def extract_json_object(text: str) -> object | None:
    """First balanced JSON object in the text (code fences allowed)."""
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        obj = extract_json_object(m.group(1))
        if isinstance(obj, dict):
            calls.append(obj)
    if not calls:
        obj = extract_json_object(text)
        if isinstance(obj, dict):
            calls.append(obj)
    return calls


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1)
    return text


# ----------------------------------------------------------------- schema validation
def validate_schema(value: object, schema: dict, path: str = "$") -> str | None:
    """Validate a JSON value against a JSON-schema subset; return an error or None."""
    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types:
        ok = False
        for ty in types:
            if ty == "object" and isinstance(value, dict):
                ok = True
            elif ty == "array" and isinstance(value, list):
                ok = True
            elif ty == "string" and isinstance(value, str):
                ok = True
            elif ty == "integer" and isinstance(value, int) and not isinstance(value, bool):
                ok = True
            elif ty == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                ok = True
            elif ty == "boolean" and isinstance(value, bool):
                ok = True
            elif ty == "null" and value is None:
                ok = True
        if not ok:
            return f"{path}: expected {types}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: not in enum"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                return f"{path}: missing {req}"
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    return f"{path}: unexpected {k}"
        for k, sub in props.items():
            if k in value:
                err = validate_schema(value[k], sub, f"{path}.{k}")
                if err:
                    return err
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            err = validate_schema(v, schema["items"], f"{path}[{i}]")
            if err:
                return err
    return None


def _norm_value(v: object) -> object:
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


# ----------------------------------------------------------------- instruction checks
def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _check_instruction(check: dict, text: str) -> bool:
    t = text.strip()
    kind = check["type"]
    if kind == "lowercase":
        return any(c.isalpha() for c in t) and t == t.lower()
    if kind == "uppercase":
        return any(c.isalpha() for c in t) and t == t.upper()
    if kind == "max_words":
        return 0 < len(_words(t)) <= check["n"]
    if kind == "min_words":
        return len(_words(t)) >= check["n"]
    if kind == "include_keywords":
        low = t.lower()
        return all(k.lower() in low for k in check["keywords"])
    if kind == "exclude_keywords":
        low = t.lower()
        return all(k.lower() not in low for k in check["keywords"])
    if kind == "no_commas":
        return len(t) > 0 and "," not in t
    if kind == "bullet_count":
        bullets = [ln for ln in t.splitlines() if re.match(r"^\s*[-*]\s+\S", ln)]
        return len(bullets) == check["n"]
    if kind == "start_with":
        return t.startswith(check["prefix"])
    if kind == "end_with":
        return t.endswith(check["suffix"])
    if kind == "wrap_quotes":
        return len(t) >= 2 and t[0] == '"' and t[-1] == '"'
    if kind == "json_keys":
        obj = extract_json_object(t)
        return isinstance(obj, dict) and all(k in obj for k in check["keys"])
    if kind == "max_sentences":
        sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", t) if s.strip()]
        return 0 < len(sentences) <= check["n"]
    raise ValueError(f"unknown instruction check {kind!r}")


# ------------------------------------------------------------------------- dispatch
def grade(item: dict, output: str, *, sandbox_weak_ok: bool = False) -> Grade:
    """Grade one output.  Pure function of (item, output) for deterministic types."""
    g = item["grader"]
    kind = g["type"]
    out = output if isinstance(output, str) else ""
    if kind == "exact":
        answers = g["answers"]
        norm = g.get("normalize", "strip")

        def n(s: str) -> str:
            s = s.strip()
            if norm in ("casefold", "alnum"):
                s = s.casefold()
            if norm == "alnum":
                s = re.sub(r"[^0-9a-z]", "", s)
            return s.rstrip(".") if norm == "strip" else s

        ok = n(out) in {n(a) for a in answers}
        return Grade(ok, "MATCH" if ok else "MISMATCH", kind)
    if kind == "choice":
        got = extract_choice(out, g.get("letters", "ABCD"))
        if got is None:
            return Grade(False, "NO_CHOICE", kind)
        ok = got == g["answer"]
        return Grade(ok, "MATCH" if ok else "WRONG_CHOICE", kind)
    if kind == "numeric":
        got = extract_final_number(out)
        if got is None:
            return Grade(False, "NO_NUMBER", kind)
        tol = float(g.get("tol", 1e-6))
        ok = abs(got - float(g["answer"])) <= tol
        return Grade(ok, "MATCH" if ok else "WRONG_NUMBER", kind)
    if kind == "tool_call":
        calls = extract_tool_calls(out)
        if not calls:
            return Grade(False, "NO_TOOL_CALL", kind)
        call = calls[0]
        name = call.get("name")
        args = call.get("arguments", call.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return Grade(False, "ARGS_NOT_JSON", kind)
        if not isinstance(name, str) or not isinstance(args, dict):
            return Grade(False, "MALFORMED_CALL", kind)
        if name != g["name"]:
            return Grade(False, "WRONG_TOOL", kind)
        schema = g.get("parameters")
        if schema:
            err = validate_schema(args, schema)
            if err:
                return Grade(False, "SCHEMA_VIOLATION", kind)
        for k, v in g["arguments"].items():
            if k not in args or _norm_value(args[k]) != _norm_value(v):
                return Grade(False, "WRONG_ARGUMENT", kind)
        return Grade(True, "MATCH", kind)
    if kind == "json_schema":
        obj = extract_json_object(out)
        if obj is None:
            return Grade(False, "NO_JSON", kind)
        err = validate_schema(obj, g["schema"])
        return Grade(err is None, "VALID" if err is None else "SCHEMA_VIOLATION", kind)
    if kind == "instruction":
        for chk in g["checks"]:
            if not _check_instruction(chk, out):
                return Grade(False, f"FAILED_{chk['type'].upper()}", kind)
        return Grade(True, "ALL_CHECKS_PASS", kind)
    if kind == "unit_tests":
        from .sandbox import run_tests

        res = run_tests(extract_code(out), g["tests"], timeout_s=float(g.get("timeout_s", 10)),
                        allow_weak_isolation=sandbox_weak_ok)
        return Grade(res.passed, f"{res.reason}@{res.isolation}", kind)
    if kind == "judge":
        raise ValueError("judge items are graded by inherit.JudgeGrader")
    raise ValueError(f"unknown grader type {kind!r}")
