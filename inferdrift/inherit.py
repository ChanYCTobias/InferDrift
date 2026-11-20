# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact grade inheritance.

* Deterministic graders: a candidate output byte-identical to the reference
  output inherits the reference grade without a grading call.  Because a grade is
  a pure function of (item, output), the inherited grade equals the grade a fresh
  call would return; a seeded audit re-grades a fixed share of inherited outputs
  to check that this holds in practice (a mismatch is reported and the fresh grade
  wins).  Audit calls are counted as overhead.
* Judge graders: a frozen judge model's recorded verdict on an identical
  (prompt, output) pair is reused rather than re-requested (``JudgeGrader``
  keeps the cache, optionally on disk across runs).  Judge verdicts are never
  re-requested for audit.

Only divergent outputs are graded, so grading cost scales with churn, not with
the bank.  (Generation savings are a separate concern.)
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import Counter

from . import domains
from .adapters.base import DecodeConfig, GenRequest
from .errors import BadInputError
from .graders import DETERMINISTIC_TYPES, Grade, grade
from .jsonio import StrictJSONError, canonical, read_jsonl_strict, sha256_text

_KEY_CHARS = set("0123456789abcdef")


def load_judge_cache(path: str | None) -> dict[str, dict]:
    """Recorded judge verdicts {key: {"correct": bool, "reason": str}} from a JSONL cache,
    parsed strictly (duplicate keys and non-finite constants are refused, as is any entry
    outside the schema).  A path that does not exist yet is an empty cache."""
    if not path or not os.path.exists(path):
        return {}
    try:
        rows = read_jsonl_strict(path)
    except StrictJSONError as e:
        raise BadInputError(f"judge cache {path}", f"is not strict JSON ({e})") from None
    except (OSError, UnicodeDecodeError) as e:
        raise BadInputError(f"judge cache {path}", f"cannot be read ({type(e).__name__})") from None
    cache: dict[str, dict] = {}
    for no, e in rows:
        where = f"judge cache {path}:{no}"
        if not isinstance(e, dict) or set(e) != {"key", "grade"}:
            raise BadInputError(where, "each line must be {\"key\": ..., \"grade\": {...}}")
        k, g = e["key"], e["grade"]
        if not isinstance(k, str) or len(k) != 64 or not set(k) <= _KEY_CHARS:
            raise BadInputError(where, "key must be 64 lowercase hex characters")
        if not isinstance(g, dict) or set(g) != {"correct", "reason"} or not isinstance(g["correct"], bool) \
                or not isinstance(g["reason"], str):
            raise BadInputError(where, "grade must be {\"correct\": true|false, \"reason\": str}")
        cache[k] = g
    return cache

JUDGE_TEMPLATE = (
    "You are grading an answer.\nQuestion:\n{question}\n\nReference criteria:\n{rubric}\n\nAnswer to grade:\n{output}"
    "\n\nReply with exactly one word: YES if the answer meets the criteria, otherwise NO."
)


class JudgeGrader:
    """Frozen judge with a verdict cache keyed by (judge, template, prompt, output)."""

    def __init__(self, adapter, meter, *, judge_digest: str, template: str = JUDGE_TEMPLATE,
                 cache_path: str | None = None) -> None:
        self.adapter = adapter
        self.meter = meter
        self.judge_digest = judge_digest
        self.template = template
        self.template_digest = sha256_text(template)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = load_judge_cache(cache_path)
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()

    def key(self, prompt: str, output: str) -> str:
        return sha256_text(canonical({"judge": self.judge_digest, "template": self.template_digest,
                                      "prompt": prompt, "output": output}))

    def grade(self, item: dict, output: str) -> tuple[Grade, bool]:
        prompt = canonical(item["messages"])
        k = self.key(prompt, output)
        with self._lock:
            if k in self.cache:
                self.hits += 1
                g = self.cache[k]
                return Grade(g["correct"], g["reason"], "judge"), False
        question = "\n".join(m.get("content", "") for m in item["messages"] if m.get("role") == "user")
        text = self.template.format(question=question, rubric=item["grader"].get("rubric", ""), output=output)
        req = GenRequest(f"judge-{k[:16]}", [{"role": "user", "content": text}], None,
                         {"enable_thinking": False}, 4, DecodeConfig("greedy"))
        res = self.adapter.generate([req], self.meter)[0]
        verdict = (res.text or "").strip().upper()
        ok = verdict.startswith("YES")
        g = Grade(ok, "JUDGE_YES" if ok else ("JUDGE_NO" if verdict.startswith("NO") else "JUDGE_UNPARSED"), "judge")
        with self._lock:
            self.calls += 1
            self.cache[k] = {"correct": g.correct, "reason": g.reason}
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(canonical({"key": k, "grade": self.cache[k]}) + "\n")
        return g, True


class GradeBook:
    def __init__(self, *, audit_fraction: float = 0.05, seed: int = 0, judge: JudgeGrader | None = None,
                 sandbox_weak_ok: bool = False) -> None:
        self.audit_fraction = domains.number(audit_fraction, "audit_fraction", ge=0.0, le=1.0)
        self.seed = domains.integer(seed, "seed", ge=0)
        self.judge = judge
        self.sandbox_weak_ok = sandbox_weak_ok
        self.calls = Counter()
        self.audit_mismatches: list[dict] = []

    def audit_selected(self, item_id: str) -> bool:
        h = int.from_bytes(hashlib.sha256(f"audit|{self.seed}|{item_id}".encode()).digest()[:8], "big")
        return h < self.audit_fraction * 2 ** 64

    def _fresh(self, item: dict, output: str) -> tuple[Grade, str]:
        if item["grader"]["type"] == "judge":
            if self.judge is None:
                raise ValueError("judge-graded item but no judge configured")
            g, called = self.judge.grade(item, output)
            return g, ("judge_call" if called else "judge_cache_hit")
        return grade(item, output, sandbox_weak_ok=self.sandbox_weak_ok), "fresh"

    def grade_reference(self, item: dict, output: str) -> Grade:
        g, how = self._fresh(item, output)
        self.calls["reference_calls" if how in ("fresh", "judge_call") else "reference_judge_cache_hits"] += 1
        return g

    def grade_reference_cached(self, item: dict, output: str, cached: Grade) -> Grade:
        """A reference grade inherited from the reference cache (made by the same grader).

        Deterministic graders: a seeded share (``audit_fraction``) is graded again; a
        mismatch is recorded and the fresh grade wins.  Judge verdicts are reused
        as recorded (they are never re-requested for audit)."""
        if item["grader"]["type"] != "judge" and self.audit_fraction > 0 and self.audit_selected(item["id"]):
            g, _ = self._fresh(item, output)
            self.calls["reference_cache_audit_calls"] += 1
            if g != cached:
                self.audit_mismatches.append({"item": item["id"], "side": "reference_cache",
                                              "inherited": cached.to_dict(), "fresh": g.to_dict()})
                return g
            return cached
        self.calls["reference_cache_inherited"] += 1
        return cached

    def grade_follower(self, item: dict, ref_output: str, ref_grade: Grade, output: str,
                       side: str = "candidate") -> tuple[Grade, str, int]:
        """Grade a candidate (or replicate) output; return (grade, how, grading_calls_spent)."""
        gtype = item["grader"]["type"]
        if gtype == "judge":
            g, how = self._fresh(item, output)
            spent = 1 if how == "judge_call" else 0
            self.calls[f"{side}_{how}"] += 1
            return g, how, spent
        if gtype not in DETERMINISTIC_TYPES:
            raise ValueError(f"unknown grader type {gtype!r}")
        if output == ref_output:
            if self.audit_fraction > 0 and self.audit_selected(item["id"]):
                g, _ = self._fresh(item, output)
                self.calls[f"{side}_audit_calls"] += 1
                if g != ref_grade:
                    self.audit_mismatches.append({"item": item["id"], "side": side, "inherited": ref_grade.to_dict(),
                                                  "fresh": g.to_dict()})
                    return g, "audit_mismatch", 1
                return ref_grade, "inherited_audited", 1
            self.calls[f"{side}_inherited"] += 1
            return ref_grade, "inherited", 0
        g, _ = self._fresh(item, output)
        self.calls[f"{side}_fresh_calls"] += 1
        return g, "fresh", 1
