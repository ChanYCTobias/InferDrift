# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Reference cache: reuse a reference stack's outputs across candidates.

Opt in with ``compare --ref-cache DIR``.  An entry lives in ``DIR/<key>/`` where
the key is the sha256 of the canonical JSON of: the cache format, the tool
version, the reference profile digest and spec, the decode settings, how a
replicate request is built (the same request again, or a fresh seed), and the
digest of the runnable bank.  It does not include the candidate, the item
order, ``--max-items``, tau or alpha, so every candidate and every subset of
the bank reuses it.

* ``outputs-<sha256>.jsonl`` (content-addressed): one line per (item,
  replicate) with the output, its grade and the grader identity the grade was
  made with (the judge's profile and template digests for judge-graded items,
  the sandbox mode for unit-tested items).  Errored outputs are never cached.
* ``meta.json``, replaced atomically after the outputs file: the outputs file's
  name and sha256, per-replicate counts and id digests, the reference's channel
  report.

Loading is strict: a mismatch between ``meta.json`` and the outputs file, a
duplicate line, a line for an item whose content changed, or any malformed
value refuses the run (``CACHE_CORRUPT``, exit 3); a cache that claims outputs
it does not hold is refused, never silently regenerated.  A missing entry is a
miss.  When the outputs file named by ``meta.json`` is missing (a concurrent
writer replaced it), ``meta.json`` is read once more, and if the file is still
missing the entry is treated as a miss.  Two concurrent writers: the last
``meta.json`` wins, and both leave a valid entry.

Grades are inherited only when the grader identity matches; otherwise the
cached output is graded again (a grading call, never a generation).
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import RefusedError
from .jsonio import StrictJSONError, atomic_write_text, canonical, loads_strict, sha256_text

FORMAT = "inferdrift/refcache/1"
META_KEYS = {"format", "key", "components", "outputs_file", "outputs_sha256", "counts", "id_digests",
             "reference_channels", "updated_utc"}
LINE_KEYS = {"id", "item_sha256", "replicate", "side", "grade", "grade_identity", "tokens"}
SIDE_KEYS = {"text", "finish", "prompt_tokens", "completion_tokens", "usage_source", "prompt_sha256", "elapsed_s"}


class CacheCorrupt(RefusedError):
    code = "CACHE_CORRUPT"


def item_sha256(item: dict) -> str:
    return sha256_text(canonical(item))


def cache_key(components: dict) -> str:
    return sha256_text(canonical(components))


def _ids_digest(ids) -> str:
    return sha256_text("\n".join(sorted(ids)))


def _bad(where: str, what: str):
    raise CacheCorrupt(f"reference cache entry {where}: {what}; delete the entry directory to rebuild it")


def _check_line(obj, no: int, bank: dict, where: str) -> tuple[tuple[str, int], dict]:
    w = f"{where} line {no}"
    if not isinstance(obj, dict) or set(obj) != LINE_KEYS:
        _bad(w, "line is not a cache record")
    if not isinstance(obj["id"], str) or obj["id"] not in bank:
        _bad(w, "item id is not in the current bank")
    if obj["item_sha256"] != item_sha256(bank[obj["id"]]):
        _bad(w, "the item's content differs from the cached item")
    if obj["replicate"] not in (0, 1) or isinstance(obj["replicate"], bool):
        _bad(w, "replicate must be 0 or 1")
    side = obj["side"]
    if not isinstance(side, dict) or set(side) != SIDE_KEYS or not isinstance(side["text"], str) \
            or side["finish"] == "error" or not isinstance(side["finish"], str):
        _bad(w, "output record is malformed or an error")
    g = obj["grade"]
    if not isinstance(g, dict) or set(g) != {"correct", "reason", "grader"} or not isinstance(g["correct"], bool) \
            or not isinstance(g["reason"], str) or not isinstance(g["grader"], str):
        _bad(w, "grade is malformed")
    if not isinstance(obj["grade_identity"], dict):
        _bad(w, "grade identity is malformed")
    if obj["tokens"] is not None and not (isinstance(obj["tokens"], list) and
                                          all(isinstance(t, int) and not isinstance(t, bool) for t in obj["tokens"])):
        _bad(w, "tokens must be a list of integers or null")
    return (obj["id"], obj["replicate"]), obj


class RefCache:
    def __init__(self, root: str, components: dict, bank_items: list[dict]) -> None:
        self.root = Path(root)
        self.components = dict(components, format=FORMAT)
        self.key = cache_key(self.components)
        self.dir = self.root / self.key
        self.bank = {it["id"]: it for it in bank_items}
        self.entries: dict[tuple[str, int], dict] = {}
        self.added: dict[tuple[str, int], dict] = {}
        self.meta: dict | None = None
        self.entry_found = False
        self.hits = {"replicate_0": 0, "replicate_1": 0}
        self.misses = 0
        self.notes: list[str] = []
        self.written = False
        self.write_error: str | None = None

    # ------------------------------------------------------------------ load
    def _read_meta(self) -> dict | None:
        path = self.dir / "meta.json"
        if not path.exists():
            return None
        try:
            meta = loads_strict(path.read_bytes())
        except (StrictJSONError, OSError) as e:
            _bad(str(self.dir), f"meta.json is not strict JSON ({type(e).__name__})")
        if not isinstance(meta, dict) or set(meta) != META_KEYS or meta["format"] != FORMAT or meta["key"] != self.key:
            _bad(str(self.dir), "meta.json does not match this cache format or key")
        if canonical(meta["components"]) != canonical(self.components):
            _bad(str(self.dir), "meta.json components differ from the key's components")
        name = meta["outputs_file"]
        if not isinstance(name, str) or os.sep in name or not name.startswith("outputs-") or not name.endswith(".jsonl"):
            _bad(str(self.dir), "meta.json names an invalid outputs file")
        for field in ("counts", "id_digests"):
            if not isinstance(meta[field], dict) or set(meta[field]) != {"0", "1"}:
                _bad(str(self.dir), f"meta.json {field} is malformed")
        return meta

    def load(self) -> None:
        meta = self._read_meta()
        if meta is None:
            return
        path = self.dir / meta["outputs_file"]
        if not path.exists():
            meta = self._read_meta()  # a concurrent writer may have replaced meta.json just now
            path = self.dir / meta["outputs_file"] if meta else path
            if meta is None or not path.exists():
                self.notes.append("the outputs file named by meta.json is missing (replaced by a concurrent writer?); "
                                  "treated as a miss")
                return
        data = path.read_bytes()
        if sha256_text(data.decode("utf-8", errors="replace")) != meta["outputs_sha256"] or \
                path.name != f"outputs-{meta['outputs_sha256']}.jsonl":
            _bad(str(self.dir), "the outputs file's sha256 differs from meta.json")
        entries = {}
        for no, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = loads_strict(line)
            except StrictJSONError:
                _bad(str(self.dir), f"line {no} is not strict JSON")
            k, rec = _check_line(obj, no, self.bank, str(self.dir))
            if k in entries:
                _bad(str(self.dir), f"duplicate record for {k}")
            entries[k] = rec
        for rep in (0, 1):
            ids = [i for (i, r) in entries if r == rep]
            if len(ids) != meta["counts"][str(rep)] or _ids_digest(ids) != meta["id_digests"][str(rep)]:
                _bad(str(self.dir), f"replicate {rep} holds {len(ids)} records, not the {meta['counts'][str(rep)]} "
                                    f"that meta.json claims")
        self.entries = entries
        self.meta = meta
        self.entry_found = True

    # ------------------------------------------------------------------ use
    def get(self, item_id: str, replicate: int) -> dict | None:
        rec = self.entries.get((item_id, replicate)) or self.added.get((item_id, replicate))
        if rec is None:
            self.misses += 1
            return None
        self.hits[f"replicate_{replicate}"] += 1
        return rec

    def channels(self) -> dict | None:
        return (self.meta or {}).get("reference_channels")

    def put(self, item: dict, replicate: int, side: dict, grade: dict, identity: dict, tokens) -> None:
        if side.get("finish") == "error" or side.get("error"):
            return  # errors are never cached
        key = (item["id"], replicate)
        if key in self.entries:
            return
        self.added[key] = {"id": item["id"], "item_sha256": item_sha256(item), "replicate": replicate,
                           "side": {k: side.get(k) for k in SIDE_KEYS}, "grade": dict(grade),
                           "grade_identity": dict(identity),
                           "tokens": tokens if isinstance(tokens, list) and all(
                               isinstance(t, int) and not isinstance(t, bool) for t in tokens) else None}

    # ------------------------------------------------------------------ write
    def write(self, channels: dict | None, utc: str) -> None:
        if not self.added and self.entry_found and (channels is None or self.channels() is not None):
            return
        try:
            allrec = dict(self.entries)
            allrec.update(self.added)
            lines = [canonical(allrec[k]) + "\n" for k in sorted(allrec)]
            text = "".join(lines)
            sha = sha256_text(text)
            name = f"outputs-{sha}.jsonl"
            self.dir.mkdir(parents=True, exist_ok=True)
            if not (self.dir / name).exists():
                atomic_write_text(self.dir / name, text)
            counts = {str(r): sum(1 for (_, rr) in allrec if rr == r) for r in (0, 1)}
            digests = {str(r): _ids_digest([i for (i, rr) in allrec if rr == r]) for r in (0, 1)}
            old = (self.meta or {}).get("outputs_file")
            meta = {"format": FORMAT, "key": self.key, "components": self.components, "outputs_file": name,
                    "outputs_sha256": sha, "counts": counts, "id_digests": digests,
                    "reference_channels": channels if channels is not None else self.channels(), "updated_utc": utc}
            atomic_write_text(self.dir / "meta.json", canonical(meta) + "\n")
            if old and old != name:
                (self.dir / old).unlink(missing_ok=True)
            self.written = True
        except OSError as e:
            self.write_error = f"{type(e).__name__}: {e}"

    def report(self, reference_generations: int) -> dict:
        return {"enabled": True, "key": self.key, "entry_found": self.entry_found, "hits": dict(self.hits),
                "misses": self.misses, "added": len(self.added), "reference_generations": reference_generations,
                "written": self.written, "write_error": self.write_error, "notes": list(self.notes),
                "components": self.components}
