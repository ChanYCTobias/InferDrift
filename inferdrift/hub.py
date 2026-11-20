# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Hub access with a download planner and a transient allowance.

Only the standard library is used.  No token is ever sent (no Authorization
header), so only public repositories are reachable.  ``HF_ENDPOINT`` overrides
the Hub URL (as in huggingface_hub), which tests use to point at a local stand-in.

Before any byte is written, ``plan_downloads`` lists every file the job would
fetch (with sizes from the Hub tree API), sums them, and refuses the job when the
sum exceeds the transient allowance (default 6 GiB) or the free space under the
work directory.  Downloads stream into the job's work directory, are checked
against the size and (for LFS files) the SHA-256 the Hub publishes, and the job
removes the directory when it ends.

Closed domains: the allowance is a finite GiB count > 0 whose byte
value fits the exact range (checked before narrowing); Hub metadata is parsed
strictly, and every listed file size must be a non-negative integer, so a NaN,
negative or missing size can never shrink a plan below the allowance.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import domains
from .errors import AllowanceExceededError, FidelityError, RefusedError, UnsafeRepositoryError
from .jsonio import StrictJSONError, loads_strict

GIB = 1 << 30
DEFAULT_ALLOWANCE_GIB = 6.0
MODEL_PATTERNS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                  "chat_template.json", "*.safetensors", "model.safetensors.index.json")
TOKENIZER_PATTERNS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                      "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
                      "config.json", "generation_config.json")
USER_AGENT = "inferdrift/0.1 (passive; no token)"


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _open(url: str, *, timeout: float = 60.0, headers: dict | None = None):
    h = {"User-Agent": USER_AGENT}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - https or a configured endpoint


def _get_json(url: str, timeout: float = 60.0):
    try:
        with _open(url, timeout=timeout) as r:
            body = r.read(64 << 20)
            link = r.headers.get("Link", "")
        return loads_strict(body), link
    except (StrictJSONError, UnicodeDecodeError) as e:
        raise RefusedError(f"Hub metadata from {url} is not strict JSON ({e}); refused before any download",
                           code="HUB_MALFORMED_JSON") from None
    except urllib.error.HTTPError as e:
        raise FidelityError(f"Hub request failed with HTTP {e.code} for {url}", code="HUB_HTTP_ERROR") from None
    except urllib.error.URLError as e:
        raise FidelityError(f"Hub unreachable for {url}: {e.reason}", code="HUB_UNREACHABLE") from None


def resolve_commit(repo: str, revision: str = "main") -> str:
    q = urllib.parse.quote(revision, safe="")
    obj, _ = _get_json(f"{endpoint()}/api/models/{repo}/revision/{q}")
    sha = obj.get("sha") if isinstance(obj, dict) else None
    if not isinstance(sha, str) or len(sha) < 7:
        raise FidelityError(f"could not resolve revision {revision!r} of {repo}", code="HUB_BAD_REVISION")
    return sha


def list_files(repo: str, revision: str) -> list[dict]:
    """[{path, size, sha256 (LFS files only)}] for every file at the revision."""
    q = urllib.parse.quote(revision, safe="")
    url = f"{endpoint()}/api/models/{repo}/tree/{q}?recursive=true"
    out = []
    pages = 0
    while url and pages < 50:
        pages += 1
        obj, link = _get_json(url)
        if not isinstance(obj, list):
            raise FidelityError(f"unexpected tree listing for {repo}", code="HUB_BAD_LISTING")
        for e in obj:
            if not isinstance(e, dict) or not isinstance(e.get("path"), str):
                raise RefusedError(f"malformed tree entry in the listing of {repo}", code="HUB_BAD_LISTING")
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            if not isinstance(lfs, dict):
                raise RefusedError(f"malformed LFS record for {e['path']} in {repo}", code="HUB_BAD_LISTING")
            size = lfs.get("size", e.get("size"))
            if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= domains.MAX_EXACT_INT:
                raise RefusedError(f"the listing of {repo} gives {e['path']} the size {size!r:.40}; a size must be a "
                                   f"non-negative integer, so the download plan cannot be checked", code="HUB_BAD_LISTING")
            oid = lfs.get("oid")
            out.append({"path": e["path"], "size": size, "sha256": oid if isinstance(oid, str) else None})
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return out


def fetch_small(repo: str, revision: str, path: str, max_bytes: int = 16 << 20) -> bytes | None:
    q = urllib.parse.quote(revision, safe="")
    url = f"{endpoint()}/{repo}/resolve/{q}/{urllib.parse.quote(path)}"
    try:
        with _open(url) as r:
            data = r.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise FidelityError(f"Hub request failed with HTTP {e.code} for {repo}/{path}", code="HUB_HTTP_ERROR") from None
    except urllib.error.URLError as e:
        raise FidelityError(f"Hub unreachable for {repo}/{path}: {e.reason}", code="HUB_UNREACHABLE") from None
    if len(data) > max_bytes:
        raise FidelityError(f"{repo}/{path} larger than {max_bytes} bytes", code="HUB_FILE_TOO_LARGE")
    return data


def select(files: list[dict], patterns: tuple[str, ...]) -> list[dict]:
    return [f for f in files if any(fnmatch.fnmatch(f["path"], p) for p in patterns)]


@dataclass
class RepoRequest:
    repo: str
    revision: str
    patterns: tuple[str, ...]
    role: str


@dataclass
class DownloadPlan:
    entries: list[dict] = field(default_factory=list)
    total_bytes: int = 0
    cap_bytes: int = 0
    free_bytes: int = 0
    ok: bool = True
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def plan_downloads(requests: list[RepoRequest], work_dir: str | os.PathLike, cap_gib: float = DEFAULT_ALLOWANCE_GIB,
                   *, margin_bytes: int = GIB) -> DownloadPlan:
    """Plan every download of a job and refuse it when it would not fit.

    Raises ``AllowanceExceededError`` before anything is written; returns the plan
    otherwise.  Identical (repo, commit, file) triples are counted once."""
    plan = DownloadPlan(cap_bytes=domains.gib_to_bytes(cap_gib, "transient_cap_gib"))
    seen: set[tuple[str, str, str]] = set()
    for req in requests:
        commit = resolve_commit(req.repo, req.revision)
        files = select(list_files(req.repo, commit), req.patterns)
        entry = {"role": req.role, "repo": req.repo, "revision": req.revision, "commit": commit, "files": [],
                 "bytes": 0}
        for f in files:
            key = (req.repo, commit, f["path"])
            entry["files"].append(f)
            if key in seen:
                continue
            seen.add(key)
            entry["bytes"] += f["size"]
        plan.total_bytes += entry["bytes"]
        plan.entries.append(entry)
    wd = Path(work_dir)
    probe = wd if wd.exists() else wd.parent
    while not probe.exists():
        probe = probe.parent
    plan.free_bytes = shutil.disk_usage(probe).free
    if plan.total_bytes > plan.cap_bytes:
        plan.ok = False
        plan.reason = (f"planned downloads total {plan.total_bytes / GIB:.2f} GiB, above the transient allowance of "
                       f"{cap_gib:g} GiB per job")
    elif plan.total_bytes + margin_bytes > plan.free_bytes:
        plan.ok = False
        plan.reason = (f"planned downloads total {plan.total_bytes / GIB:.2f} GiB but only "
                       f"{plan.free_bytes / GIB:.2f} GiB is free under the work directory (1 GiB margin kept)")
    if not plan.ok:
        err = AllowanceExceededError(f"refused before any download: {plan.reason}")
        err.plan = plan  # type: ignore[attr-defined]
        raise err
    return plan


def check_remote_code(repo: str, revision: str) -> dict:
    """Passive pre-check of a Hub repository: refuse custom code before downloading weights."""
    from .passive import config_flags

    commit = resolve_commit(repo, revision)
    files = list_files(repo, commit)
    names = [f["path"] for f in files]
    cfg_raw = fetch_small(repo, commit, "config.json") if "config.json" in names else None
    tok_raw = fetch_small(repo, commit, "tokenizer_config.json") if "tokenizer_config.json" in names else None
    parsed = {}
    for name, raw in (("config.json", cfg_raw), ("tokenizer_config.json", tok_raw)):
        try:
            parsed[name] = loads_strict(raw) if raw else None
        except (StrictJSONError, UnicodeDecodeError) as e:
            # a duplicate key could hide an auto_map entry: ambiguous metadata is refused, not guessed
            raise UnsafeRepositoryError(f"{repo}: {name} is not strict JSON ({e}); refused") from None
    cfg, tok = parsed["config.json"], parsed["tokenizer_config.json"]
    flags = config_flags(cfg if isinstance(cfg, dict) else None, tok if isinstance(tok, dict) else None, names)
    return {"commit": commit, "flags": flags, "files": files, "config": cfg, "tokenizer_config": tok}


def download(entry: dict, dest: str | os.PathLike, *, timeout: float = 120.0) -> Path:
    """Download a planned entry into ``dest`` (created), verifying size and LFS SHA-256."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for f in entry["files"]:
        target = root / f["path"]
        if target.exists() and target.stat().st_size == f["size"]:
            continue
        if ".." in Path(f["path"]).parts or Path(f["path"]).is_absolute():
            raise UnsafeRepositoryError(f"unsafe file path {f['path']!r} in {entry['repo']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        url = f"{endpoint()}/{entry['repo']}/resolve/{entry['commit']}/{urllib.parse.quote(f['path'])}"
        h = hashlib.sha256()
        got = 0
        try:
            with _open(url, timeout=timeout) as r, open(part, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    got += len(chunk)
                    if got > f["size"] + (1 << 20):
                        raise FidelityError(f"{f['path']} is larger than the Hub listing says", code="HUB_SIZE_MISMATCH")
                    h.update(chunk)
                    out.write(chunk)
        except urllib.error.URLError as e:
            part.unlink(missing_ok=True)
            raise FidelityError(f"download failed for {entry['repo']}/{f['path']}: {e}", code="HUB_DOWNLOAD_FAILED") from None
        if got != f["size"]:
            part.unlink(missing_ok=True)
            raise FidelityError(f"size mismatch for {f['path']}: {got} != {f['size']}", code="HUB_SIZE_MISMATCH")
        if f.get("sha256") and h.hexdigest() != f["sha256"]:
            part.unlink(missing_ok=True)
            raise FidelityError(f"SHA-256 mismatch for {f['path']}", code="HUB_HASH_MISMATCH")
        os.replace(part, target)
    return root


def dir_bytes(path: str | os.PathLike) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total
