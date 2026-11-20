# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Passive inspection of model repositories.

Everything here reads bytes and parses JSON, safetensors headers and GGUF
metadata with the standard library.  Nothing here imports transformers,
unpickles a file, renders a chat template or imports a module found in a
repository: ``auto_map`` entries, ``*.py`` files and pickle weights are *listed
and flagged*, never executed.  This is the boundary the hostile-repository test
(``tests/test_safety.py``) exercises.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO

from .jsonio import StrictJSONError, loads_strict

PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
CODE_SUFFIXES = (".py", ".pyc", ".so", ".pth")
SMALL_JSON = ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.json", "preprocessor_config.json", "quantization_config.json")
MAX_JSON_BYTES = 16 << 20


# ------------------------------------------------------------------------ safetensors
def read_safetensors_header(f: BinaryIO, max_header: int = 64 << 20) -> dict:
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("not a safetensors file (short read)")
    (n,) = struct.unpack("<Q", raw)
    if n <= 0 or n > max_header:
        raise ValueError(f"implausible safetensors header length {n}")
    try:
        header = loads_strict(f.read(n))  # a duplicate tensor name could hide parameters
    except (StrictJSONError, UnicodeDecodeError) as e:
        raise ValueError(f"safetensors header is not strict JSON ({e})") from None
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header


def summarize_safetensors_header(header: dict) -> dict:
    dtypes: dict[str, int] = {}
    params = 0
    tensors = 0
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape", [])
        count = 1
        for d in shape:
            count *= int(d)
        params += count
        tensors += 1
        dt = str(info.get("dtype"))
        dtypes[dt] = dtypes.get(dt, 0) + count
    meta = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    return {"tensors": tensors, "parameters_stored": params, "dtype_parameters": dtypes,
            "metadata_keys": sorted(meta)[:50]}


# ------------------------------------------------------------------------------- GGUF
GGUF_VALUE_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
                    12: "<d"}
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
              26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
GGUF_FILE_TYPES = {0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0",
                   8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M",
                   13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
                   17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
                   21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S",
                   25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
                   29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16"}
_BIG_ARRAY_KEYS_HASHED = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores",
                          "tokenizer.ggml.token_type")


class _Reader:
    def __init__(self, f: BinaryIO, limit: int) -> None:
        self.f = f
        self.limit = limit
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.limit:
            raise ValueError("GGUF metadata exceeds the read limit")
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF file")
        self.pos += n
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > (64 << 20):
            raise ValueError("implausible GGUF string length")
        return self.read(n).decode("utf-8", errors="replace")


def _gguf_value(r: _Reader, vtype: int, key: str, hasher=None):
    if vtype in GGUF_VALUE_TYPES:
        return r.unpack(GGUF_VALUE_TYPES[vtype])
    if vtype == 8:
        return r.string()
    if vtype == 9:
        etype = r.unpack("<I")
        n = r.unpack("<Q")
        if n > 10_000_000:
            raise ValueError("implausible GGUF array length")
        if key in _BIG_ARRAY_KEYS_HASHED:
            h = hashlib.sha256()
            for _ in range(n):
                v = _gguf_value(r, etype, key)
                h.update(repr(v).encode("utf-8"))
                h.update(b"\x00")
            return {"array_len": n, "sha256": h.hexdigest()}
        vals = [_gguf_value(r, etype, key) for _ in range(n)]
        return vals if n <= 64 else {"array_len": n, "head": vals[:8]}
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(f: BinaryIO, limit: int = 256 << 20, with_tensors: bool = True) -> dict:
    r = _Reader(f, limit)
    if r.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.unpack("<I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    if n_kv > 100_000 or n_tensors > 1_000_000:
        raise ValueError("implausible GGUF counts")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.unpack("<I")
        kv[key] = _gguf_value(r, vtype, key)
    out: dict = {"gguf_version": version, "n_tensors": n_tensors, "metadata": kv}
    if with_tensors:
        types: dict[str, int] = {}
        params = 0
        for _ in range(n_tensors):
            r.string()
            nd = r.unpack("<I")
            if nd > 8:
                raise ValueError("implausible tensor rank")
            count = 1
            for _ in range(nd):
                count *= r.unpack("<Q")
            t = r.unpack("<I")
            r.unpack("<Q")
            tn = GGML_TYPES.get(t, f"type_{t}")
            types[tn] = types.get(tn, 0) + count
            params += count
        out["tensor_type_parameters"] = types
        out["parameters_stored"] = params
    return out


def summarize_gguf(meta: dict) -> dict:
    kv = meta.get("metadata", {})
    ft = kv.get("general.file_type")
    tmpl = kv.get("tokenizer.chat_template")
    toks = kv.get("tokenizer.ggml.tokens")
    return {
        "architecture": kv.get("general.architecture"),
        "name": kv.get("general.name"),
        "file_type": GGUF_FILE_TYPES.get(ft, f"file_type_{ft}") if ft is not None else None,
        "quantization_version": kv.get("general.quantization_version"),
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if isinstance(tmpl, str) else None,
        "tokenizer_tokens_sha256": toks.get("sha256") if isinstance(toks, dict) else None,
        "context_length": next((v for k, v in kv.items() if k.endswith(".context_length")), None),
        "parameters_stored": meta.get("parameters_stored"),
        "tensor_type_parameters": meta.get("tensor_type_parameters"),
    }


# --------------------------------------------------------------------- repository scan
def _sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def _load_json(p: Path, malformed: list | None = None) -> dict | None:
    """A small repository JSON file, parsed strictly.  A file that is not strict JSON
    (duplicate key, NaN/Infinity, syntax error) is recorded in ``malformed`` and read as
    absent: ambiguous metadata is flagged, never guessed at."""
    try:
        if p.stat().st_size > MAX_JSON_BYTES:
            return None
        obj = loads_strict(p.read_bytes())
        return obj if isinstance(obj, dict) else None
    except (StrictJSONError, UnicodeDecodeError):
        if malformed is not None:
            malformed.append(p.name)
        return None
    except OSError:
        return None


def chat_template_from(tok_cfg: dict | None, files: dict[str, bytes]) -> str | None:
    """Chat template text as stored (tokenizer_config.json, chat_template.jinja or .json); never rendered."""
    if "chat_template.jinja" in files:
        return files["chat_template.jinja"].decode("utf-8", errors="replace")
    if "chat_template.json" in files:
        try:
            obj = loads_strict(files["chat_template.json"])
            if isinstance(obj, dict) and isinstance(obj.get("chat_template"), str):
                return obj["chat_template"]
        except (StrictJSONError, UnicodeDecodeError):
            pass
    if tok_cfg:
        t = tok_cfg.get("chat_template")
        if isinstance(t, str):
            return t
        if isinstance(t, list):  # list of named templates
            return json.dumps(t, sort_keys=True)
    return None


def config_flags(config: dict | None, tok_cfg: dict | None, file_names: list[str],
                 malformed_json: list[str] | None = None) -> dict:
    auto_map = {}
    for src, cfg in (("config.json", config), ("tokenizer_config.json", tok_cfg)):
        if cfg and isinstance(cfg.get("auto_map"), dict):
            auto_map[src] = cfg["auto_map"]
    code_files = sorted(n for n in file_names if n.endswith(CODE_SUFFIXES))
    pickles = sorted(n for n in file_names if n.endswith(PICKLE_SUFFIXES))
    safetensors = sorted(n for n in file_names if n.endswith(".safetensors"))
    ggufs = sorted(n for n in file_names if n.endswith(".gguf"))
    custom_generate = sorted(n for n in file_names if n.startswith("custom_generate/"))
    return {
        "remote_code_declared": bool(auto_map),
        "auto_map": auto_map,
        "code_files": code_files,
        "custom_generate_files": custom_generate,
        "pickle_weight_files": pickles,
        "safetensors_files": safetensors,
        "gguf_files": ggufs,
        "malformed_json": sorted(set(malformed_json or [])),
        "executed_anything": False,
        "policy": "passive: JSON, safetensors headers and GGUF metadata parsed with the standard library; "
                  "code, hooks, pickles and templates are listed and hashed, never executed or rendered",
    }


def inspect_local_dir(path: str | os.PathLike, *, hash_limit_bytes: int = 64 << 20) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    files = []
    names = []
    small: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() and not p.resolve().is_relative_to(root.resolve()):
            continue  # never follow links out of the repository
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".cache/") or "/.git/" in f"/{rel}":
            continue
        size = p.stat().st_size
        names.append(rel)
        entry = {"path": rel, "size": size}
        if size <= hash_limit_bytes:
            entry["sha256"] = _sha256_file(p)
        files.append(entry)
        if rel in SMALL_JSON or rel in ("chat_template.jinja", "tokenizer.json"):
            if size <= MAX_JSON_BYTES:
                small[rel] = p.read_bytes()
    malformed: list[str] = []
    config = _load_json(root / "config.json", malformed) if (root / "config.json").exists() else None
    tok_cfg = _load_json(root / "tokenizer_config.json", malformed) if (root / "tokenizer_config.json").exists() else None
    gen_cfg = _load_json(root / "generation_config.json", malformed) \
        if (root / "generation_config.json").exists() else None
    if "chat_template.json" in small:
        try:
            loads_strict(small["chat_template.json"])
        except (StrictJSONError, UnicodeDecodeError):
            malformed.append("chat_template.json")
    tmpl = chat_template_from(tok_cfg, small)
    st_summary = None
    st_files = [n for n in names if n.endswith(".safetensors")]
    if st_files:
        agg = {"tensors": 0, "parameters_stored": 0, "dtype_parameters": {}}
        for n in st_files:
            with open(root / n, "rb") as f:
                s = summarize_safetensors_header(read_safetensors_header(f))
            agg["tensors"] += s["tensors"]
            agg["parameters_stored"] += s["parameters_stored"]
            for k, v in s["dtype_parameters"].items():
                agg["dtype_parameters"][k] = agg["dtype_parameters"].get(k, 0) + v
        st_summary = agg
    gguf_summary = {}
    for n in names:
        if n.endswith(".gguf"):
            with open(root / n, "rb") as f:
                gguf_summary[n] = summarize_gguf(read_gguf_metadata(f))
    return {
        "source": "local_dir",
        "files": files,
        "flags": config_flags(config, tok_cfg, names, malformed),
        "config": summarize_config(config),
        "generation_config": gen_cfg,
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
        "tokenizer_json_sha256": hashlib.sha256(small["tokenizer.json"]).hexdigest() if "tokenizer.json" in small else None,
        "tokenizer_config_sha256": hashlib.sha256(small["tokenizer_config.json"]).hexdigest()
        if "tokenizer_config.json" in small else None,
        "safetensors": st_summary,
        "gguf": gguf_summary or None,
    }


def summarize_config(config: dict | None) -> dict | None:
    if not config:
        return None
    keys = ("model_type", "architectures", "torch_dtype", "dtype", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings",
            "tie_word_embeddings", "rope_scaling", "sliding_window", "attn_implementation")
    out = {k: config[k] for k in keys if k in config}
    q = config.get("quantization_config")
    if isinstance(q, dict):
        out["quantization_config"] = {k: q[k] for k in ("quant_method", "bits", "group_size", "sym", "desc_act",
                                                         "weight_dtype", "fmt") if k in q}
    return out


def read_gguf_file(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        return summarize_gguf(read_gguf_metadata(f))


def gguf_from_bytes(data: bytes) -> dict:
    return summarize_gguf(read_gguf_metadata(io.BytesIO(data), limit=len(data), with_tensors=False))
