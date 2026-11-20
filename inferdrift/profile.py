# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Stack profiles.

A stack profile records everything about an inference stack that can change its
outputs.  Each field carries its value and where it came from:

* ``passive:<file>``   parsed from repository metadata without executing anything
* ``runtime:<lib>``    observed from the running reference process
* ``endpoint:<path>``  reported by the serving endpoint
* ``declared:spec``    stated in the stack spec
* ``unobservable``     the stack does not expose it (recorded, never guessed)
"""

from __future__ import annotations

import os
import platform
from typing import Any

from .jsonio import sha256_obj

FIELDS = (
    "model_id", "model_revision", "weight_dtype", "quantization", "kv_dtype", "engine", "engine_version",
    "attention_backend", "gemm_backend", "hardware", "parallelism", "speculative_decoding",
    "chat_template_sha256", "tokenizer_sha256", "decode_parameters", "output_caps", "prompt_rendering",
    "parameters_stored",
)
UNOBSERVABLE = "unobservable"


def field_value(value: Any, source: str) -> dict:
    return {"value": value, "source": source}


def unobservable(note: str = "") -> dict:
    return {"value": None, "source": UNOBSERVABLE, **({"note": note} if note else {})}


def new_profile(kind: str, spec_display: str) -> dict:
    prof = {"kind": kind, "spec": spec_display, "fields": {f: unobservable() for f in FIELDS}, "flags": {}}
    return prof


def finalize(profile: dict) -> dict:
    profile["digest"] = sha256_obj({k: v.get("value") for k, v in profile["fields"].items()})
    return profile


def diff(ref: dict, cand: dict) -> list[dict]:
    out = []
    for f in FIELDS:
        a = ref["fields"].get(f, {}).get("value")
        b = cand["fields"].get(f, {}).get("value")
        if a != b:
            out.append({"field": f, "reference": a, "candidate": b,
                        "reference_source": ref["fields"].get(f, {}).get("source"),
                        "candidate_source": cand["fields"].get(f, {}).get("source")})
    return out


def host_hardware() -> dict:
    cpu = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu or platform.processor() or None, "logical_cpus": os.cpu_count(),
            "machine": platform.machine(), "os": platform.system(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
