# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic mode: what each stack declares, what the run observed, and advice.

* Declared: the endpoint spec's ``deterministic=on|off``; for the in-process
  transformers reference, ``torch.are_deterministic_algorithms_enabled()`` (read,
  never changed); and engine settings read from an engine's metadata routes,
  for the settings listed in ``CONFIRMED_SETTINGS`` only.  A setting the engine
  does not report is ``unobservable``, never "off".
* Observed: byte identity and answer-level flips between a stack's two runs of
  the same requests, and the channel probe's greedy repeat identity.
* Advice: only when a stack is observed to be nondeterministic.  It is generic
  unless ``CONFIRMED_SETTINGS`` names a setting for that engine.

``CONFIRMED_SETTINGS`` maps an engine kind ("vllm", "llamacpp", "sglang",
"openai", "hf") to a list of entries
``{"setting": str, "how_to_enable": str, "detect": {"route": "/path", "key": str}}``.
An entry is added only once the setting is confirmed to exist in that engine;
the list ships empty, so the advice is generic for now.
"""

from __future__ import annotations

CONFIRMED_SETTINGS: dict[str, list[dict]] = {}

GENERIC_ADVICE = ("This stack's outputs differ between runs of identical requests. If its engine offers a "
                  "deterministic or batch-invariant mode, turn it on and run again: a deterministic stack needs far "
                  "fewer items for the same tolerance (see the power section of the report).")


def declared(adapter, profile: dict | None) -> dict:
    """What the stack declares about deterministic execution (no generation request is sent)."""
    flags = (profile or {}).get("flags") or {}
    kind = (profile or {}).get("kind")
    out = {"spec": flags.get("declared_deterministic"),
           "torch_deterministic_algorithms": flags.get("deterministic_algorithms"),
           "engine_settings": []}
    for entry in CONFIRMED_SETTINGS.get(kind, []):
        det = entry.get("detect") or {}
        value = "unobservable"
        client = getattr(adapter, "client", None)
        if client is not None and isinstance(det.get("route"), str) and isinstance(det.get("key"), str):
            try:
                status, obj, _ = client.get(det["route"], allow_status=(401, 403, 404, 405))
                if status == 200 and isinstance(obj, dict) and det["key"] in obj:
                    value = obj[det["key"]]
            except Exception:  # an unreachable route is unobservable, never "off"
                value = "unobservable"
        out["engine_settings"].append({"setting": entry.get("setting"), "route": det.get("route"),
                                       "key": det.get("key"), "value": value})
    return out


def _declared_on(decl: dict) -> bool:
    if decl.get("spec") == "on" or decl.get("torch_deterministic_algorithms") is True:
        return True
    return any(s.get("value") is True or s.get("value") in ("on", "true", "1", 1)
               for s in decl.get("engine_settings", []))


def section(kind: str | None, decl: dict, observed: dict) -> dict:
    """Report section for one stack.  ``observed`` holds the replicate statistics
    (``items``, ``byte_identical``, ``answer_flips``), the channel probe's
    ``channel_repeat_identity`` and ``batch_composition_varied``."""
    items = observed.get("items") or 0
    byte_diff = items - (observed.get("byte_identical") or 0)
    ch = observed.get("channel_repeat_identity")
    nondet = byte_diff > 0 or (ch is not None and ch < 1.0)
    if nondet:
        status = "observed nondeterministic"
    elif items > 0 or ch is not None:
        status = "observed deterministic"
    else:
        status = "not measured"
    rec = None
    if nondet:
        specific = [e for e in CONFIRMED_SETTINGS.get(kind, []) if e.get("how_to_enable")]
        rec = GENERIC_ADVICE + ("" if not specific else " For this engine: " + "; ".join(
            f"{e['setting']}: {e['how_to_enable']}" for e in specific) + ".")
    return {"declared": decl, "observed": observed, "status": status,
            "basis": (f"{items} items run twice" if items else "no item was run twice") +
                     ("" if ch is None else f"; channel probe repeat identity {ch:.3f}"),
            "declared_on_but_nondeterministic": bool(nondet and _declared_on(decl)),
            "recommendation": rec,
            "settings_source": "confirmed list" if CONFIRMED_SETTINGS.get(kind) else
                               "none confirmed for this engine yet; the advice is generic"}
