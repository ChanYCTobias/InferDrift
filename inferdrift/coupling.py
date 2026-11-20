# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Choosing the coupling between the two stacks' decoders.

* greedy        - both stacks decode greedily; coupled by construction (outputs of
                  identical deterministic stacks are identical).
* shared_gumbel - seeded sampling with shared randomness: both stacks run in this
                  process and add the same Gumbel noise per (item, step) before the
                  argmax (Gumbel-max coupling).  Identical stacks give identical
                  samples, so flips measure the change, not sampling noise.
* independent   - sampling without verified shared randomness (an endpoint that
                  ignores the seed, or two different engines whose samplers cannot be
                  coupled).  Flagged.  The verdict then uses the excess harmful-flip
                  rate over a reference replicate (a second, independently sampled
                  reference run), with union-bound limits that are honestly wider.

A shared seed sent to two endpoints is not shared randomness unless both run the
same sampler implementation; it is reported as ``seed_shared_unverified`` and the
independent-sampling bounds apply.

Under the paired criteria (``net_change``, ``harmful_excess``) a stack whose
greedy outputs are not repeatable is routed to the noise floor (the reference
runs twice and the verdict is judged against that floor), and without verified
shared randomness every run gets its own derived seed (see ``compare.run_indices``).
"""

from __future__ import annotations


def plan_coupling(decode, ref_adapter, cand_adapter, ref_ch: dict | None, cand_ch: dict | None,
                  criterion: str = "harmful_abs") -> dict:
    flags = []
    paired = criterion != "harmful_abs"
    for name, ch in (("reference", ref_ch), ("candidate", cand_ch)):
        rate = (ch or {}).get("greedy_repeat_identity_rate")
        if rate is not None and rate < 1.0:
            if paired:
                flags.append(f"{name} is not deterministic under greedy decoding (repeat identity {rate:.2f}); "
                             f"routed to the noise floor: the reference runs twice and the verdict is judged "
                             f"against that floor")
            else:
                flags.append(f"{name} is not deterministic under greedy decoding (repeat identity {rate:.2f}); "
                             f"part of the observed churn is the stack's own nondeterminism")
    if decode.mode == "greedy":
        return {"mode": "greedy", "coupled": True, "independent_bounds": False,
                "reason": "greedy decoding on both stacks", "flags": flags}
    if ref_adapter.supports_shared_noise and cand_adapter.supports_shared_noise:
        return {"mode": "shared_gumbel", "coupled": True, "independent_bounds": False,
                "reason": "both stacks run in-process; shared Gumbel noise per (item, step)", "flags": flags}
    seeds_ok = bool((ref_ch or {}).get("seed_honored")) and bool((cand_ch or {}).get("seed_honored"))
    if not seeds_ok:
        which = [n for n, ch in (("reference", ref_ch), ("candidate", cand_ch)) if not (ch or {}).get("seed_honored")]
        flags.append(f"seed not honoured by: {', '.join(which)}; sampling is independent")
        mode = "independent"
    else:
        flags.append("both endpoints honour seeds, but a shared seed only couples identical samplers; "
                     "treated as independent sampling" + ("; every run gets its own derived seed" if paired else ""))
        mode = "seed_shared_unverified"
    return {"mode": mode, "coupled": False, "independent_bounds": True,
            "reason": "no verified shared randomness between the stacks", "flags": flags}
