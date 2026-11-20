# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Channel detection: what a stack lets the comparison observe and control.

Probes (all metered; none of them is graded):

* ``greedy_repeat_identity_rate`` - the same greedy request sent ``repeats`` times
  per probe prompt, in separate calls; the share of repeats byte-identical to the
  first.  Below 1.0 the stack is not deterministic under greedy decoding.
* ``top_logprobs`` / ``top_logprobs_k`` - a greedy request asking for top-5
  token log-probabilities.
* ``prompt_logprobs`` - log-probabilities of the prompt tokens.
* ``seed_honored`` - two sampled requests (temperature 1) with the same seed must
  match, and a request with another seed must differ from them at least once.
* ``usage_reported`` / ``max_tokens_honored`` - provider token usage, and a
  request capped at 4 new tokens returning at most 4.
"""

from __future__ import annotations

from .adapters.base import DecodeConfig, GenRequest
from .errors import CapReached, FidelityError
from .redact import scrub_text

GREEDY_PROBES = [
    "List the first five prime numbers separated by commas.",
    "Name three colours of the rainbow.",
    "What is 17 multiplied by 3? Reply with only the number.",
]
SAMPLE_PROBE = "Write a list of twelve unrelated random words."
NO_THINK = {"enable_thinking": False}


def _req(item_id: str, text: str, max_new: int, decode: DecodeConfig, seed: int | None = None,
         top_logprobs: int = 0) -> GenRequest:
    return GenRequest(item_id, [{"role": "user", "content": text}], None, dict(NO_THINK), max_new, decode,
                      seed=seed, coupling_key=item_id, top_logprobs=top_logprobs)


def detect_channels(adapter, meter, *, repeats: int = 3, top_k: int = 5) -> dict:
    rep: dict = {"stack": adapter.role, "adapter": adapter.kind, "probe_generations": 0, "errors": []}
    greedy = DecodeConfig("greedy")

    def run(reqs):
        rep["probe_generations"] += len(reqs)
        return adapter.generate(reqs, meter)

    try:
        # greedy repeat identity
        same = total = 0
        firsts = {}
        for r in range(repeats):
            outs = run([_req(f"probe-greedy-{i}", t, 32, greedy) for i, t in enumerate(GREEDY_PROBES)])
            for i, o in enumerate(outs):
                if o.error:
                    rep["errors"].append(scrub_text(o.error))
                if r == 0:
                    firsts[i] = o.text
                else:
                    total += 1
                    same += int(o.text == firsts[i])
        rep["greedy_repeat_identity_rate"] = (same / total) if total else None
        rep["usage_reported"] = None
        # top-k logprobs
        o = run([_req("probe-logprobs", GREEDY_PROBES[0], 8, greedy, top_logprobs=top_k)])[0]
        lp = o.logprobs
        rep["top_logprobs"] = bool(lp)
        rep["top_logprobs_k"] = _observed_k(lp)
        rep["usage_reported"] = o.usage_source in ("provider", "local")
        # prompt logprobs
        try:
            pl = adapter.prompt_logprobs(_req("probe-prompt-lp", GREEDY_PROBES[1], 1, greedy), meter)
            rep["prompt_logprobs"] = bool(pl) and any(v is not None for v in pl)
        except FidelityError as e:
            rep["prompt_logprobs"] = False
            rep["errors"].append(scrub_text(str(e)))
        # max_tokens honoured
        o = run([_req("probe-cap", "Write a long paragraph about the history of bridges.", 4, greedy)])[0]
        rep["max_tokens_honored"] = (o.completion_tokens is not None and o.completion_tokens <= 4) \
            if o.usage_source in ("provider", "local") else None
        # seed honouring
        sample = DecodeConfig("sample", temperature=1.0, top_p=1.0, top_k=0, seed=7)
        a = run([_req("probe-seed", SAMPLE_PROBE, 24, sample, seed=4242)])[0]
        b = run([_req("probe-seed", SAMPLE_PROBE, 24, sample, seed=4242)])[0]
        c1 = run([_req("probe-seed", SAMPLE_PROBE, 24,
                       DecodeConfig("sample", temperature=1.0, top_p=1.0, top_k=0, seed=8), seed=9191)])[0]
        c2 = run([_req("probe-seed", SAMPLE_PROBE, 24,
                       DecodeConfig("sample", temperature=1.0, top_p=1.0, top_k=0, seed=9), seed=31337)])[0]
        same_seed_identical = a.text == b.text and not a.error
        varies = (c1.text != a.text) or (c2.text != a.text)
        rep["seed_same_identical"] = same_seed_identical
        rep["seed_changes_output"] = varies
        rep["seed_honored"] = bool(same_seed_identical and varies)
    except CapReached:
        raise
    except FidelityError as e:
        rep["errors"].append(scrub_text(str(e)))
    return rep


def _observed_k(lp) -> int:
    if not lp:
        return 0
    first = lp[0] if isinstance(lp, list) and lp else None
    if isinstance(first, dict):
        for key in ("top_logprobs", "probs"):
            if isinstance(first.get(key), list):
                return len(first[key])
        return 1
    if isinstance(first, list):
        return len(first)
    return 0
