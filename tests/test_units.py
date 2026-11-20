# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Passive parsing, cost meter, stack specs, redaction, judge reuse and closed library domains."""

from __future__ import annotations

import base64
import io
import struct
import time
import urllib.parse

import pytest

from helpers import new_test_key, tiny_safetensors
from inferdrift import passive
from inferdrift.adapters.base import DecodeConfig, GenRequest, GenResult
from inferdrift.adapters.openai_compat import OpenAICompatAdapter
from inferdrift.compare import CompareConfig
from inferdrift.errors import BadInputError, CapReached, SpecError
from inferdrift.inherit import GradeBook, JudgeGrader
from inferdrift.jsonio import StrictJSONError, loads_strict, sha256_text
from inferdrift.meter import Caps, CostMeter, PriceCard
from inferdrift.redact import REDACTED, register_secret, scrub_text
from inferdrift.spec import parse_spec


# ------------------------------------------------------------------ passive parsing
def test_safetensors_header():
    s = passive.summarize_safetensors_header(passive.read_safetensors_header(io.BytesIO(tiny_safetensors())))
    assert s["parameters_stored"] == 4 and s["dtype_parameters"] == {"F32": 4}


def _gstr(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def test_gguf_metadata():
    kv = [(_gstr("general.architecture") + struct.pack("<I", 8) + _gstr("qwen3")),
          (_gstr("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 2)),
          (_gstr("tokenizer.chat_template") + struct.pack("<I", 8) + _gstr("{{ x }}")),
          (_gstr("qwen3.context_length") + struct.pack("<I", 4) + struct.pack("<I", 40960)),
          (_gstr("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 2) +
           _gstr("a") + _gstr("b"))]
    tensors = (_gstr("w") + struct.pack("<I", 2) + struct.pack("<QQ", 32, 4) + struct.pack("<I", 2) +
               struct.pack("<Q", 0))
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kv)) + b"".join(kv) + tensors
    g = passive.summarize_gguf(passive.read_gguf_metadata(io.BytesIO(blob)))
    assert g["architecture"] == "qwen3"
    assert g["file_type"] == "MOSTLY_Q4_0"
    assert g["context_length"] == 40960
    assert g["parameters_stored"] == 128
    assert g["tensor_type_parameters"] == {"Q4_0": 128}
    assert g["chat_template_sha256"] == sha256_text("{{ x }}")


# ------------------------------------------------------------------ strict JSON
@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": 1e999}'])
def test_strict_json_refuses(text):
    with pytest.raises(StrictJSONError):
        loads_strict(text)


def test_strict_json_control():
    assert loads_strict('{"a": 1.5, "b": [1, 2]}') == {"a": 1.5, "b": [1, 2]}


# ------------------------------------------------------------------ cost meter
def test_dollar_cap_stops_at_or_below_the_cap():
    import random

    rng = random.Random(9)
    m = CostMeter(Caps(max_usd=0.5), {"candidate": PriceCard(10.0, 30.0)})
    with pytest.raises(CapReached):
        while True:
            pin, mout = rng.randint(10, 3000), rng.randint(1, 300)
            r = m.reserve("candidate", pin, mout)
            m.settle(r, rng.randint(1, pin), rng.randint(0, mout), usage_source="provider")
    snap = m.snapshot()
    assert snap["stopped_by"] == "usd" and snap["usd_total"] <= 0.5


def test_overrun_stops_the_next_request():
    m = CostMeter(Caps(max_usd=1.0), {"candidate": PriceCard(10.0, 30.0)})
    r = m.reserve("candidate", 100, 10)
    m.settle(r, 5000, 10, usage_source="provider")  # the provider bills more than was reserved
    with pytest.raises(CapReached):
        m.reserve("candidate", 1, 1)
    assert len(m.snapshot()["overruns"]) == 1


def test_wall_cap():
    m = CostMeter(Caps(max_wall_s=0.05))
    time.sleep(0.1)
    with pytest.raises(CapReached) as e:
        m.check_wall()
    assert e.value.cap == "wall"


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("kw", [dict(max_usd=NAN), dict(max_usd=INF), dict(max_usd=-1.0), dict(max_usd=0.0),
                                dict(max_usd=True), dict(max_usd="0.5"), dict(max_tokens=-1), dict(max_tokens=0),
                                dict(max_tokens=True), dict(max_tokens=1.5), dict(max_tokens=2 ** 53),
                                dict(max_wall_s=NAN), dict(max_wall_s=-5.0), dict(max_wall_s=0)])
def test_caps_refuse_out_of_domain(kw):
    with pytest.raises(BadInputError):
        Caps(**kw)


@pytest.mark.parametrize("text", ["nan,nan", "inf,1", "1e999,1", "-1,2", "1", "1,2,3", "", " 1,2", "1,2 ", "0x10,1",
                                  "1_0,2", "true,1"])
def test_price_parse_refuses(text):
    with pytest.raises(BadInputError):
        PriceCard.parse(text)


def test_price_and_caps_controls():
    assert Caps() == Caps(None, None, None) and Caps(0.5, 1000, 60).max_usd == 0.5
    assert PriceCard.parse("10,30") == PriceCard(10.0, 30.0) and PriceCard.parse(None) is None
    assert PriceCard().cost(1000, 1000) == 0.0


@pytest.mark.parametrize("kw", [dict(tau=NAN), dict(tau=1.0), dict(alpha=2.0), dict(alpha=0.0),
                                dict(audit_fraction=1.5), dict(transient_cap_gib=NAN), dict(max_items=0),
                                dict(threads=0), dict(stats_mode="other"), dict(seed=-1)])
def test_compare_config_refuses_out_of_domain(kw):
    with pytest.raises(BadInputError):
        CompareConfig(ref="http://127.0.0.1:9/v1?model=m", cand="http://127.0.0.1:9/v1?model=m", **kw)


def test_dollar_cap_needs_a_price_for_every_endpoint():
    with pytest.raises(BadInputError):
        CompareConfig(ref="http://127.0.0.1:9/v1", cand="http://127.0.0.1:9/v1", caps=Caps(max_usd=1.0),
                      ref_price=PriceCard(1.0, 1.0))
    CompareConfig(ref="http://127.0.0.1:9/v1", cand="http://127.0.0.1:9/v1", caps=Caps(max_usd=1.0),
                  ref_price=PriceCard(1.0, 1.0), cand_price=PriceCard(1.0, 1.0))


# ------------------------------------------------------------------ specs and redaction
def test_spec_rejects_url_credentials():
    with pytest.raises(SpecError):
        parse_spec("https://user:pw@host/v1")


def test_spec_parse():
    s = parse_spec("hf:Qwen/Qwen3-0.6B@abc?attn=eager&threads=4")
    assert s.repo == "Qwen/Qwen3-0.6B" and s.revision == "abc" and s.params["attn"] == "eager"
    with pytest.raises((SpecError, BadInputError)):
        parse_spec("http://127.0.0.1:9/v1?model=a&model=b")


def test_template_kwargs_are_opt_in():
    req = GenRequest("i", [{"role": "user", "content": "x"}], None, {"enable_thinking": False}, 8, DecodeConfig())
    plain = OpenAICompatAdapter(parse_spec("http://127.0.0.1:9/v1?model=m"), "candidate")._chat_body(req)
    opted = OpenAICompatAdapter(parse_spec("http://127.0.0.1:9/v1?model=m&send_template_kwargs=1"),
                                "candidate")._chat_body(req)
    assert "chat_template_kwargs" not in plain
    assert opted.get("chat_template_kwargs") == {"enable_thinking": False}


def test_registered_secret_is_redacted_in_all_forms():
    k = new_test_key()
    register_secret(k)
    t = scrub_text(f"a {k} b {urllib.parse.quote(k)} c {base64.b64encode(k.encode()).decode()}")
    assert k not in t and REDACTED in t


# ------------------------------------------------------------------ judge reuse
class _StubJudge:
    role = "judge"
    kind = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, reqs, meter):
        self.calls += len(reqs)
        return [GenResult(r.item_id, "YES" if "Answer to grade:\nIt is Paris" in r.messages[0]["content"] else "NO",
                          "stop", 1, 1, "local") for r in reqs]


def test_judge_verdicts_are_reused(tmp_path):
    j = _StubJudge()
    cache = tmp_path / "judge_cache.jsonl"
    gb = GradeBook(judge=JudgeGrader(j, CostMeter(Caps()), judge_digest="stub-v1", cache_path=str(cache)),
                   audit_fraction=0.5)
    item = {"id": "j1", "messages": [{"role": "user", "content": "Capital of France?"}],
            "grader": {"type": "judge", "rubric": "mentions Paris"}}
    rg = gb.grade_reference(item, "It is Paris.")
    cg, how, spent = gb.grade_follower(item, "It is Paris.", rg, "It is Paris.")
    assert j.calls == 1 and how == "judge_cache_hit" and spent == 0 and cg.correct
    cg2, how2, _ = gb.grade_follower(item, "It is Paris.", rg, "It is Lyon.")
    assert j.calls == 2 and how2 == "judge_call" and not cg2.correct
    j2 = _StubJudge()
    JudgeGrader(j2, CostMeter(Caps()), judge_digest="stub-v1", cache_path=str(cache)).grade(item, "It is Lyon.")
    assert j2.calls == 0  # the on-disk cache persists across graders


def test_shared_gumbel_noise():
    torch = pytest.importorskip("torch")
    from inferdrift.adapters.hf import gumbel_noise

    a = gumbel_noise(torch, 7, "item#0", 3, 1000)
    b = gumbel_noise(torch, 7, "item#0", 3, 1000)
    c = gumbel_noise(torch, 7, "item#1", 3, 1000)
    assert torch.equal(a, b) and not torch.equal(a, c)
