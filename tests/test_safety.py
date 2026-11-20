# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Repository code is never executed, and downloads above the allowance are refused before any byte is written."""

from __future__ import annotations

import json

from helpers import FakeHub, build_hostile_repo, fire_hooks_deliberately, run_main, run_with_audit
from inferdrift import hub


def test_hostile_repository_is_never_executed(tmp_path):
    sentinels = tmp_path / "sentinels"
    repo = tmp_path / "hostile_repo"
    hooks = build_hostile_repo(repo, sentinels)
    env = {"INFERDRIFT_TEST_SENTINEL_DIR": str(sentinels)}

    p, audit = run_with_audit(repo, ["inspect", repo], tmp_path, env)
    assert p.returncode == 0 and audit.get("exit") == 0, p.stderr[-400:]
    assert not sentinels.exists() or list(sentinels.iterdir()) == []
    assert audit.get("events") == []
    profile = json.loads(p.stdout.split("__AUDIT__")[0])
    flags = profile["profile"]["flags"]
    assert flags["remote_code_declared"] is True
    assert "pytorch_model.bin" in flags["pickle_weight_files"]

    p2, audit2 = run_with_audit(repo, ["compare", "--ref", f"hf:{repo}", "--cand", f"hf:{repo}", "--bank", "smoke",
                                       "--out", tmp_path / "refused"], tmp_path, env)
    assert audit2.get("exit") == 3, p2.stdout[-400:]
    assert not sentinels.exists() or list(sentinels.iterdir()) == []
    assert audit2.get("events") == []

    # positive control: the same hooks, executed on purpose, do fire
    control = tmp_path / "control_sentinels"
    repo2 = tmp_path / "hostile_repo_control"
    build_hostile_repo(repo2, control)
    fire_hooks_deliberately(repo2, control)
    assert sorted(x.name for x in control.iterdir()) == sorted(hooks.values())


def _repos():
    gib = 1 << 30
    cfg = json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}).encode()
    tok = json.dumps({"tokenizer_class": "Qwen2Tokenizer"}).encode()
    small_files = {"config.json": {"size": len(cfg), "content": cfg},
                   "tokenizer_config.json": {"size": len(tok), "content": tok}}
    big = {"commit": "a" * 40, "files": dict(small_files, **{
        "model-00001-of-00002.safetensors": {"size": 5 * gib, "content": None},
        "model-00002-of-00002.safetensors": {"size": 2 * gib, "content": None}})}
    small = {"commit": "b" * 40, "files": dict(small_files, **{"model.safetensors": {"size": 1 * gib,
                                                                                      "content": None}})}
    return {"big/model": big, "small/model": small}, 8 * gib + 2 * (len(cfg) + len(tok))


def test_download_allowance_refuses_before_any_weight_is_fetched(tmp_path, monkeypatch):
    repos, total = _repos()
    fake = FakeHub(repos)
    monkeypatch.setenv("HF_ENDPOINT", fake.url)
    try:
        work = tmp_path / "work"
        code, _ = run_main(["compare", "--ref", "hf:small/model", "--cand", "hf:big/model", "--bank", "smoke",
                            "--out", tmp_path / "out", "--work-dir", work, "--keep-work-dir"])
        rep = json.loads((tmp_path / "out" / "report.json").read_text())
        assert code == 3
        assert rep["refusal"]["code"] == "TRANSIENT_ALLOWANCE_EXCEEDED"
        assert rep["refusal"]["plan"]["total_bytes"] == total
        assert [x for x in fake.log if "/resolve/" in x and x.endswith(".safetensors")] == []
        assert hub.dir_bytes(work) == 0
        # control: the same plan fits a larger allowance, so the cap alone caused the refusal
        code2, out2 = run_main(["plan-downloads", "--ref", "hf:small/model", "--cand", "hf:big/model",
                                "--transient-cap-gib", "9"])
        assert code2 == 0 and json.loads(out2)["ok"] is True
        code3, _ = run_main(["plan-downloads", "--ref", "hf:small/model", "--cand", "hf:big/model"])
        assert code3 == 3
    finally:
        fake.close()
