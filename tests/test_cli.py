# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Command line: entry point, help text, and closed numeric domains (refused before any request, exit 3)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys

import pytest

from helpers import (Standins, cli_env, correct_output, make_test_bank, run_cli, run_main, strict_last_line, url,
                     write_bank, write_replay)
from inferdrift import __version__
from inferdrift.jsonio import loads_strict

PRICE = "10,30"


def test_python_m_entry_point(tmp_path):
    p = run_cli(["--version"], tmp_path)
    assert p.returncode == 0 and p.stdout.strip() == f"inferdrift {__version__}"
    assert __version__ == "0.2.0.dev0"


def test_plan_option_is_marked_experimental(capsys):
    from inferdrift.cli import main

    with pytest.raises(SystemExit) as e:
        main(["compare", "--help"])
    assert e.value.code == 0
    assert "EXPERIMENTAL" in capsys.readouterr().out


def test_banks_command():
    code, out = run_main(["banks"])
    listing = loads_strict(out)
    assert code == 0
    assert listing["math_short"] == 140 and listing["needle_16k"] == 8
    assert listing["bundles"]["core"] == ["math_short", "multiple_choice", "tool_calling", "instruction_following",
                                          "code_unit"]


def test_bad_arguments_are_refusals():
    code, out = run_main(["compare", "--ref", "http://127.0.0.1:9/v1"])
    line = strict_last_line(out)
    assert code == 3 and line["status"] == "refused" and line["code"] == "BAD_ARGUMENTS" and "--cand" in line["message"]


def test_standin_server_listens(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "inferdrift", "standin", "--port", "0", "--price-in", "10",
                             "--price-out", "30"], env=cli_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(tmp_path))
    try:
        first = proc.stdout.readline()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
    started = strict_last_line(first) or {}
    assert started.get("standin") == "listening" and started.get("price_usd_per_mtok") == [10.0, 30.0]


@pytest.mark.parametrize("extra", [["--price-in", "nan"], ["--price-out", "-1"], ["--price-in", "inf"],
                                   ["--truncate-frac", "2"], ["--nondeterministic", "nan"], ["--fail-every", "-1"],
                                   ["--port", "70000"]])
def test_standin_refuses_bad_values(extra):
    code, out = run_main(["standin", "--port", "0"] + extra)
    line = strict_last_line(out)
    assert code == 3 and line["code"] == "BAD_INPUT"


def test_standin_on_a_busy_port_is_a_clean_error(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        p = run_cli(["standin", "--port", port], tmp_path, timeout=60)
    line = strict_last_line(p.stdout)
    assert p.returncode == 4 and "Traceback" not in p.stderr
    assert line["status"] == "error" and line["code"] == "LISTEN_FAILED" and str(port) in line["message"]


def test_raw_mode_without_the_hf_extra_is_a_clean_error(tmp_path, monkeypatch):
    tok = tmp_path / "tokenizer"
    tok.mkdir()
    (tok / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tok / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    monkeypatch.setitem(sys.modules, "transformers", None)  # as if the hf extra were not installed
    s = Standins()
    try:
        _, port = s.start()
        code, out = run_main(["channels", f"{url(port)}?mode=raw&tokenizer=hf:{tok}"])
    finally:
        s.close()
    line = strict_last_line(out)
    assert code == 4
    assert line["status"] == "error" and line["code"] == "MISSING_EXTRA" and "hf extra" in line["message"]


# name, extra args, price for both stacks (None = not given), candidate spec query, expected field substring
CLI_CASES = [
    ("max_usd_nan", ["--max-usd", "nan"], PRICE, "", "--max-usd"),
    ("max_usd_inf", ["--max-usd", "inf"], PRICE, "", "--max-usd"),
    ("max_usd_negative", ["--max-usd", "-1"], PRICE, "", "--max-usd"),
    ("max_usd_zero", ["--max-usd", "0"], PRICE, "", "--max-usd"),
    ("max_usd_overflow_1e999", ["--max-usd", "1e999"], PRICE, "", "--max-usd"),
    ("max_usd_hex", ["--max-usd", "0x1p-4"], PRICE, "", "--max-usd"),
    ("max_usd_blank", ["--max-usd", " 0.5"], PRICE, "", "--max-usd"),
    ("max_usd_bool_text", ["--max-usd", "true"], PRICE, "", "--max-usd"),
    ("max_usd_empty", ["--max-usd", ""], PRICE, "", "--max-usd"),
    ("max_usd_without_price", ["--max-usd", "0.05"], None, "", "max_usd"),
    ("price_nan", ["--max-usd", "0.05"], "nan,nan", "", "--ref-price"),
    ("price_negative", ["--max-usd", "0.05"], "-1,30", "", "--ref-price"),
    ("price_above_max", ["--max-usd", "0.05"], "2e6,30", "", "--ref-price"),
    ("price_one_part", ["--max-usd", "0.05"], "10", "", "--ref-price"),
    ("max_wall_nan", ["--max-wall-s", "nan"], PRICE, "", "--max-wall-s"),
    ("max_wall_zero", ["--max-wall-s", "0"], PRICE, "", "--max-wall-s"),
    ("max_tokens_zero", ["--max-tokens", "0"], PRICE, "", "--max-tokens"),
    ("max_tokens_fraction", ["--max-tokens", "1.5"], PRICE, "", "--max-tokens"),
    ("max_tokens_exponent", ["--max-tokens", "1e3"], PRICE, "", "--max-tokens"),
    ("max_tokens_beyond_exact", ["--max-tokens", "9007199254740993"], PRICE, "", "--max-tokens"),
    ("tau_nan", ["--tau", "nan"], PRICE, "", "--tau"),
    ("tau_zero", ["--tau", "0"], PRICE, "", "--tau"),
    ("tau_one", ["--tau", "1"], PRICE, "", "--tau"),
    ("alpha_two", ["--alpha", "2"], PRICE, "", "--alpha"),
    ("audit_fraction_above_one", ["--audit-fraction", "1.5"], PRICE, "", "--audit-fraction"),
    ("transient_inf", ["--transient-cap-gib", "inf"], PRICE, "", "--transient-cap-gib"),
    ("transient_zero", ["--transient-cap-gib", "0"], PRICE, "", "--transient-cap-gib"),
    ("max_items_zero", ["--max-items", "0"], PRICE, "", "--max-items"),
    ("seed_negative", ["--seed", "-1"], PRICE, "", "--seed"),
    ("seed_beyond_exact", ["--seed", "9007199254740992"], PRICE, "", "--seed"),
    ("temperature_zero", ["--decode", "sample", "--temperature", "0"], PRICE, "", "--temperature"),
    ("top_p_above_one", ["--decode", "sample", "--top-p", "1.5"], PRICE, "", "--top-p"),
    ("looks_decreasing", ["--looks", "64,32"], PRICE, "", "looks"),
    ("looks_empty_part", ["--looks", "32,,64"], PRICE, "", "--looks"),
    ("threads_zero", ["--threads", "0"], PRICE, "", "--threads"),
    ("api_key_env_not_a_name", ["--cand-api-key-env", "not a name"], PRICE, "", "--cand-api-key-env"),
    ("spec_timeout_nan", [], PRICE, "?timeout_s=nan", "timeout_s"),
    ("spec_concurrency_zero", [], PRICE, "?max_concurrency=0", "max_concurrency"),
    ("spec_duplicate_param", [], PRICE, "?model=a&model=b", None),
    ("argparse_bad_choice", ["--decode", "fuzzy"], PRICE, "", None),
    ("argparse_unknown_option", ["--max-dollars", "1"], PRICE, "", None),
]


@pytest.fixture(scope="module")
def priced(tmp_path_factory):
    d = tmp_path_factory.mktemp("priced")
    items = make_test_bank(96)
    bank = write_bank(items, d / "bank.jsonl")
    replay = write_replay(items, d / "replay.jsonl", correct_output)
    s = Standins()
    sa, pa = s.start(backend=f"replay:{replay}", price_in=10.0, price_out=30.0)
    sb, pb = s.start(backend=f"replay:{replay}", price_in=10.0, price_out=30.0)
    yield sa, pa, sb, pb, bank
    s.close()


@pytest.mark.parametrize("name,extra,price,cand_q,field", CLI_CASES, ids=[c[0] for c in CLI_CASES])
def test_out_of_domain_values_are_refused_before_any_request(tmp_path, priced, name, extra, price, cand_q, field):
    sa, pa, sb, pb, bank = priced
    before = sa.totals["requests"] + sb.totals["requests"]
    out = tmp_path / "out"
    args = ["compare", "--ref", url(pa), "--cand", url(pb) + cand_q, "--bank", bank, "--out", out, "--skip-channels"]
    args += (["--ref-price", price, "--cand-price", price] if price is not None else []) + extra
    code, stdout = run_main(args)
    line = strict_last_line(stdout)
    assert code == 3
    assert isinstance(line, dict) and line["status"] == "refused"
    assert line["code"] in ("BAD_INPUT", "BAD_SPEC", "BAD_ARGUMENTS")
    assert sa.totals["requests"] + sb.totals["requests"] == before
    assert not out.exists()
    if field is not None:
        assert field in (line.get("field") or "") or field in (line.get("message") or "")


def test_seed_at_the_top_of_its_domain_still_runs(tmp_path, priced):
    sa, pa, sb, pb, bank = priced
    code, _ = run_main(["compare", "--ref", url(pa), "--cand", url(pb), "--bank", bank, "--out", tmp_path / "top",
                        "--skip-channels", "--decode", "sample", "--seed", "9007199254740991", "--max-items", "32"])
    rep = json.loads((tmp_path / "top" / "report.json").read_text())
    assert code == 0 and rep["status"] == "completed" and rep["coupling"]["independent_bounds"] is True
