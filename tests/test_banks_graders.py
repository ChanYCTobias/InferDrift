# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Item banks, graders and the unit-test sandbox."""

from __future__ import annotations

import json
import socket

import pytest

from inferdrift import graders
from inferdrift.banks import ALL_SUITES, BUNDLES, bank_digest, load_bank, load_suite, regenerate
from inferdrift.errors import BadInputError
from inferdrift.jsonio import canonical, sha256_text
from inferdrift.sandbox import namespaces_available, run_tests


def test_bundle_sizes():
    assert len(load_bank("core")) == 500
    assert len(load_bank("full")) == 508
    assert len(load_bank("smoke")) == 8 * len(BUNDLES["core"])
    assert len(load_bank("math_short,tool_calling")) == 140 + 90


def test_shipped_banks_regenerate_byte_for_byte(tmp_path):
    digests = regenerate(tmp_path / "regen")
    shipped = {s: sha256_text("".join(canonical(it) + "\n" for it in load_suite(s))) for s in ALL_SUITES}
    assert digests == shipped


def test_bank_digest_is_order_sensitive():
    items = load_bank("smoke")
    assert bank_digest(items) != bank_digest(list(reversed(items)))


def _bank_file(tmp_path, text):
    p = tmp_path / "bank.jsonl"
    p.write_text(text, encoding="utf-8")
    return str(p)


GOOD = {"id": "a", "suite": "s", "messages": [{"role": "user", "content": "2+2?"}], "max_new_tokens": 8,
        "grader": {"type": "numeric", "answer": 4}}


def test_user_bank_control_loads(tmp_path):
    assert load_bank(_bank_file(tmp_path, json.dumps(GOOD) + "\n"))[0]["id"] == "a"


@pytest.mark.parametrize("line", [
    '{"id": "a", "id": "b", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "numeric", "answer": 4}}',
    '{"id": "a", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "numeric", "answer": NaN}}',
    '{"id": "a", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "numeric", "answer": 1e999}}',
    '{"id": "a", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 0, '
    '"grader": {"type": "numeric", "answer": 4}}',
    '{"id": "a", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "numeric", "answer": 4}, "extra": 1}',
    '{"id": "a", "suite": "s", "messages": [{"role": "robot", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "numeric", "answer": 4}}',
    '{"id": "a", "suite": "s", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 8, '
    '"grader": {"type": "mystery"}}',
])
def test_malformed_bank_lines_are_refused(tmp_path, line):
    with pytest.raises(BadInputError):
        load_bank(_bank_file(tmp_path, line + "\n"))


GRADER_CASES = [
    ({"type": "numeric", "answer": 42}, "Steps... Answer: 42", True),
    ({"type": "numeric", "answer": 42}, "Answer: 41", False),
    ({"type": "numeric", "answer": 1500}, "The total is 1,500", True),
    ({"type": "numeric", "answer": -3}, "It is -3.", True),
    ({"type": "choice", "answer": "B"}, "B", True),
    ({"type": "choice", "answer": "B"}, "The answer is C", False),
    ({"type": "choice", "answer": "C"}, "(C)", True),
    ({"type": "tool_call", "name": "f", "arguments": {"x": 1}, "parameters": {"type": "object", "properties": {
        "x": {"type": "integer"}}, "required": ["x"]}}, '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>',
     True),
    ({"type": "tool_call", "name": "f", "arguments": {"x": 1}}, '{"name": "g", "arguments": {"x": 1}}', False),
    ({"type": "tool_call", "name": "f", "arguments": {"x": 1}, "parameters": {"type": "object", "properties": {
        "x": {"type": "integer"}}, "required": ["x"]}}, '{"name": "f", "arguments": {"x": "1"}}', False),
    ({"type": "instruction", "checks": [{"type": "lowercase"}]}, "all lower case here.", True),
    ({"type": "instruction", "checks": [{"type": "lowercase"}]}, "Not lower.", False),
    ({"type": "instruction", "checks": [{"type": "bullet_count", "n": 3}]}, "- a\n- b\n- c", True),
    ({"type": "instruction", "checks": [{"type": "max_words", "n": 3}]}, "one two three four", False),
    ({"type": "instruction", "checks": [{"type": "json_keys", "keys": ["a", "b"]}]}, '{"a": 1, "b": 2}', True),
    ({"type": "instruction", "checks": [{"type": "end_with", "suffix": "That is all."}]}, "x. That is all.", True),
    ({"type": "json_schema", "schema": {"type": "object", "required": ["k"]}}, '{"k": 1}', True),
    ({"type": "exact", "answers": ["Paris"], "normalize": "casefold"}, " paris ", True),
]


@pytest.mark.parametrize("grader,output,expected", GRADER_CASES)
def test_grader_known_answers(grader, output, expected):
    assert graders.grade({"id": "x", "grader": grader}, output).correct is expected


def test_sandbox_without_isolation_refuses_to_run_code():
    if namespaces_available():
        pytest.skip("namespaces are available here; the refusal path is not reachable")
    r = run_tests("def f(x):\n    return x + 1\n", "assert f(1) == 2\n")
    assert not r.passed and r.reason == "SANDBOX_UNAVAILABLE"


def test_sandbox_grades_code():
    weak = not namespaces_available()
    ok = run_tests("def f(x):\n    return x + 1\n", "assert f(1) == 2\n", allow_weak_isolation=weak)
    wrong = run_tests("def f(x):\n    return x\n", "assert f(1) == 2\n", allow_weak_isolation=weak)
    loop = run_tests("def f(x):\n    while True:\n        pass\n", "f(1)\n", timeout_s=3, allow_weak_isolation=weak)
    assert ok.passed and ok.isolation == ("python_socket_block" if weak else "netns")
    assert not wrong.passed and wrong.reason == "ASSERTION_FAILED"
    assert not loop.passed and loop.reason in ("TIMEOUT", "EXIT_-9", "EXIT_137", "EXIT_-24")


def test_sandbox_blocks_network():
    weak = not namespaces_available()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        net = run_tests("import socket\ns = socket.socket()\ns.settimeout(1)\ns.connect(('127.0.0.1', %d))\n" % port,
                        "assert True\n", allow_weak_isolation=weak)
    finally:
        srv.close()
    assert not net.passed and net.reason == "CANDIDATE_ERROR"


def test_weak_isolation_path(monkeypatch):
    """The fallback used when user and network namespaces are unavailable (for example on some CI runners)."""
    import inferdrift.sandbox as sb

    monkeypatch.setattr(sb, "namespaces_available", lambda: False)
    refused = sb.run_tests("def f(x):\n    return x + 1\n", "assert f(1) == 2\n")
    assert not refused.passed and refused.reason == "SANDBOX_UNAVAILABLE"
    ok = sb.run_tests("def f(x):\n    return x + 1\n", "assert f(1) == 2\n", allow_weak_isolation=True)
    assert ok.passed and ok.isolation == "python_socket_block"
    net = sb.run_tests("import socket\ns = socket.socket()\n", "assert True\n", allow_weak_isolation=True)
    assert not net.passed and net.reason == "CANDIDATE_ERROR"
