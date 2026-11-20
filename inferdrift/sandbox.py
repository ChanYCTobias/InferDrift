# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Network-less sandbox for unit-test grading.

Candidate code and its tests run in a fresh ``python -I -S`` subprocess (isolated
mode, no site packages, empty environment) inside a temporary directory, under
CPU-time, address-space and file-size limits and a wall-clock timeout.

Network isolation uses a new user and network namespace (``unshare -rn``): the
child sees only a loopback interface that is down, so no connection can leave
it.  When namespaces are unavailable (for example inside an unprivileged
container) the sandbox reports ``isolation = 'unavailable'`` and refuses to run
code unless the caller explicitly accepts the weaker ``'python_socket_block'``
mode, which disables the socket module inside the child before the candidate
code runs.  The isolation level is recorded with every grade.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_PRELUDE = r'''
import sys
def _no_net(*a, **k):
    raise OSError("network disabled in inferdrift sandbox")
try:
    import socket as _s
    _s.socket = _no_net
    _s.create_connection = _no_net
    _s.getaddrinfo = _no_net
except Exception:
    pass
'''

_RUNNER = r'''
import sys, traceback
ns = {"__name__": "__candidate__"}
src = open("candidate.py", encoding="utf-8").read()
tests = open("tests.py", encoding="utf-8").read()
try:
    exec(compile(src, "candidate.py", "exec"), ns)
except BaseException:
    traceback.print_exc(limit=2)
    sys.exit(3)
try:
    exec(compile(tests, "tests.py", "exec"), ns)
except AssertionError:
    traceback.print_exc(limit=2)
    sys.exit(1)
except BaseException:
    traceback.print_exc(limit=2)
    sys.exit(2)
print("ALL_TESTS_PASSED")
'''


def _limits(cpu_s: int, mem_bytes: int, fsize: int):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        os.setsid()

    return apply


_PROBED: dict[str, bool] = {}


def namespaces_available() -> bool:
    if "ns" not in _PROBED:
        ok = False
        unshare = shutil.which("unshare")
        if unshare:
            try:
                p = subprocess.run([unshare, "-rn", sys.executable, "-I", "-S", "-c",
                                    "import socket\ns=socket.socket()\ns.settimeout(1)\n"
                                    "try:\n s.connect(('1.1.1.1',80))\n print('NET')\n"
                                    "except OSError:\n print('NONET')"],
                                   capture_output=True, text=True, timeout=20, env={})
                ok = p.returncode == 0 and p.stdout.strip() == "NONET"
            except Exception:
                ok = False
        _PROBED["ns"] = ok
    return _PROBED["ns"]


@dataclass
class SandboxResult:
    passed: bool
    exit_code: int | None
    reason: str
    isolation: str
    stdout_tail: str


def run_tests(code: str, tests: str, *, timeout_s: float = 10.0, allow_weak_isolation: bool = False,
              cpu_s: int = 10, mem_bytes: int = 1 << 30, fsize: int = 1 << 22) -> SandboxResult:
    if namespaces_available():
        isolation = "netns"
    elif allow_weak_isolation:
        isolation = "python_socket_block"
    else:
        return SandboxResult(False, None, "SANDBOX_UNAVAILABLE", "unavailable", "")
    with tempfile.TemporaryDirectory(prefix="inferdrift-sbx-") as d:
        with open(os.path.join(d, "candidate.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(d, "tests.py"), "w", encoding="utf-8") as f:
            f.write(tests)
        with open(os.path.join(d, "runner.py"), "w", encoding="utf-8") as f:
            f.write(_PRELUDE + _RUNNER)
        cmd = [sys.executable, "-I", "-S", "runner.py"]
        if isolation == "netns":
            cmd = [shutil.which("unshare") or "unshare", "-rn"] + cmd
        try:
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout_s,
                               env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                               preexec_fn=_limits(cpu_s, mem_bytes, fsize))
        except subprocess.TimeoutExpired:
            return SandboxResult(False, None, "TIMEOUT", isolation, "")
        passed = p.returncode == 0 and p.stdout.strip().endswith("ALL_TESTS_PASSED")
        reason = {0: "PASSED" if passed else "NO_PASS_MARKER", 1: "ASSERTION_FAILED", 2: "TEST_ERROR",
                  3: "CANDIDATE_ERROR"}.get(p.returncode, f"EXIT_{p.returncode}")
        return SandboxResult(passed, p.returncode, reason, isolation, (p.stdout + p.stderr)[-400:])
