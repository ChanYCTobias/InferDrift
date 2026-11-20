# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the offline test suite.

Everything here runs without network access: a known-answer test bank, replay
files for the local stand-in endpoint, a CLI runner, a fake Hugging Face Hub on
127.0.0.1 and a repository whose every hook would leave a sentinel file if it
were ever executed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import re
import secrets
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from inferdrift.cli import main as cli_main  # noqa: E402
from inferdrift.jsonio import StrictJSONError, loads_strict  # noqa: E402
from inferdrift.standin import StandinState, request_key, serve  # noqa: E402


# ------------------------------------------------------------------ test bank
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather in a city.",
                                         "parameters": {"type": "object", "properties": {
                                             "city": {"type": "string"},
                                             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                                             "required": ["city", "unit"]}}}


def make_test_bank(n: int) -> list[dict]:
    """Numeric, multiple-choice and tool-call items with known answers."""
    items = []
    for i in range(n):
        k = i % 3
        if k == 0:
            items.append({"id": f"t-num-{i:04d}", "suite": "t_numeric", "messages": [
                {"role": "user", "content": f"What is {i} + {i + 7}? Reply with only the number."}],
                "tools": None, "template_kwargs": {}, "max_new_tokens": 16,
                "grader": {"type": "numeric", "answer": 2 * i + 7, "tol": 0}})
        elif k == 1:
            items.append({"id": f"t-mc-{i:04d}", "suite": "t_choice", "messages": [
                {"role": "user", "content": f"Question {i}: pick option {'ABCD'[i % 4]}.\nA) a\nB) b\nC) c\nD) d"}],
                "tools": None, "template_kwargs": {}, "max_new_tokens": 8,
                "grader": {"type": "choice", "answer": "ABCD"[i % 4]}})
        else:
            city = ["Paris", "Oslo", "Lima", "Seoul"][i % 4]
            unit = ["celsius", "fahrenheit"][i % 2]
            items.append({"id": f"t-tool-{i:04d}", "suite": "t_tool", "messages": [
                {"role": "user", "content": f"Weather in {city} in {unit}? ({i})"}], "tools": [TOOL],
                "template_kwargs": {}, "max_new_tokens": 64,
                "grader": {"type": "tool_call", "name": "get_weather", "arguments": {"city": city, "unit": unit},
                           "parameters": TOOL["function"]["parameters"]}})
    return items


def correct_output(item: dict) -> str:
    g = item["grader"]
    if g["type"] == "numeric":
        return f"The answer is {g['answer']}."
    if g["type"] == "choice":
        return f"The correct option is {g['answer']}."
    return "<tool_call>\n" + json.dumps({"name": g["name"], "arguments": g["arguments"]}) + "\n</tool_call>"


def wrong_output(item: dict) -> str:
    g = item["grader"]
    if g["type"] == "numeric":
        return f"The answer is {g['answer'] + 1}."
    if g["type"] == "choice":
        return "The correct option is " + "BCDA"["ABCD".index(g["answer"])] + "."
    return "I cannot check the weather."


def write_bank(items: list[dict], path: Path) -> Path:
    path.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items), encoding="utf-8")
    return path


def write_replay(items: list[dict], path: Path, text_fn) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"key": request_key(messages=it["messages"], tools=it.get("tools")),
                                "text": text_fn(it)}) + "\n")
    return path


def new_test_key() -> str:
    """A random credential-shaped string, made at run time so no key-like literal is stored here."""
    return "idtest" + secrets.token_hex(16)


# ------------------------------------------------------------------ stand-in
class Standins:
    """Starts local stand-in endpoints on free ports and shuts them all down at the end."""

    def __init__(self) -> None:
        self.servers = []

    def start(self, **kw) -> tuple[StandinState, int]:
        state = StandinState(**kw)
        httpd, port = serve(state)
        self.servers.append(httpd)
        return state, port

    def close(self) -> None:
        stoppers = [threading.Thread(target=s.shutdown) for s in self.servers]
        for t in stoppers:
            t.start()
        for t in stoppers:
            t.join()
        for s in self.servers:
            s.server_close()


def url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


# ------------------------------------------------------------------ CLI runners
def run_main(args: list[str]) -> tuple[int, str]:
    """Run the CLI in this process; return (exit code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main([str(a) for a in args])
    return code, buf.getvalue()


def compare(ref_port: int, cand_port: int, bank: Path, out: Path, extra=()) -> tuple[int, str, dict]:
    code, stdout = run_main(["compare", "--ref", url(ref_port), "--cand", url(cand_port), "--bank", bank,
                             "--out", out] + list(extra))
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    return code, stdout, rep


def cli_env(extra: dict | None = None) -> dict:
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1"})
    env.update(extra or {})
    return env


def run_cli(args: list[str], cwd: Path, env_extra: dict | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run ``python -m inferdrift`` in a subprocess (used where process isolation matters)."""
    return subprocess.run([sys.executable, "-m", "inferdrift"] + [str(a) for a in args], env=cli_env(env_extra),
                          capture_output=True, text=True, timeout=timeout, cwd=str(cwd))


def strict_last_line(stdout: str):
    lines = [x for x in (stdout or "").strip().splitlines() if x.strip()]
    if not lines:
        return None
    try:
        return loads_strict(lines[-1])
    except StrictJSONError:
        return None


def scan_for(paths: list[Path], needles: list[str], extra_texts=()) -> list[str]:
    """Locations of every needle (or its URL-encoded or base64 form) in files under paths and in extra_texts."""
    import base64
    import urllib.parse

    forms = set()
    for n in needles:
        forms.update({n, urllib.parse.quote(n, safe=""), base64.b64encode(n.encode()).decode().rstrip("=")})
    blobs = [(f"text{i}", t) for i, t in enumerate(extra_texts)]
    for root in paths:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for p in files:
            blobs.append((str(p), p.read_bytes().decode("utf-8", errors="replace")))
    return [where for where, text in blobs for f in forms if f and f in text]


# ------------------------------------------------------------------ fake Hub
class FakeHub:
    """Serves the Hub API routes the package uses, from an in-memory table.

    repos = {"org/name": {"commit": "...", "files": {"path": {"size": n, "content": bytes | None}}}}
    Every request path is logged so a test can show which files were (not) fetched."""

    def __init__(self, repos: dict) -> None:
        self.repos = repos
        self.log: list[str] = []
        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                hub.log.append(path)
                m = re.match(r"^/api/models/([^/]+/[^/]+)/revision/(.+)$", path)
                if m and m.group(1) in hub.repos:
                    return self._json(200, {"sha": hub.repos[m.group(1)]["commit"]})
                m = re.match(r"^/api/models/([^/]+/[^/]+)/tree/(.+)$", path)
                if m and m.group(1) in hub.repos:
                    files = hub.repos[m.group(1)]["files"]
                    return self._json(200, [{"type": "file", "path": p, "size": f["size"], "oid": "0" * 40,
                                             **({"lfs": {"oid": "0" * 64, "size": f["size"]}} if f["size"] > 1 << 20
                                                else {})} for p, f in files.items()])
                m = re.match(r"^/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", path)
                if m and m.group(1) in hub.repos:
                    f = hub.repos[m.group(1)]["files"].get(m.group(3))
                    if f is None or f.get("content") is None:
                        return self._json(404, {"error": "not served by the fake hub"})
                    data = f["content"]
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._json(404, {"error": "unknown"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ------------------------------------------------------------------ hostile repository
def _payload(name: str, sentinel_dir: Path) -> str:
    return (
        "import os, pathlib\n"
        f"d = pathlib.Path(os.environ.get('INFERDRIFT_TEST_SENTINEL_DIR') or {str(sentinel_dir)!r})\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        f"(d / '{name}.fired').write_text('executed')\n"
    )


class _Hook:
    def __init__(self, code: str) -> None:
        self.code = code

    def __reduce__(self):
        return (exec, (self.code,))


def tiny_safetensors() -> bytes:
    header = {"__metadata__": {"format": "pt", "note": "harmless metadata"},
              "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)


def build_hostile_repo(root: Path, sentinel_dir: Path) -> dict:
    """A model directory whose every hook writes a sentinel file if it is ever executed."""
    root.mkdir(parents=True, exist_ok=True)
    hooks = {}
    jinja_sentinel = sentinel_dir / "jinja_template.fired"
    ssti = "{{ cycler.__init__.__globals__.os.system('touch " + str(jinja_sentinel) + "') }}"
    (root / "config.json").write_text(json.dumps({
        "model_type": "hooknet", "architectures": ["HookForCausalLM"], "torch_dtype": "float32",
        "auto_map": {"AutoConfig": "configuration_hook.HookConfig",
                     "AutoModelForCausalLM": "modeling_hook.HookForCausalLM"}}))
    (root / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "HookTokenizer", "auto_map": {"AutoTokenizer": ["tokenization_hook.HookTokenizer", None]},
        "chat_template": ssti}))
    (root / "chat_template.jinja").write_text(ssti)
    for name in ("configuration_hook", "modeling_hook", "tokenization_hook"):
        (root / f"{name}.py").write_text(_payload(name, sentinel_dir))
        hooks[name] = f"{name}.fired"
    (root / "__init__.py").write_text(_payload("package_init", sentinel_dir))
    hooks["package_init"] = "package_init.fired"
    (root / "custom_generate").mkdir(exist_ok=True)
    (root / "custom_generate" / "generate.py").write_text(_payload("custom_generate", sentinel_dir))
    hooks["custom_generate"] = "custom_generate.fired"
    (root / "hook.pth").write_text("import os; exec(" + repr(_payload("pth_hook", sentinel_dir)) + ")\n")
    hooks["pth_hook"] = "pth_hook.fired"
    (root / "pytorch_model.bin").write_bytes(pickle.dumps(_Hook(_payload("pickle_weights", sentinel_dir))))
    hooks["pickle_weights"] = "pickle_weights.fired"
    (root / "model.safetensors").write_bytes(tiny_safetensors())
    return hooks


def fire_hooks_deliberately(root: Path, control_dir: Path) -> None:
    """Positive control: execute each Python hook on purpose, in subprocesses, so the sentinels are shown to work."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "INFERDRIFT_TEST_SENTINEL_DIR": str(control_dir),
           "PYTHONDONTWRITEBYTECODE": "1"}
    control_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        ["-c", f"import runpy; runpy.run_path({str(root / 'modeling_hook.py')!r})"],
        ["-c", f"import runpy; runpy.run_path({str(root / 'tokenization_hook.py')!r})"],
        ["-c", f"import runpy; runpy.run_path({str(root / 'configuration_hook.py')!r})"],
        ["-c", f"import runpy; runpy.run_path({str(root / '__init__.py')!r})"],
        ["-c", f"import runpy; runpy.run_path({str(root / 'custom_generate' / 'generate.py')!r})"],
        ["-c", f"exec(open({str(root / 'hook.pth')!r}).read())"],
        ["-c", f"import pickle; pickle.load(open({str(root / 'pytorch_model.bin')!r}, 'rb'))"],
    ]
    for args in runs:
        subprocess.run([sys.executable, "-I"] + args, env=env, capture_output=True, text=True, timeout=60)


AUDIT_DRIVER = r'''
import json, sys
root = sys.argv[1]
events = []
MARKS = ("_hook", "cycler", root)
def _where(event, args):
    # Only the location of code is examined (the file a code object or module comes from,
    # or a process command line), never source text, so a docstring cannot match.
    if event == "exec":
        return getattr(args[0], "co_filename", repr(args[0]))
    if event == "compile":
        return repr(args[1]) if len(args) > 1 else ""
    if event == "import":
        return repr(args[:2])
    return repr(args)[:2000]
def hook(event, args):
    if event in ("os.system", "pickle.find_class"):
        events.append([event, repr(args)[:300]])
    elif event in ("exec", "compile", "import", "subprocess.Popen", "os.exec", "os.posix_spawn", "socket.connect",
                   "ctypes.dlopen"):
        text = _where(event, args)
        if any(m in text for m in MARKS):
            events.append([event, text[:300]])
sys.addaudithook(hook)
from inferdrift.cli import main
code = main(sys.argv[2:])
sys.stdout.write("\n__AUDIT__" + json.dumps({"exit": code, "events": events}) + "\n")
'''


def run_with_audit(root: Path, cli_args: list[str], cwd: Path, env_extra: dict) -> tuple[subprocess.CompletedProcess, dict]:
    p = subprocess.run([sys.executable, "-c", AUDIT_DRIVER, str(root)] + [str(a) for a in cli_args],
                       env=cli_env(env_extra), capture_output=True, text=True, timeout=300, cwd=str(cwd))
    audit = {}
    if "__AUDIT__" in p.stdout:
        audit = json.loads(p.stdout.split("__AUDIT__", 1)[1].strip().splitlines()[0])
    return p, audit
