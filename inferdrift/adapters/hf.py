# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face transformers reference adapter (CPU by default).

* Loads only safetensors weights with ``trust_remote_code=False`` from a local
  directory that the planner downloaded (or the user supplied) after a passive
  check; repositories that declare custom code are refused before transformers is
  imported (see ``compare.prepare_stacks``).
* Greedy decoding, or sampling coupled across stacks by shared Gumbel noise: the
  noise for (run seed, item, step) is drawn from a counter-seeded generator, added
  to the temperature-scaled, top-k/top-p-masked logits, and the argmax is taken.
  Two stacks that see the same noise and nearly the same logits pick the same token
  (Gumbel-max coupling), so identical stacks give identical samples.
* Optional weight-only quantization transform ``quant=rtn-int<b>-g<g>``:
  asymmetric round-to-nearest with a per-group scale and zero point on every linear
  layer of the decoder blocks; weights are stored dequantized (simulated INT-b
  weights, fp32 compute).
* Thread count is pinned (default 8): on hybrid CPUs the PyTorch default of one
  thread per logical CPU can be many times slower.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

from ..errors import CapReached, PartialCapReached
from ..profile import field_value, finalize, host_hardware, new_profile, unobservable
from .base import Adapter, GenRequest, GenResult
from .render import HFRenderer

DEFAULT_THREADS = 8
DEFAULT_BATCH = 16


def _noise_seed(seed: int, key: str, step: int) -> int:
    h = hashlib.sha256(f"gumbel|{seed}|{key}|{step}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def gumbel_noise(torch, seed: int, key: str, step: int, vocab: int):
    g = torch.Generator(device="cpu").manual_seed(_noise_seed(seed, key, step))
    u = torch.rand(vocab, generator=g, dtype=torch.float64).clamp_(1e-300, 1.0 - 1e-16)
    return (-torch.log(-torch.log(u))).to(torch.float32)


def parse_quant(q: str | None) -> dict | None:
    if not q:
        return None
    m = re.fullmatch(r"rtn-int(\d+)-g(\d+)", q)
    if not m:
        raise ValueError(f"unknown quant {q!r} (expected rtn-int<bits>-g<group>)")
    bits, gs = int(m.group(1)), int(m.group(2))
    if not (2 <= bits <= 8) or gs <= 0:
        raise ValueError("bits must be in 2..8 and group size positive")
    return {"method": "rtn", "bits": bits, "group_size": gs, "symmetric": False,
            "scope": "decoder linear layers (embeddings and lm_head untouched)",
            "storage": "dequantized fp32 (simulated weight-only quantization)"}


def rtn_quantize_(torch, model, bits: int, group_size: int) -> int:
    qmax = 2 ** bits - 1
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and ".layers." in name:
                w = mod.weight.data
                out_f, in_f = w.shape
                if in_f % group_size:
                    continue
                wg = w.reshape(out_f, in_f // group_size, group_size)
                wmin = wg.amin(dim=-1, keepdim=True)
                wmax = wg.amax(dim=-1, keepdim=True)
                scale = (wmax - wmin).clamp(min=1e-8) / qmax
                zp = torch.round(-wmin / scale).clamp(0, qmax)
                q = torch.clamp(torch.round(wg / scale) + zp, 0, qmax)
                mod.weight.data = ((q - zp) * scale).reshape(out_f, in_f).to(w.dtype)
                n += 1
    return n


class HFAdapter(Adapter):
    kind = "hf"
    supports_shared_noise = True

    def __init__(self, spec, role: str, local_dir: str, passive_profile: dict, *, threads: int | None = None,
                 batch_size: int | None = None) -> None:
        super().__init__(spec, role)
        import torch  # lazy
        import transformers
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.local_dir = local_dir
        self.passive = passive_profile
        self.threads = int(spec.params.get("threads", threads or DEFAULT_THREADS))
        self.batch_size = int(spec.params.get("batch", batch_size or DEFAULT_BATCH))
        self.max_batch_tokens = int(spec.params.get("max_batch_tokens", 48000))
        torch.set_num_threads(self.threads)
        tmpl_path = spec.params.get("template")
        self.template_override = None
        if tmpl_path:
            with open(os.path.expanduser(tmpl_path), encoding="utf-8") as f:
                self.template_override = f.read()
        self.renderer = HFRenderer(local_dir, self.template_override)
        self.tok = self.renderer.tok
        dtype_name = spec.params.get("dtype", "float32")
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
        self.attn = spec.params.get("attn", "sdpa")
        kwargs = dict(attn_implementation=self.attn, trust_remote_code=False, use_safetensors=True,
                      local_files_only=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=self.dtype, **kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=self.dtype, **kwargs)
        self.model.eval()
        self.model.to("cpu")
        self.quant = parse_quant(spec.params.get("quant"))
        self.quantized_layers = 0
        if self.quant:
            self.quantized_layers = rtn_quantize_(torch, self.model, self.quant["bits"], self.quant["group_size"])
        gen_cfg = (passive_profile or {}).get("generation_config") or {}
        self.eos_ids = self.renderer.eos_ids(gen_cfg)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids[0]
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # Drop the repository's sampling defaults (for example do_sample/temperature/top_k in
        # generation_config.json) so that only the decoding inferdrift sets is in effect.
        from transformers import GenerationConfig

        self.model.generation_config = GenerationConfig(eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
                                                        do_sample=False)

    # ------------------------------------------------------------------- profile
    def profile(self) -> dict:
        torch = self.torch
        p = new_profile(self.kind, self.spec.display())
        f = p["fields"]
        src_id = self.spec.repo or self.spec.local_path
        f["model_id"] = field_value(src_id, "declared:spec")
        rev = (self.passive or {}).get("commit") or self.spec.revision
        f["model_revision"] = field_value(rev, "passive:hub_tree" if (self.passive or {}).get("commit") else "declared:spec")
        f["weight_dtype"] = field_value(str(self.dtype).replace("torch.", ""), "runtime:torch")
        f["quantization"] = field_value(self.quant or {"method": "none"}, "declared:spec" if self.quant else "runtime:torch")
        f["kv_dtype"] = field_value(str(self.dtype).replace("torch.", ""), "runtime:transformers(dynamic cache)")
        f["engine"] = field_value("transformers", "runtime:transformers")
        f["engine_version"] = field_value({"transformers": self.transformers_version, "torch": torch.__version__},
                                          "runtime:transformers")
        f["attention_backend"] = field_value(getattr(self.model.config, "_attn_implementation", self.attn),
                                             "runtime:transformers")
        f["gemm_backend"] = field_value({"device": "cpu", "mkl": bool(torch.backends.mkl.is_available()),
                                         "mkldnn": bool(torch.backends.mkldnn.is_available()),
                                         "threads": torch.get_num_threads()}, "runtime:torch")
        f["hardware"] = field_value(host_hardware(), "runtime:os")
        f["parallelism"] = field_value({"tp": 1, "pp": 1, "batch_size": self.batch_size}, "runtime:inferdrift")
        f["speculative_decoding"] = field_value(None, "runtime:inferdrift (none)")
        f["chat_template_sha256"] = field_value(self.renderer.template_sha256,
                                                "declared:spec(template override)" if self.template_override
                                                else "passive:tokenizer_config.json")
        f["tokenizer_sha256"] = field_value((self.passive or {}).get("tokenizer_json_sha256"), "passive:tokenizer.json")
        f["decode_parameters"] = field_value({"greedy": "do_sample=False", "eos_token_ids": self.eos_ids},
                                             "runtime:inferdrift")
        f["output_caps"] = field_value({"per_item": "max_new_tokens from the bank"}, "runtime:inferdrift")
        f["prompt_rendering"] = field_value("transformers.apply_chat_template", "runtime:transformers")
        f["parameters_stored"] = field_value(self.n_params, "runtime:torch")
        try:  # a read-only fact about this process; never changed by inferdrift
            p["flags"]["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:  # pragma: no cover - very old torch
            p["flags"]["deterministic_algorithms"] = None
        p["flags"]["quantized_layers"] = self.quantized_layers
        p["flags"]["passive"] = (self.passive or {}).get("flags")
        return finalize(p)

    # ---------------------------------------------------------------- generation
    def _processors(self, keys: list[str], prompt_len: int, decode, seed: int):
        torch = self.torch
        from transformers import LogitsProcessor, LogitsProcessorList

        class SharedGumbel(LogitsProcessor):
            def __call__(self, input_ids, scores):
                step = input_ids.shape[1] - prompt_len
                s = scores / max(decode.temperature, 1e-6)
                if decode.top_k and decode.top_k > 0:
                    kth = torch.topk(s, min(decode.top_k, s.shape[-1]), dim=-1).values[..., -1:]
                    s = s.masked_fill(s < kth, float("-inf"))
                if decode.top_p < 1.0:
                    sorted_s, idx = torch.sort(s, descending=True, dim=-1)
                    probs = torch.softmax(sorted_s, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > decode.top_p
                    sorted_s = sorted_s.masked_fill(remove, float("-inf"))
                    s = torch.full_like(s, float("-inf")).scatter(-1, idx, sorted_s)
                noise = torch.stack([gumbel_noise(torch, seed, k, step, s.shape[-1]) for k in keys])
                return s + noise

        return LogitsProcessorList([SharedGumbel()])

    def _stopper(self, meter):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Deadline(StoppingCriteria):
            def __init__(self):
                self.hit = False

            def __call__(self, input_ids, scores, **kw):
                rem = meter.remaining_wall()
                if rem is not None and rem <= 0:
                    self.hit = True
                    return True
                return False

        d = Deadline()
        return d, StoppingCriteriaList([d])

    def _batches(self, prepared: list[tuple[int, GenRequest, str, list[int]]]):
        groups: dict[tuple, list] = {}
        for entry in prepared:
            _, req, _, ids = entry
            groups.setdefault((req.max_new_tokens, req.decode.mode), []).append(entry)
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            entries = sorted(groups[key], key=lambda e: len(e[3]))
            batch = []
            for e in entries:
                longest = max([len(x[3]) for x in batch] + [len(e[3])])
                if batch and (len(batch) >= self.batch_size or
                              (len(batch) + 1) * (longest + key[0]) > self.max_batch_tokens):
                    yield batch
                    batch = []
                batch.append(e)
            if batch:
                yield batch

    def generate(self, reqs: list[GenRequest], meter) -> list[GenResult]:
        torch = self.torch
        from transformers import GenerationConfig

        prepared = []
        for i, r in enumerate(reqs):
            text = self.renderer.render(r.messages, r.tools, r.template_kwargs)
            prepared.append((i, r, text, self.renderer.encode(text)))
        results: dict[int, GenResult] = {}
        for batch in self._batches(prepared):
            reservations = []
            try:
                for _, r, _, ids in batch:
                    reservations.append(meter.reserve(self.role, len(ids), r.max_new_tokens))
            except CapReached as e:
                for res in reservations:
                    meter.cancel(res)
                raise PartialCapReached(self._prefix(results, len(reqs)), e) from None
            t0 = time.monotonic()
            max_new = batch[0][1].max_new_tokens
            decode = batch[0][1].decode
            enc = self.tok([e[2] for e in batch], return_tensors="pt", padding=True, add_special_tokens=False)
            prompt_len = enc["input_ids"].shape[1]
            cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False, eos_token_id=self.eos_ids,
                                   pad_token_id=self.pad_id, temperature=None, top_p=None, top_k=None)
            kwargs = {}
            if decode.mode == "sample":
                keys = [e[1].coupling_key or e[1].item_id for e in batch]
                seed = decode.seed
                kwargs["logits_processor"] = self._processors(keys, prompt_len, decode, seed)
            deadline, stopping = self._stopper(meter)
            want_k = max(e[1].top_logprobs for e in batch)
            if want_k:
                cfg.output_scores = True
                cfg.return_dict_in_generate = True
            with torch.inference_mode():
                out = self.model.generate(**enc, generation_config=cfg, stopping_criteria=stopping, **kwargs)
            step_lp = None
            if want_k:
                step_lp = [torch.log_softmax(s.float(), dim=-1) for s in out.scores]
                out = out.sequences
            if deadline.hit:
                for res in reservations:
                    meter.cancel(res)
                raise PartialCapReached(self._prefix(results, len(reqs)),
                                        CapReached("wall-time cap reached during generation", cap="wall"))
            elapsed = time.monotonic() - t0
            gen = out[:, prompt_len:]
            for row, (i, r, text, ids), res in zip(gen, batch, reservations):
                toks = row.tolist()
                cut = len(toks)
                fr = "length"
                for j, t in enumerate(toks):
                    if t in self.eos_ids:
                        cut, fr = j, "stop"
                        break
                toks = toks[:cut]
                if fr == "length" and len(toks) < r.max_new_tokens:
                    fr = "unknown"  # ended short of the cap without an end token: never reported as a clean stop
                meter.settle(res, len(ids), len(toks), usage_source="local")
                lp = None
                if step_lp is not None and r.top_logprobs:
                    b = batch.index((i, r, text, ids))
                    lp = []
                    for j, t in enumerate(toks):
                        row_lp = step_lp[j][b]
                        top = torch.topk(row_lp, r.top_logprobs)
                        lp.append({"token": self.renderer.decode([t]), "logprob": float(row_lp[t]),
                                   "top_logprobs": [{"token": self.renderer.decode([int(x)]), "logprob": float(v)}
                                                    for v, x in zip(top.values, top.indices)]})
                results[i] = GenResult(r.item_id, self.renderer.decode(toks), fr, len(ids), len(toks), "local",
                                       elapsed_s=elapsed / len(batch),
                                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                       logprobs=lp, meta={"tokens": toks})
        return [results[i] for i in range(len(reqs))]

    @staticmethod
    def _prefix(results: dict, n: int) -> list:
        out = []
        for i in range(n):
            if i not in results:
                break
            out.append(results[i])
        return out

    def prompt_logprobs(self, req: GenRequest, meter) -> list | None:
        torch = self.torch
        text = self.renderer.render(req.messages, req.tools, req.template_kwargs)
        ids = self.renderer.encode(text)
        res = meter.reserve(self.role, len(ids), 0, kind="probe")
        with torch.inference_mode():
            logits = self.model(torch.tensor([ids])).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        vals = [None] + [float(lp[j - 1, ids[j]]) for j in range(1, len(ids))]
        meter.settle(res, len(ids), 0, usage_source="local")
        return vals

    def close(self) -> None:
        self.model = None

