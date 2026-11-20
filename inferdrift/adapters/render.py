# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Client-side prompt rendering with a Hugging Face tokenizer (raw mode).

Used by the reference adapter and by endpoint adapters in ``mode=raw``, so that
both stacks see byte-identical prompts unless the stack itself changes them.
transformers renders chat templates in a sandboxed Jinja environment; custom
tokenizer code is never loaded (``trust_remote_code=False``)."""

from __future__ import annotations

import hashlib


class HFRenderer:
    def __init__(self, local_dir: str, template_override: str | None = None) -> None:
        from transformers import AutoTokenizer  # lazy: only the hf extra needs it

        self.tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=False, local_files_only=True)
        if template_override is not None:
            self.tok.chat_template = template_override
        self.template_sha256 = hashlib.sha256((self.tok.chat_template or "").encode()).hexdigest()
        if getattr(self.tok, "pad_token_id", None) is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def render(self, messages: list, tools: list | None, template_kwargs: dict | None) -> str:
        kw = dict(template_kwargs or {})
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                            tokenize=False, **kw)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def eos_ids(self, generation_config: dict | None = None) -> list[int]:
        ids = set()
        if generation_config:
            e = generation_config.get("eos_token_id")
            if isinstance(e, int):
                ids.add(e)
            elif isinstance(e, list):
                ids.update(x for x in e if isinstance(x, int))
        if self.tok.eos_token_id is not None:
            ids.add(int(self.tok.eos_token_id))
        for t in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                ids.add(tid)
        return sorted(ids)
