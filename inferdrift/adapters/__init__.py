# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Stack adapters.

| adapter | module | exercised so far |
|---|---|---|
| Hugging Face transformers (reference, runs on the CPU) | ``hf`` | live |
| llama.cpp server | ``llamacpp`` | live (llama-server, CPU build) |
| any OpenAI-compatible endpoint | ``openai_compat`` | against the local stand-in only |
| vLLM | ``vllm`` | against the stand-in's protocol emulation; not yet validated live |
| SGLang | ``sglang`` | against the stand-in's protocol emulation only |

See the support statement in the README.
"""
