# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""InferDrift: coupled differential evaluation of two LLM inference stacks.

Library entry points::

    from inferdrift.compare import CompareConfig, run_compare
    from inferdrift.banks import load_bank
    from inferdrift.passive import inspect_local_dir
    from inferdrift.stats import StatsBackend

The core package has no third-party dependencies; the transformers reference
adapter needs the ``hf`` extra.
"""

__version__ = "0.1.0.dev0"
