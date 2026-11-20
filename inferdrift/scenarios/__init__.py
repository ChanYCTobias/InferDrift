# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Example scenarios: reference and candidate specs for an attention-backend change, a quantization
change and a chat-template bug.  They describe comparisons; they record no results."""

from __future__ import annotations

from importlib import resources

from ..jsonio import loads_strict


def list_scenarios() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files(__package__).iterdir() if p.name.endswith(".json"))


def load_scenario(name: str) -> dict:
    """A scenario fixture, parsed strictly (duplicate keys and NaN/Infinity are refused)."""
    return loads_strict(resources.files(__package__).joinpath(f"{name}.json").read_text(encoding="utf-8"))


def scenario_file(name: str) -> str:
    return str(resources.files(__package__).joinpath(name))
