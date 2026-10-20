"""The packaged models.dev snapshot and the rules that cut it."""

import json
from dataclasses import replace
from functools import cache
from importlib import resources
from typing import Any

import pytest

from gateway.services.pricing import (
    active_generations,
    bundled_generation,
    load_bundled_index,
    reset_generations,
    set_accepted_generations,
    trim_catalog,
)


def _rates(input_: float, output: float, cache_read: float, cache_write: float) -> dict[str, Any]:
    return {"input": input_, "output": output, "cache_read": cache_read, "cache_write": cache_write}


# Anthropic's first-party rates per million tokens, as published at
# https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-16).
# The cache-read rate is not one ratio of input: Opus 5.5 and Sonnet 5.5 read at
# 0.05x and Fable 5.1 at 0.025x. The snapshot's cache_write is the 5m rate; it
# carries no 1h rate.
ANTHROPIC_PUBLISHED_COSTS: dict[str, dict[str, Any]] = {
    "claude-fable-5-1": _rates(10, 50, 0.25, 12.5),
    "claude-fable-5": _rates(10, 50, 1, 12.5),
    "claude-opus-5-5": _rates(4, 20, 0.2, 5),
    "claude-opus-5": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-8": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-7": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-6": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-5": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-5-20251101": _rates(5, 25, 0.5, 6.25),
    "claude-sonnet-5-5": _rates(2, 10, 0.1, 2.5),
    "claude-sonnet-5": _rates(2, 10, 0.2, 2.5),
    "claude-sonnet-4-6": _rates(3, 15, 0.3, 3.75),
    "claude-sonnet-4-5": _rates(3, 15, 0.3, 3.75),
    "claude-sonnet-4-5-20250929": _rates(3, 15, 0.3, 3.75),
    "claude-haiku-5-5": {
        **_rates(0.1, 0.5, 0.01, 0.125),
        "tiers": [{**_rates(0.5, 2.5, 0.05, 0.625), "tier": {"type": "context", "size": 100_000}}],
    },
    "claude-haiku-4-5": _rates(1, 5, 0.1, 1.25),
    "claude-haiku-4-5-20251001": _rates(1, 5, 0.1, 1.25),
}


@cache
def _bundled_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        resources.files("gateway").joinpath("data", "models_dev_pricing.json").read_text("utf-8")
    )
    return document


@pytest.mark.parametrize("model", ANTHROPIC_PUBLISHED_COSTS)
def test_the_bundled_snapshot_carries_anthropics_published_rates(model: str) -> None:
    assert _bundled_document()["anthropic"]["models"][model]["cost"] == ANTHROPIC_PUBLISHED_COSTS[model]


def test_the_bundled_snapshot_prices_well_known_models() -> None:
    index = load_bundled_index()

    assert len(index) > 1000
    entry = index.resolve("openai", "gpt-4o")
    assert entry is not None and entry.priced and entry.context_window
    assert bundled_generation().index is index


def test_the_bundled_snapshot_stays_small_and_carries_only_price_fields() -> None:
    raw = resources.files("gateway").joinpath("data", "models_dev_pricing.json").read_text("utf-8")
    assert len(raw.encode()) < 1_500_000
    document = json.loads(raw)
    assert document["_meta"]["generated_at"]
    allowed = {"id", "name", "cost", "limit", "canonical_model_id", "status"}
    for key, provider in document.items():
        if key != "_meta":
            assert all(set(model) <= allowed for model in provider["models"].values())


def test_trim_keeps_prices_and_drops_the_rest() -> None:
    trimmed = trim_catalog(
        {
            "p": {
                "id": "p",
                "name": "P",
                "env": ["KEY"],
                "models": {
                    "m": {
                        "id": "m",
                        "name": "M",
                        "description": "x",
                        "cost": {"input": 1, "reasoning": 5, "tiers": []},
                        "limit": {"context": 10, "output": 5},
                        "status": "beta",
                    },
                    "bad": "not a model",
                },
            },
            "junk": 3,
        }
    )

    assert trimmed == {
        "p": {
            "id": "p",
            "name": "P",
            "models": {
                "m": {"name": "M", "cost": {"input": 1, "tiers": []}, "limit": {"context": 10}, "status": "beta"}
            },
        }
    }


def test_the_generation_list_falls_back_to_the_bundled_snapshot() -> None:
    reset_generations()
    assert active_generations() == (bundled_generation(),)

    assert bundled_generation().baseline
    generation = replace(bundled_generation(), baseline=False)
    set_accepted_generations([generation])
    baseline, accepted = active_generations()
    assert baseline.baseline
    assert baseline.effective_at < accepted.effective_at
    assert accepted == generation
    reset_generations()
