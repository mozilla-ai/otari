"""The packaged models.dev snapshot and the rules that cut it."""

import json
from importlib import resources

from gateway.services.pricing import (
    active_generations,
    bundled_generation,
    load_bundled_index,
    reset_generations,
    set_accepted_generations,
    trim_catalog,
)


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
            "models": {"m": {"name": "M", "cost": {"input": 1, "tiers": []}, "limit": {"context": 10}, "status": "beta"}},
        }
    }


def test_the_generation_list_falls_back_to_the_bundled_snapshot() -> None:
    reset_generations()
    assert active_generations() == (bundled_generation(),)

    generation = bundled_generation()
    set_accepted_generations([generation])
    assert active_generations() == (generation,)
    reset_generations()
