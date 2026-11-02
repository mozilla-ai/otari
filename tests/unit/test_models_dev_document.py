"""The typed models.dev document: what it reads from ``api.json`` and how it writes a snapshot back."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gateway.types.models_dev import (
    ModelsDevCatalog,
    ModelsDevCost,
    ModelsDevRates,
    ModelsDevTier,
    models_dev_provider_id,
    parse_catalog,
    snapshot_document,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "gateway" / "types" / "models_dev.py"
# Five models cut from the live v1 ``api.json`` on 2026-11-02, every field as published.
SAMPLE = ROOT / "tests" / "fixtures" / "models_dev_v1_sample.json"


@pytest.fixture(scope="module")
def sample() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(SAMPLE.read_text("utf-8"))
    return document


@pytest.fixture(scope="module")
def catalog(sample: dict[str, Any]) -> ModelsDevCatalog:
    return parse_catalog(sample)


def test_the_live_v1_shape_parses_every_field_the_gateway_reads(catalog: ModelsDevCatalog) -> None:
    provider = catalog.providers["openai"]
    assert (provider.id, provider.name, provider.doc) == ("openai", "OpenAI", "https://platform.openai.com/docs/models")

    model = provider.models["gpt-4o"]
    assert model.id == "gpt-4o"
    assert model.name == "GPT-4o"
    assert model.description and model.family == "gpt"
    assert model.input_modalities == ("text", "image", "pdf")
    assert model.output_modalities == ("text",)
    assert (model.reasoning, model.tool_call, model.structured_output) == (False, True, True)
    assert (model.attachment, model.temperature, model.open_weights) == (True, True, False)
    assert (model.context_window, model.max_output_tokens) == (128000, 16384)
    assert (model.knowledge, model.release_date, model.last_updated) == ("2023-09", "2024-05-13", "2024-08-06")
    assert model.status is None
    assert model.cost == ModelsDevCost(rates=ModelsDevRates(input=2.5, output=10, cache_read=1.25))


def test_a_deprecated_model_keeps_its_status(catalog: ModelsDevCatalog) -> None:
    assert catalog.providers["openai"].models["gpt-4"].status == "deprecated"


def test_a_zero_limit_is_unknown(catalog: ModelsDevCatalog) -> None:
    """models.dev writes 0 where it does not know a limit; a client must not read that as no tokens."""
    model = catalog.providers["openai"].models["gpt-image-1-mini"]
    assert (model.context_window, model.max_output_tokens) == (None, None)
    assert model.output_modalities == ("text", "image")
    assert model.cost is None


def test_tiers_and_the_older_over_200k_block_are_both_kept(catalog: ModelsDevCatalog) -> None:
    cost = catalog.providers["google"].models["gemini-2.5-pro"].cost
    assert cost is not None
    assert cost.rates == ModelsDevRates(input=1.25, output=10, cache_read=0.125)
    assert cost.tiers == (
        ModelsDevTier(kind="context", size=200000, rates=ModelsDevRates(input=2.5, output=15, cache_read=0.25)),
    )
    assert cost.context_over_200k == ModelsDevRates(input=2.5, output=15, cache_read=0.25)
    claude = catalog.providers["anthropic"].models["claude-sonnet-4-5"]
    assert claude.canonical_model_id == "anthropic/claude-sonnet-4-5"


def test_ids_default_to_the_document_keys() -> None:
    catalog = parse_catalog({"p": {"models": {"m": {"name": "M"}}}})
    assert catalog.providers["p"].id == "p"
    assert catalog.providers["p"].models["m"].id == "m"
    assert parse_catalog({"p": {"id": "real", "models": {"m": {"id": "real-m"}}}}).providers["p"].models["m"].id == (
        "real-m"
    )


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"openai": ["gpt-4o"]}, id="provider-not-an-object"),
        pytest.param({"openai": {"models": ["gpt-4o"]}}, id="models-not-an-object"),
        pytest.param({"openai": {}}, id="no-models"),
    ],
)
def test_a_provider_of_the_wrong_shape_is_left_out(document: dict[str, Any]) -> None:
    assert parse_catalog(document).providers == {}


def test_a_model_of_the_wrong_shape_is_left_out() -> None:
    catalog = parse_catalog({"openai": {"models": {"gpt-4o": "GPT-4o", "gpt-4": {}}}})
    assert list(catalog.providers["openai"].models) == ["gpt-4"]


@pytest.mark.parametrize("document", [[], "x", None, 3])
def test_a_document_that_is_not_an_object_raises(document: object) -> None:
    with pytest.raises(TypeError):
        parse_catalog(document)


def test_a_rate_is_kept_as_written_and_a_non_number_is_named_as_malformed() -> None:
    """Whether a number is a believable price is the price index's call; a string or a flag is no rate, and is named."""
    cost = {"input": -1, "output": 2_000_000, "cache_read": "0.5", "cache_write": True}
    parsed = parse_catalog({"p": {"models": {"m": {"cost": cost}}}}).providers["p"].models["m"].cost

    assert parsed == ModelsDevCost(
        rates=ModelsDevRates(input=-1, output=2_000_000, malformed=("cache_read", "cache_write"))
    )
    plain = parse_catalog({"p": {"models": {"m": {"cost": {"input": 1}}}}}).providers["p"].models["m"].cost
    assert plain == ModelsDevCost(rates=ModelsDevRates(input=1))


@pytest.mark.parametrize("size", [0, -5, 1.5, "100", True, None, float("inf"), float("nan")])
def test_an_unusable_tier_size_is_absent(size: object) -> None:
    cost = parse_catalog({"p": {"models": {"m": {"cost": {"tiers": [{"tier": {"type": "context", "size": size}}]}}}}})
    tier = cost.providers["p"].models["m"].cost.tiers[0]  # type: ignore[union-attr]
    assert (tier.kind, tier.size) == ("context", None)


def test_an_integral_float_tier_size_is_an_int() -> None:
    cost = parse_catalog({"p": {"models": {"m": {"cost": {"tiers": [{"tier": {"size": 100.0}, "input": 1}]}}}}})
    assert cost.providers["p"].models["m"].cost.tiers == (  # type: ignore[union-attr]
        ModelsDevTier(kind=None, size=100, rates=ModelsDevRates(input=1)),
    )


def test_flags_and_lists_of_the_wrong_shape_read_as_absent() -> None:
    model = (
        parse_catalog(
            {
                "p": {
                    "models": {
                        "m": {"modalities": {"input": "text", "output": ["text", 3]}, "limit": "big", "name": ""}
                    }
                }
            }
        )
        .providers["p"]
        .models["m"]
    )
    assert (model.input_modalities, model.output_modalities) == ((), ("text",))
    assert (model.context_window, model.name) == (None, None)


def test_model_for_maps_the_any_llm_name_and_the_listing_prefix(catalog: ModelsDevCatalog) -> None:
    """Gemini lists ``models/<id>``; models.dev keys the bare id under ``google``."""
    assert models_dev_provider_id("gemini") == "google"
    assert models_dev_provider_id("openai") == "openai"
    assert catalog.model_for("gemini", "models/gemini-2.5-pro") is catalog.providers["google"].models["gemini-2.5-pro"]
    assert catalog.model_for("gemini", "gemini-2.5-pro") is not None
    assert catalog.model_for("gemini", "models/gemini-9") is None
    assert catalog.model_for("mystery", "gemini-2.5-pro") is None
    assert catalog.provider_for("gemini") is catalog.providers["google"]


def test_models_walks_every_model_with_its_provider(catalog: ModelsDevCatalog) -> None:
    assert [(provider.id, model.id) for provider, model in catalog.models()] == [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4"),
        ("openai", "gpt-image-1-mini"),
        ("anthropic", "claude-sonnet-4-5"),
        ("google", "gemini-2.5-pro"),
    ]


def test_the_catalog_is_immutable(catalog: ModelsDevCatalog) -> None:
    with pytest.raises(TypeError):
        catalog.providers["x"] = catalog.providers["openai"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        catalog.providers["openai"].models["gpt-4o"].name = "x"  # type: ignore[misc]


def test_the_snapshot_keeps_the_price_fields_and_drops_the_rest() -> None:
    document = snapshot_document(
        parse_catalog(
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
    )

    assert document == {
        "p": {
            "id": "p",
            "name": "P",
            "models": {"m": {"name": "M", "cost": {"input": 1}, "limit": {"context": 10}, "status": "beta"}},
        }
    }


def test_the_snapshot_reads_back_as_the_same_prices(sample: dict[str, Any], catalog: ModelsDevCatalog) -> None:
    """A stored snapshot is read like a live download, so what the two say about a price must agree."""
    document = snapshot_document(catalog)
    again = parse_catalog(document)
    assert [(p.id, m.id, m.cost, m.context_window) for p, m in again.models()] == [
        (p.id, m.id, m.cost, m.context_window) for p, m in catalog.models()
    ]
    assert snapshot_document(again) == document
    gemini = "gemini-2.5-pro"
    assert document["google"]["models"][gemini]["cost"] == sample["google"]["models"][gemini]["cost"]
    # A model's key doubles as its id, so the snapshot writes none.
    assert "id" not in document["openai"]["models"]["gpt-4o"]


def test_the_module_is_standard_library_only() -> None:
    """``scripts/update_models_dev_snapshot.py`` loads the file by path, with no gateway on the path."""
    tree = ast.parse(MODULE.read_text("utf-8"))
    imported = {
        (alias.name if isinstance(node, ast.Import) else node.module or "").partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names

    spec = importlib.util.spec_from_file_location("models_dev_by_path", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.snapshot_document(module.parse_catalog({"p": {"models": {}}})) == {"p": {"id": "p", "models": {}}}
