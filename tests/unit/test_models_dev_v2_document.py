"""The v2 models.dev document: read into the same typed catalog as v1, plus what only v2 carries."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from gateway.types.models_dev import (
    MODELS_DEV_SOURCES,
    ModelsDevApi,
    ModelsDevCachePrices,
    ModelsDevCatalog,
    ModelsDevCost,
    ModelsDevIssues,
    ModelsDevModalityCost,
    ModelsDevModel,
    ModelsDevMultiplier,
    ModelsDevOverride,
    ModelsDevRates,
    ModelsDevTier,
    ModelsDevUnitPrice,
    parse_catalog,
    parse_catalog_v2,
    snapshot_document,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
# Cut from the live v2 ``api.json?type=all`` on 2026-11-02, every field as published:
# the five models of the v1 sample, and an image model priced per image.
SAMPLE = FIXTURES / "models_dev_v2_sample.json"
V1_SAMPLE = FIXTURES / "models_dev_v1_sample.json"

# The fields only the v2 reader fills.
V2_ONLY = ("type", "api", "pricing")


@pytest.fixture(scope="module")
def catalog() -> ModelsDevCatalog:
    return parse_catalog_v2(json.loads(SAMPLE.read_text("utf-8")))


@pytest.fixture(scope="module")
def v1_catalog() -> ModelsDevCatalog:
    return parse_catalog(json.loads(V1_SAMPLE.read_text("utf-8")))


def _model(provider: str, model: str, **cost: Any) -> ModelsDevModel:
    """One model read from a v2 document that gives it only ``cost``."""
    return parse_catalog_v2({provider: {"models": {model: {"cost": cost}}}}).providers[provider].models[model]


def test_the_live_v2_shape_has_no_issues(catalog: ModelsDevCatalog) -> None:
    assert catalog.issues == ModelsDevIssues(models=6)
    assert catalog.refusal() is None


def _without_over_200k(model: ModelsDevModel) -> ModelsDevModel:
    """v1 repeats its 200K context tier as ``context_over_200k``, which v2 has no copy of.

    The price index reads that block only for a model with no context tiers, so
    a model whose tiers are the same is priced the same without it.
    """
    if model.cost is None or not model.cost.tiers:
        return model
    return dataclasses.replace(model, cost=dataclasses.replace(model.cost, context_over_200k=None))


def test_v2_fills_every_field_v1_fills_with_the_same_values(
    catalog: ModelsDevCatalog, v1_catalog: ModelsDevCatalog
) -> None:
    """The listing, the price index and the snapshot must not see which document was read."""
    names = [field.name for field in dataclasses.fields(ModelsDevModel) if field.name not in V2_ONLY]
    for provider, v1_model in v1_catalog.models():
        model = _without_over_200k(v1_model)
        read = catalog.providers[provider.id].models[model.id]
        assert {name: getattr(read, name) for name in names} == {name: getattr(model, name) for name in names}, model.id
    for key, provider in v1_catalog.providers.items():
        assert (catalog.providers[key].id, catalog.providers[key].name, catalog.providers[key].doc) == (
            provider.id,
            provider.name,
            provider.doc,
        )


def test_the_v1_reader_fills_none_of_the_v2_fields(v1_catalog: ModelsDevCatalog) -> None:
    """v1 gives no type, so a v1 embedding model must not read as a chat model."""
    assert all((model.type, model.api, model.pricing) == (None, (), None) for _provider, model in v1_catalog.models())


def test_media_types_read_as_the_v1_modalities(catalog: ModelsDevCatalog) -> None:
    model = catalog.providers["openai"].models["gpt-4o"]
    assert (model.input_modalities, model.output_modalities) == (("text", "image", "pdf"), ("text",))
    assert model.attachment


def test_the_type_and_the_api_block_are_carried(catalog: ModelsDevCatalog) -> None:
    openai = catalog.providers["openai"].models
    assert (openai["gpt-4o"].type, openai["gpt-image-1-mini"].type) == ("chat", "image")
    assert openai["gpt-4o"].api == (
        ModelsDevApi(protocol="responses", base_url="https://api.openai.com/v1"),
        ModelsDevApi(protocol="chat-completions", base_url="https://api.openai.com/v1"),
    )
    assert openai["gpt-image-1-mini"].api == (ModelsDevApi(protocol="images", base_url="https://api.openai.com/v1"),)


def test_a_cache_write_per_ttl_gives_v1_its_first_price(catalog: ModelsDevCatalog) -> None:
    model = catalog.providers["anthropic"].models["claude-sonnet-4-5"]
    assert model.pricing is not None
    text = model.pricing.modalities["text"]
    assert text.input is not None
    assert text.input.cache == ModelsDevCachePrices(read=0.3, write=3.75, write_ttls=(("5m", 3.75), ("1h", 6)))
    assert model.cost == ModelsDevCost(rates=ModelsDevRates(input=3, output=15, cache_read=0.3, cache_write=3.75))
    assert model.pricing.web_search == ModelsDevUnitPrice(price=10)


def test_context_overrides_become_the_v1_tiers_and_multipliers_are_carried(catalog: ModelsDevCatalog) -> None:
    model = catalog.providers["google"].models["gemini-2.5-pro"]
    assert model.cost is not None and model.pricing is not None
    assert model.cost.tiers == (
        ModelsDevTier(kind="context", size=200_000, rates=ModelsDevRates(input=2.5, output=15, cache_read=0.25)),
    )
    text = model.pricing.modalities["text"]
    assert text.output is not None
    assert text.output.overrides == (ModelsDevOverride(when={"context": {"gt": 200000}}, price=15),)
    priority, flex = model.pricing.multipliers
    assert priority == ModelsDevMultiplier(when={"service_tier": {"equals": "priority"}}, factor={"text": 1.8})
    assert flex.factor == {"text": {"input": {"price": 0.5}, "output": {"price": 0.5}}}


def test_a_price_in_another_unit_has_no_v1_price(catalog: ModelsDevCatalog) -> None:
    model = catalog.providers["openrouter"].models["qwen/qwen-image-3-pro"]
    assert model.cost is None
    assert model.pricing is not None
    image = model.pricing.modalities["image"]
    assert image.input == ModelsDevUnitPrice(price=0.003, unit="image")
    assert image.output is not None
    assert (image.output.price, image.output.unit) == (0.04, "image")
    assert image.output.overrides == (ModelsDevOverride(when={"resolution": {"equals": "2K"}}, price=0.075),)


def test_a_price_in_a_unit_not_known_here_is_kept_and_prices_nothing() -> None:
    """The document's unit is kept, so a stored catalog reads back the same; v1 shows no price for it."""
    model = _model("p", "m", text={"input": {"price": 1, "unit": "parsec"}, "output": {"price": 2}})
    assert model.cost is None
    assert model.pricing is not None
    assert model.pricing.modalities["text"].input == ModelsDevUnitPrice(price=1, unit="parsec")


def test_a_unit_not_known_here_is_named_once_and_is_not_damage() -> None:
    """A unit models.dev adds is reported, but every model priced in it still reads, so none refuses the document."""
    parsec = {"cost": {"text": {"input": {"price": 1, "unit": "parsec"}, "output": {"price": 2, "unit": "parsec"}}}}
    furlong = {"cost": {"video": {"output": {"price": 3, "unit": "furlong"}}}}
    catalog = parse_catalog_v2({"p": {"models": {"a": parsec, "b": parsec, "c": furlong, "d": {}}}})

    assert catalog.issues == ModelsDevIssues(models=4, unknown_units=("parsec", "furlong"))
    assert catalog.refusal() is None
    assert catalog.issues.describe() == (
        "0 unreadable values in 0 of 4 models; price units not known here: parsec, furlong"
    )


def test_an_odd_unit_is_kept_raw_and_described_bounded_and_quoted() -> None:
    """The log line carries fetched text: a unit with a line break or no end is quoted and cut.

    A short printable unit stays as written, markup included; the catalog
    warning is logged with markup off, so the console handler prints it as is.
    """
    odd = {"[/]": 1, "a\nb": 2, "x" * 100: 3}
    models = {f"m{i}": {"cost": {"text": {"input": {"price": p, "unit": u}}}} for i, (u, p) in enumerate(odd.items())}
    catalog = parse_catalog_v2({"p": {"models": models}})

    assert catalog.issues.unknown_units == tuple(odd)
    assert catalog.issues.describe().endswith("price units not known here: [/], 'a\\nb', '" + "x" * 64 + "'")


def test_a_model_with_no_cost_has_no_prices(catalog: ModelsDevCatalog) -> None:
    model = catalog.providers["openai"].models["gpt-image-1-mini"]
    assert (model.cost, model.pricing) == (None, None)


@pytest.mark.parametrize(
    ("cost", "has_v1_price"),
    [
        pytest.param({"text": {"input": {"price": 1}, "output": {"price": 2}}}, True, id="per-token"),
        pytest.param({"text": {"input": {"price": 1, "unit": "token"}, "output": {"price": 2}}}, True, id="token"),
        pytest.param({"text": {"input": {"price": 1, "unit": "character"}, "output": {"price": 2}}}, False, id="char"),
        pytest.param({"text": {"input": {"price": 1}}}, False, id="no-output"),
        pytest.param({"text": {"output": {"price": 2}}}, False, id="no-input"),
        pytest.param({"video": {"output": {"price": 0.5, "unit": "second"}}}, False, id="media-only"),
    ],
)
def test_v1_has_a_price_only_where_text_input_and_output_are_per_token(
    cost: dict[str, Any], *, has_v1_price: bool
) -> None:
    assert (_model("p", "m", **cost).cost is not None) is has_v1_price


def test_only_an_override_on_context_alone_is_a_tier() -> None:
    model = _model(
        "p",
        "m",
        text={
            "input": {
                "price": 1,
                "overrides": [
                    {"when": {"context": {"gte": 1000}}, "price": 2, "cache": {"write": {"5m": 3}}},
                    {"when": {"context": {"gt": 5000}, "service_tier": {"equals": "flex"}}, "price": 9},
                    {"when": {"service_tier": {"equals": "flex"}}, "price": 9},
                    {"when": {}, "price": 9},
                ],
            },
            "output": {"price": 4, "overrides": [{"when": {"context": {"gt": 8000}}, "price": 6}]},
        },
    )
    assert model.cost is not None
    assert model.cost.tiers == (
        ModelsDevTier(kind="context", size=1000, rates=ModelsDevRates(input=2, output=4, cache_write=3)),
        ModelsDevTier(kind="context", size=8000, rates=ModelsDevRates(input=1, output=6)),
    )


def test_an_explicit_cache_write_stands_in_when_there_is_no_default_one() -> None:
    model = _model("p", "m", text={"input": {"price": 1, "cache": {"explicit": {"write": 2}}}, "output": {"price": 3}})
    assert model.cost is not None
    assert model.cost.rates == ModelsDevRates(input=1, output=3, cache_write=2)


@pytest.mark.parametrize(
    ("model", "issue", "malformed"),
    [
        pytest.param({"type": 3}, "type: expected a string", (), id="type"),
        pytest.param({"canonical_id": 3}, "canonical_id: expected a string", (), id="canonical"),
        pytest.param({"capabilities": []}, "capabilities: expected an object", (), id="capabilities"),
        pytest.param({"capabilities": {"tools": True}}, "capabilities.tools: expected an object", (), id="tools-shape"),
        pytest.param(
            {"capabilities": {"reasoning": {"supported": "yes"}}},
            "capabilities.reasoning.supported: expected a boolean",
            (),
            id="supported",
        ),
        pytest.param(
            {"modalities": {"input": ["text", "model/gltf"]}},
            "modalities.input: expected a known media type",
            (),
            id="media-type",
        ),
        pytest.param({"api": []}, "api: expected an object", (), id="api"),
        pytest.param({"api": {"messages": 1}}, "api.messages: expected an object", (), id="api-entry"),
        pytest.param({"api": {"messages": {"base_url": 1}}}, "api.messages.base_url: expected a string", (), id="url"),
        pytest.param({"cost": {"text": 1}}, "cost.text: expected an object", (), id="modality"),
        pytest.param(
            {"cost": {"text": {"input": {"price": "1"}, "output": {"price": 2}}}},
            "cost.text.input.price: expected a number",
            ("input",),
            id="price",
        ),
        pytest.param(
            {"cost": {"text": {"input": {"price": 1, "unit": 1}, "output": {"price": 2}}}},
            "cost.text.input.unit: expected a string",
            (),
            id="unit",
        ),
        pytest.param(
            {"cost": {"text": {"input": {"price": 1, "cache": {"write": {"5m": "3"}}}, "output": {"price": 2}}}},
            "cost.text.input.cache.write.5m: expected a number",
            ("cache_write",),
            id="ttl",
        ),
        pytest.param(
            {"cost": {"text": {"input": {"price": 1, "cache": {"read": True}}, "output": {"price": 2}}}},
            "cost.text.input.cache.read: expected a number",
            ("cache_read",),
            id="cache-read",
        ),
        pytest.param(
            {"cost": {"text": {"input": {"price": 1, "overrides": {}}, "output": {"price": 2}}}},
            "cost.text.input.overrides: expected a list",
            (),
            id="overrides",
        ),
        pytest.param(
            {"cost": {"text": {"input": {"price": 1, "overrides": [{"when": 3}]}, "output": {"price": 2}}}},
            "cost.text.input.overrides[0].when: expected an object",
            (),
            id="when",
        ),
        pytest.param(
            {"cost": {"multipliers": [{"when": {}, "factor": "2"}]}},
            "cost.multipliers[0].factor: expected a number or an object",
            (),
            id="factor",
        ),
        pytest.param({"cost": {"multipliers": [3]}}, "cost.multipliers[0]: expected an object", (), id="multiplier"),
        pytest.param({"cost": {"web_search": []}}, "cost.web_search: expected an object", (), id="web-search"),
    ],
)
def test_a_v2_field_of_the_wrong_type_is_named_with_its_path(
    model: dict[str, Any], issue: str, malformed: tuple[str, ...]
) -> None:
    catalog = parse_catalog_v2({"p": {"models": {"m": model}}})
    assert catalog.issues == ModelsDevIssues(count=1, examples=(f"p.models.m.{issue}",), models=1, damaged_models=1)
    cost = catalog.providers["p"].models["m"].cost
    assert (cost.rates.malformed if cost is not None else ()) == malformed


def test_a_malformed_override_price_is_named_on_its_tier() -> None:
    """The price refresh refuses a document by its malformed rates, tiers included."""
    input_ = {"price": 1, "overrides": [{"when": {"context": {"gt": 10}}, "price": "2"}]}
    model = _model("p", "m", text={"input": input_, "output": {"price": 3}})
    assert model.cost is not None
    (tier,) = model.cost.tiers
    assert tier.rates.malformed == ("input",)


def test_the_v2_refusal_rule_is_the_v1_one() -> None:
    assert parse_catalog_v2({}).refusal() == "the document lists no models"
    models: dict[str, Any] = {"a": {}, "b": {}, "c": "C", "d": {"type": 3}}
    assert parse_catalog_v2({"p": {"models": models}}).refusal() == (
        "2 of 4 models are damaged; p.models.c: expected an object; p.models.d.type: expected a string"
    )


def test_conditions_and_factors_are_read_only(catalog: ModelsDevCatalog) -> None:
    pricing = catalog.providers["google"].models["gemini-2.5-pro"].pricing
    assert pricing is not None
    with pytest.raises(TypeError):
        pricing.multipliers[0].when["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        pricing.modalities["x"] = ModelsDevModalityCost()  # type: ignore[index]


def test_a_snapshot_cut_from_v2_reads_back_as_the_same_prices(catalog: ModelsDevCatalog) -> None:
    """The bundled snapshot is cut from v2 and stored in the v1 shape, which every gateway reads."""
    again = parse_catalog(snapshot_document(catalog))
    assert [(p.id, m.id, m.cost, m.context_window) for p, m in again.models()] == [
        (p.id, m.id, m.cost, m.context_window) for p, m in catalog.models()
    ]


def test_v2_is_tried_before_v1() -> None:
    assert [(source.name, source.parse) for source in MODELS_DEV_SOURCES] == [
        ("v2", parse_catalog_v2),
        ("v1", parse_catalog),
    ]
    v2, v1 = (source.url for source in MODELS_DEV_SOURCES)
    assert v2 == "https://models.dev/experimental/v2.0/api.json?type=all"
    assert v1 == "https://models.dev/api.json"
