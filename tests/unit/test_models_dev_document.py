"""The typed models.dev document: what it reads from ``api.json`` and how it writes a snapshot back."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gateway.types.models_dev import (
    MAX_ISSUE_EXAMPLES,
    ModelsDevCatalog,
    ModelsDevCost,
    ModelsDevIssues,
    ModelsDevRates,
    ModelsDevTier,
    models_dev_provider_id,
    models_dev_provider_ids,
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


def test_a_tier_size_too_large_for_a_float_is_read_as_written() -> None:
    """JSON carries integers of any size; one past float range must not raise out of the parser."""
    catalog = parse_catalog({"p": {"models": {"m": {"cost": {"tiers": [{"tier": {"size": 10**400}}]}}}}})
    assert catalog.providers["p"].models["m"].cost.tiers[0].size == 10**400  # type: ignore[union-attr]
    assert catalog.issues.count == 0


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


def test_a_flag_of_the_wrong_type_reads_as_false() -> None:
    model = parse_catalog({"p": {"models": {"m": {"reasoning": "false", "tool_call": 1}}}}).providers["p"].models["m"]
    assert (model.reasoning, model.tool_call) == (False, False)


def test_the_live_v1_shape_has_no_issues(catalog: ModelsDevCatalog) -> None:
    assert catalog.issues == ModelsDevIssues(models=5)
    assert catalog.refusal() is None


def _issues(document: dict[str, Any]) -> tuple[str, ...]:
    return parse_catalog(document).issues.examples


@pytest.mark.parametrize(
    ("document", "issue"),
    [
        pytest.param({"p": ["m"]}, "p: expected an object", id="provider"),
        pytest.param({"p": {}}, "p.models: expected an object", id="no-models"),
        pytest.param({"p": {"models": []}}, "p.models: expected an object", id="models"),
        pytest.param({"p": {"name": 3, "models": {}}}, "p.name: expected a string", id="provider-name"),
        pytest.param({"p": {"models": {"m": "M"}}}, "p.models.m: expected an object", id="model"),
    ],
)
def test_a_provider_or_model_it_cannot_read_is_named(document: dict[str, Any], issue: str) -> None:
    assert _issues(document) == (issue,)


@pytest.mark.parametrize(
    ("model", "issue"),
    [
        pytest.param({"name": 3}, "name: expected a string", id="string"),
        pytest.param({"reasoning": "false"}, "reasoning: expected a boolean", id="flag"),
        pytest.param({"modalities": []}, "modalities: expected an object", id="modalities"),
        pytest.param({"modalities": {"input": "text"}}, "modalities.input: expected a list of strings", id="list"),
        pytest.param(
            {"modalities": {"output": ["text", 3]}}, "modalities.output: expected a list of strings", id="item"
        ),
        pytest.param({"limit": 5}, "limit: expected an object", id="limit"),
        pytest.param({"limit": {"context": "big"}}, "limit.context: expected a non-negative integer", id="context"),
        pytest.param({"limit": {"output": -1}}, "limit.output: expected a non-negative integer", id="negative"),
        pytest.param({"cost": []}, "cost: expected an object", id="cost"),
        pytest.param({"cost": {"input": "0.5"}}, "cost.input: expected a number", id="rate"),
        pytest.param({"cost": {"tiers": {}}}, "cost.tiers: expected a list", id="tiers"),
        pytest.param({"cost": {"tiers": [3]}}, "cost.tiers[0]: expected an object", id="tier"),
        pytest.param({"cost": {"tiers": [{"tier": 3}]}}, "cost.tiers[0].tier: expected an object", id="tier-kind"),
        pytest.param(
            {"cost": {"tiers": [{"tier": {"size": 1.5}}]}},
            "cost.tiers[0].tier.size: expected a positive integer",
            id="tier-size",
        ),
        pytest.param({"cost": {"tiers": [{"input": True}]}}, "cost.tiers[0].input: expected a number", id="tier-rate"),
        pytest.param({"cost": {"context_over_200k": 1}}, "cost.context_over_200k: expected an object", id="over-200k"),
    ],
)
def test_a_field_of_the_wrong_type_is_named_with_its_path(model: dict[str, Any], issue: str) -> None:
    issues = parse_catalog({"p": {"models": {"m": model}}}).issues
    assert issues == ModelsDevIssues(count=1, examples=(f"p.models.m.{issue}",), models=1, damaged_models=1)


def test_an_absent_empty_zero_or_unknown_field_is_no_issue() -> None:
    """models.dev writes 0 for an unknown limit, and a field the gateway does not read never fails a download."""
    model = {"name": "", "limit": {"context": 0}, "cost": {"tiers": [{"tier": {"type": "context"}}]}, "new": [1]}
    assert parse_catalog({"p": {"new": 1, "models": {"m": model, "bare": {}}}}).issues == ModelsDevIssues(models=2)


def test_the_examples_stop_at_the_cap_and_the_count_does_not() -> None:
    models = {f"m{n}": {"name": n} for n in range(MAX_ISSUE_EXAMPLES + 3)}
    issues = parse_catalog({"p": {"models": models}}).issues
    assert issues.count == issues.damaged_models == issues.models == MAX_ISSUE_EXAMPLES + 3
    assert len(issues.examples) == MAX_ISSUE_EXAMPLES


def test_a_key_that_is_not_plain_text_is_quoted_in_a_path() -> None:
    """A path is logged, and a key is fetched data, so a line break or a long key cannot reach the log as written."""
    assert _issues({"p": {"models": {"a\nb": "M"}}}) == ("p.models.'a\\nb': expected an object",)
    (issue,) = _issues({"p": {"models": {"k" * 500: "M"}}})
    assert len(issue) < 120


def test_a_document_with_no_models_is_refused() -> None:
    assert (
        parse_catalog({"p": {"models": []}}).refusal() == "the document lists no models; p.models: expected an object"
    )
    assert parse_catalog({}).refusal() == "the document lists no models"


def test_a_document_is_refused_when_over_a_quarter_of_its_models_are_damaged() -> None:
    """One broken provider (the largest is about 7% of the catalog) passes; a format change upstream does not."""
    models: dict[str, Any] = {"a": {}, "b": {}, "c": {}, "d": {"name": 3}}
    assert parse_catalog({"p": {"models": models}}).refusal() is None

    models["c"] = "C"
    assert parse_catalog({"p": {"models": models}}).refusal() == (
        "2 of 4 models are damaged; p.models.c: expected an object; p.models.d.name: expected a string"
    )


def test_model_for_maps_the_any_llm_name_and_the_listing_prefix(catalog: ModelsDevCatalog) -> None:
    """Gemini lists ``models/<id>``; models.dev keys the bare id under ``google``."""
    assert models_dev_provider_id("gemini") == "google"
    assert models_dev_provider_id("openai") == "openai"
    assert catalog.model_for("gemini", "models/gemini-2.5-pro") is catalog.providers["google"].models["gemini-2.5-pro"]
    assert catalog.model_for("gemini", "gemini-2.5-pro") is not None
    assert catalog.model_for("gemini", "models/gemini-9") is None
    assert catalog.model_for("mystery", "gemini-2.5-pro") is None
    assert catalog.provider_for("gemini") is catalog.providers["google"]


def test_vertex_claude_is_found_where_v2_lists_it_and_where_v1_does() -> None:
    """v2 moved Vertex's Claude models under ``google-vertex``; v1 and older snapshots keep them apart."""
    assert models_dev_provider_ids("vertexaianthropic") == ("google-vertex-anthropic", "google-vertex")
    assert models_dev_provider_ids("gemini") == ("google",)
    claude = "claude-sonnet-4-5@20250929"
    for provider in ("google-vertex-anthropic", "google-vertex"):
        catalog = parse_catalog({provider: {"models": {claude: {"name": "Claude"}}}})
        assert catalog.provider_for("vertexaianthropic") is catalog.providers[provider]
        assert catalog.model_for("vertexaianthropic", claude) is catalog.providers[provider].models[claude]
    both = parse_catalog({"google-vertex-anthropic": {"models": {}}, "google-vertex": {"models": {claude: {}}}})
    assert both.model_for("vertexaianthropic", claude) is both.providers["google-vertex"].models[claude]


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
