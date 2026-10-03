"""Catalog selection operates on all authorized summaries, before the page is sliced."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.schemas.catalog import CatalogCapabilities, CatalogModelSummary, CatalogQuery
from gateway.services.catalog import query_catalog


def model(identifier: str, **overrides: Any) -> CatalogModelSummary:
    fields: dict[str, Any] = {
        "id": identifier,
        "name": identifier,
        "vendor": None,
        "capabilities": CatalogCapabilities(),
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "offering_count": 1,
        "provider_count": 1,
        "providers": ["home_lab"],
        "selectors": [f"home_lab:{identifier}"],
        "price_sources": [],
        "unpriced_count": 1,
        "discovered": False,
    }
    return CatalogModelSummary(**(fields | overrides))


GLM = model(
    "z-ai/glm-5.3",
    name="GLM-5.3",
    vendor="Z.ai",
    capabilities=CatalogCapabilities(reasoning=True, tool_call=True, structured_output=True),
    context_window=200_000,
    release_date="2026-07-01",
    providers=["fireworks", "nebius"],
    provider_count=2,
    selectors=["fireworks:accounts/fireworks/models/glm-5p3", "nebius:zai-org/GLM-5.3"],
    price_sources=["defaults", "deployment"],
    min_input_price_per_million=0.5,
    min_output_price_per_million=2,
    unpriced_count=0,
    discovered=True,
    open_weights=True,
)
KIMI = model(
    "moonshotai/kimi-k2.6",
    name="Kimi K2.6",
    vendor="Moonshot AI",
    capabilities=CatalogCapabilities(tool_call=True),
    context_window=262_144,
    release_date="2026-05-01",
    input_modalities=["image", "text"],
    output_modalities=["image", "text"],
    providers=["nebius"],
    price_sources=["defaults"],
    min_input_price_per_million=0.6,
    min_output_price_per_million=2.4,
    unpriced_count=0,
    discovered=True,
)
LOCAL = model("qwen3-32b")
MODELS = [GLM, KIMI, LOCAL]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, MODELS),
        ({"search": "  MOONSHOT  "}, [KIMI]),
        ({"search": "glm-5"}, [GLM]),
        ({"search": "accounts/fireworks"}, [GLM]),
        ({"search": "nebius"}, [GLM, KIMI]),
        ({"search": "does-not-exist"}, []),
        ({"provider": ["fireworks"]}, [GLM]),
        ({"provider": ["fireworks", "home_lab"]}, [GLM, LOCAL]),
        ({"provider": ["missing"]}, []),
        ({"vendor": [""]}, [LOCAL]),
        ({"vendor": ["", "Z.ai"]}, [GLM, LOCAL]),
        ({"input_modality": ["image", "text"]}, [KIMI]),
        ({"input_modality": ["image", "audio"]}, []),
        ({"output_modality": ["image"]}, [KIMI]),
        ({"output_modality": ["image", "audio"]}, []),
        ({"capability": ["reasoning", "tool_call"]}, [GLM]),
        ({"capability": ["structured_output"]}, [GLM]),
        ({"capability": ["attachment"]}, []),
        ({"capability": ["open_weights"]}, [GLM]),
        ({"min_context": 250_000}, [KIMI]),
        ({"max_input": 0.5}, [GLM]),
        ({"pricing": "custom"}, [GLM]),
        ({"pricing": "default"}, [GLM, KIMI]),
        ({"pricing": "priced"}, [GLM, KIMI]),
        ({"pricing": "unpriced"}, [LOCAL]),
        ({"source": "custom"}, [LOCAL]),
        ({"source": "discovered"}, [GLM, KIMI]),
        ({"provider": ["nebius"], "capability": ["reasoning"], "max_input": 0.5}, [GLM]),
    ],
)
def test_filters_preserve_the_dashboard_semantics(filters: dict[str, Any], expected: list[CatalogModelSummary]) -> None:
    assert query_catalog(MODELS, CatalogQuery(**filters)).models == expected


def test_free_prices_and_organization_overrides_are_not_unknown() -> None:
    free = model("free", min_input_price_per_million=0, price_sources=["organization"], unpriced_count=0)
    assert query_catalog([LOCAL, free], CatalogQuery(max_input=0, pricing="custom")).models == [free]


def test_release_windows_include_the_boundary_but_not_unknown_or_future_dates() -> None:
    models = [model(str(day), release_date=day) for day in ("2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12")]
    now = datetime(2026, 9, 11, 23, 59, tzinfo=UTC)
    assert query_catalog([LOCAL, *models], CatalogQuery(released_within_days=1), now=now).models == models[1:3]
    assert query_catalog([LOCAL, *models], CatalogQuery(), now=now).models == [*models, LOCAL]


@pytest.mark.parametrize("column", ["input", "output", "context", "released"])
@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_unknown_sort_values_are_last_and_ties_are_stable(column: str, direction: str) -> None:
    tie = GLM.model_copy(update={"id": "z-ai/tie", "name": "Another model"})
    result = query_catalog(
        [LOCAL, GLM, KIMI, tie], CatalogQuery.model_validate({"sort": column, "direction": direction})
    ).models
    assert result[-1] == LOCAL
    assert result.index(tie) < result.index(GLM)
    assert (
        query_catalog(
            list(reversed([LOCAL, GLM, KIMI, tie])),
            CatalogQuery.model_validate({"sort": column, "direction": direction}),
        ).models
        == result
    )


def test_all_sort_columns_and_name_ties_have_deterministic_pages() -> None:
    same_name = GLM.model_copy(update={"id": "z-ai/a"})
    assert query_catalog([GLM, same_name], CatalogQuery()).models == [same_name, GLM]
    assert query_catalog([GLM, same_name], CatalogQuery(direction="desc")).models == [GLM, same_name]
    assert query_catalog(MODELS, CatalogQuery(sort="providers", direction="desc")).models == [GLM, KIMI, LOCAL]
    assert query_catalog(MODELS, CatalogQuery(sort="input", direction="desc")).models == [KIMI, GLM, LOCAL]
    assert query_catalog(MODELS, CatalogQuery(sort="released", direction="desc")).models == [GLM, KIMI, LOCAL]


def test_facets_list_every_authorized_choice_whatever_the_filters_match() -> None:
    facets = query_catalog(
        MODELS, CatalogQuery.model_validate({"capability": ["reasoning"], "include_facets": True, "limit": 1})
    ).facets
    assert facets is not None
    assert facets.total_count == 3
    assert facets.providers == ["fireworks", "home_lab", "nebius"]
    assert [facet.value for facet in facets.vendors] == ["", "Moonshot AI", "Z.ai"]
    assert next(facet for facet in facets.vendors if facet.value == "Z.ai").vendor_slug == "z-ai"
    empty = query_catalog(MODELS, CatalogQuery(search="no-match", include_facets=True)).facets
    assert empty == facets


def test_matches_and_choices_beyond_the_old_thousand_model_window_are_available() -> None:
    models = [model(f"model-{i:04}") for i in range(1205)] + [KIMI]
    query = CatalogQuery(provider=["nebius"], skip=0, limit=25, include_facets=True)
    page = query_catalog(models, query)
    assert page.models == [KIMI]
    assert page.count == 1
    facets = page.facets
    assert facets is not None
    assert facets.total_count == 1206
    assert "Moonshot AI" in [facet.value for facet in facets.vendors]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 1001},
        {"skip": -1},
        {"min_context": -1},
        {"max_input": -1},
        {"max_input": float("nan")},
        {"max_input": float("inf")},
        {"released_within_days": -1},
        {"released_within_days": 36501},
        {"sort": "invalid"},
        {"direction": "invalid"},
        {"capability": ["invalid"]},
        {"search": "x" * 201},
        {"provider": ["x"] * 101},
    ],
)
def test_invalid_queries_are_rejected(params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CatalogQuery(**params)
