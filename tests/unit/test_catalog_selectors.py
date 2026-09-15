"""The short spellings a request may use for a model, and where they resolve."""

from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog_selectors as selectors
from gateway.services.provider_kwargs import resolve_provider_selector

_ROWS = [
    selectors.OfferingRow("nebius:zai-org/GLM-5.3", "nebius", "nebius", "glm-5.3", 0.5),
    selectors.OfferingRow("fireworks:accounts/fireworks/models/glm-5p3", "fireworks", "fireworks", "glm-5p3", 0.7),
    selectors.OfferingRow("fireworks:accounts/fireworks/models/glm-5p3-fp8", "fireworks", "fireworks", "glm-5p3", 0.6),
    selectors.OfferingRow("cerebras:gpt-oss-120b", "cerebras", "cerebras", "gpt-oss-120b", None),
]
_IDENTITIES = {
    "z-ai/glm-5.3": (
        "glm-5.3",
        (
            "nebius:zai-org/GLM-5.3",
            "fireworks:accounts/fireworks/models/glm-5p3",
            "fireworks:accounts/fireworks/models/glm-5p3-fp8",
        ),
    ),
    "openai/gpt-oss-120b": ("gpt-oss-120b", ("cerebras:gpt-oss-120b",)),
}


@pytest.fixture
def index() -> Iterator[selectors.SelectorIndex]:
    built = selectors.build_selector_index(_ROWS, _IDENTITIES)
    selectors.set_selector_index(built)
    yield built
    selectors.reset_selector_index()


def test_a_short_spelling_is_kept_only_where_it_is_unambiguous(index: selectors.SelectorIndex) -> None:
    assert index.short["nebius:glm-5.3"] == "nebius:zai-org/GLM-5.3"
    # Two Fireworks builds clean to the same id, so neither gets the short form.
    assert "fireworks:glm-5p3" not in index.short
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") == "nebius:glm-5.3"
    assert selectors.short_selector_for("fireworks:accounts/fireworks/models/glm-5p3") is None


def test_a_slug_resolves_to_the_cheapest_priced_offering(index: selectors.SelectorIndex) -> None:
    assert selectors.model_selector_for_slug("z-ai/glm-5.3") == "nebius:zai-org/GLM-5.3"
    # ``openai`` is a provider's own name, and Cerebras is not it, so the slug
    # is not indexed at all rather than sending an OpenAI request elsewhere.
    assert selectors.model_selector_for_slug("openai/gpt-oss-120b") is None


def test_a_slug_whose_vendor_is_a_provider_resolves_only_on_that_provider() -> None:
    """A vendor slug is also a legacy ``provider/model`` request for that provider.

    Reproduces otari#1015: with only ``azure`` configured, ``openai/gpt-4o``
    used to resolve to ``azure:gpt-4o`` and run on Azure credentials at Azure's
    price, because the slug's cheapest (and only) offering was Azure's.
    """
    rows = [
        selectors.OfferingRow("azure:gpt-4o", "azure", "azure", "gpt-4o", 2.5),
        selectors.OfferingRow("openai-eu:gpt-4o", "openai-eu", "openai", "gpt-4o", 3.0),
    ]
    identities: dict[str, tuple[str, tuple[str, ...]]] = {
        "openai/gpt-4o": ("gpt-4o", ("azure:gpt-4o", "openai-eu:gpt-4o"))
    }

    azure_only = selectors.build_selector_index(rows[:1], identities)
    assert "openai/gpt-4o" not in azure_only.models

    both = selectors.build_selector_index(rows, identities)
    # The cheaper offering is Azure's; the slug still reaches OpenAI's, because
    # that is the provider the caller named.
    assert both.models["openai/gpt-4o"] == "openai-eu:gpt-4o"


def test_a_real_selector_is_never_rewritten(index: selectors.SelectorIndex) -> None:
    assert selectors.resolve_catalog_selector("cerebras:gpt-oss-120b") is None
    # The legacy slash spelling of a real offering is that offering, not a model id.
    assert selectors.resolve_catalog_selector("cerebras/gpt-oss-120b") is None
    assert selectors.resolve_catalog_selector("openai/gpt-oss-120b") is None
    assert selectors.resolve_catalog_selector("nebius:glm-5.3") == "nebius:zai-org/GLM-5.3"
    assert selectors.resolve_catalog_selector("z-ai/glm-5.3") == "nebius:zai-org/GLM-5.3"
    assert selectors.resolve_catalog_selector("nope") is None


def test_the_resolver_relabels_a_short_or_model_selector_like_an_alias(index: selectors.SelectorIndex) -> None:
    config = GatewayConfig(
        master_key="k",
        providers={"nebius": {"api_key": "x"}, "fireworks": {"api_key": "x"}, "cerebras": {"api_key": "x"}},
    )
    short = resolve_provider_selector(config, "nebius:glm-5.3")
    assert (short.instance, short.model, short.alias) == ("nebius", "zai-org/GLM-5.3", "nebius:glm-5.3")
    slug = resolve_provider_selector(config, "z-ai/glm-5.3")
    assert (slug.instance, slug.model, slug.alias) == ("nebius", "zai-org/GLM-5.3", "z-ai/glm-5.3")
    verbatim = resolve_provider_selector(config, "cerebras:gpt-oss-120b")
    assert (verbatim.model, verbatim.alias) == ("gpt-oss-120b", None)


def test_an_empty_index_resolves_nothing() -> None:
    selectors.reset_selector_index()
    assert selectors.resolve_catalog_selector("z-ai/glm-5.3") is None
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") is None


@pytest.mark.parametrize(("fetch", "expected_cached_only"), [(False, True), (True, False)])
@pytest.mark.asyncio
async def test_only_an_operator_triggered_rebuild_may_dial(fetch: bool, expected_cached_only: bool) -> None:
    """The scheduled rebuild reads the discovery cache; the requested one may dial.

    The refresher runs on a timer against every configured provider, so dialing
    there would put a fanout the operator never asked for on that timer, and
    would dial even while ``model_cache_ttl_seconds`` is 0, whose whole meaning
    is that the reads dial for themselves.
    """
    from unittest.mock import AsyncMock, patch

    from gateway.services import selector_index_service
    from gateway.services.merged_catalog_service import MergedCatalog

    empty = MergedCatalog(models={}, aliases={}, dynamic_policies={}, discovered_keys=set())
    config = GatewayConfig(master_key="k", providers={"nebius": {"api_key": "x"}})

    with patch.object(selector_index_service, "build_merged_catalog", new=AsyncMock(return_value=empty)) as build:
        await selector_index_service.rebuild_selector_index(AsyncMock(), config, fetch=fetch)

    assert build.await_args is not None
    assert build.await_args.kwargs["cached_only"] is expected_cached_only
