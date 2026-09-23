"""The spellings a request may use for a model, and where they resolve."""

import uuid
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog_selectors as selectors
from gateway.services.provider_kwargs import resolve_provider_selector

_ROWS = [
    selectors.OfferingRow("nebius:zai-org/GLM-5.3", "nebius", "nebius", 0.5),
    selectors.OfferingRow("fireworks:accounts/fireworks/models/glm-5p3", "fireworks", "fireworks", 0.7),
    selectors.OfferingRow("fireworks:accounts/fireworks/models/glm-5p3-fp8", "fireworks", "fireworks", 0.6),
    selectors.OfferingRow("cerebras:gpt-oss-120b", "cerebras", "cerebras", None),
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


def test_a_cleaned_id_is_not_a_spelling(index: selectors.SelectorIndex) -> None:
    """Only the catalog id and the pinned form are accepted; ``instance:<cleaned id>`` goes to the provider as is."""
    assert selectors.resolve_catalog_selector("nebius:glm-5.3") is None
    assert selectors.resolve_catalog_selector("fireworks:glm-5p3") is None


def test_a_pinned_spelling_reaches_the_cheapest_build_on_its_instance(index: selectors.SelectorIndex) -> None:
    """``instance:<catalog id>`` names the model on one provider, and never leaves it."""
    assert index.pinned["nebius:z-ai/glm-5.3"] == "nebius:zai-org/GLM-5.3"
    assert index.pinned["fireworks:z-ai/glm-5.3"] == "fireworks:accounts/fireworks/models/glm-5p3-fp8"
    assert (
        selectors.resolve_catalog_selector("fireworks:z-ai/glm-5.3")
        == "fireworks:accounts/fireworks/models/glm-5p3-fp8"
    )
    assert (
        selectors.resolve_catalog_selector("Fireworks:Z-AI/GLM-5.3")
        == "fireworks:accounts/fireworks/models/glm-5p3-fp8"
    )
    assert selectors.resolve_catalog_selector("cerebras:z-ai/glm-5.3") is None


def test_the_catalog_advertises_the_pinned_spelling(index: selectors.SelectorIndex) -> None:
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") == "nebius:z-ai/glm-5.3"
    assert selectors.short_selector_for("fireworks:accounts/fireworks/models/glm-5p3-fp8") == "fireworks:z-ai/glm-5.3"
    # The dearer build is not what the pin reaches, so it advertises nothing.
    assert selectors.short_selector_for("fireworks:accounts/fireworks/models/glm-5p3") is None


def test_a_slug_resolves_to_the_cheapest_priced_offering(index: selectors.SelectorIndex) -> None:
    assert selectors.model_selector_for_slug("z-ai/glm-5.3") == "nebius:zai-org/GLM-5.3"
    # ``openai`` is a provider's own name, but OpenAI serves nothing here, so the
    # model is reached through the one provider that does.
    assert selectors.model_selector_for_slug("openai/gpt-oss-120b") == "cerebras:gpt-oss-120b"


def test_a_slug_whose_vendor_is_a_provider_prefers_that_provider() -> None:
    """The caller who names OpenAI's model while OpenAI serves it means OpenAI's price.

    Where OpenAI serves nothing, the model is reached through whoever resells
    it: the alternative is a selector that fails for a model the deployment
    demonstrably serves.
    """
    rows = [
        selectors.OfferingRow("azure:gpt-4o", "azure", "azure", 2.5),
        selectors.OfferingRow("openai-eu:gpt-4o", "openai-eu", "openai", 3.0),
    ]
    identities: dict[str, tuple[str, tuple[str, ...]]] = {
        "openai/gpt-4o": ("gpt-4o", ("azure:gpt-4o", "openai-eu:gpt-4o"))
    }

    azure_only = selectors.build_selector_index(rows[:1], identities)
    assert azure_only.models["openai/gpt-4o"] == "azure:gpt-4o"

    both = selectors.build_selector_index(rows, identities)
    # The cheaper offering is Azure's; the slug still reaches OpenAI's, because
    # that is the vendor the caller named and OpenAI serves it.
    assert both.models["openai/gpt-4o"] == "openai-eu:gpt-4o"


def test_a_vendor_is_matched_to_its_own_provider_by_the_identity_table() -> None:
    """``moonshotai`` is the vendor's id segment; ``moonshot`` is the provider it publishes on."""
    rows = [
        selectors.OfferingRow("nebius:moonshotai/Kimi-K3", "nebius", "nebius", 0.5),
        selectors.OfferingRow("moonshot:kimi-k3", "moonshot", "moonshot", 0.9),
    ]
    identities: dict[str, tuple[str, tuple[str, ...]]] = {
        "moonshotai/kimi-k3": ("kimik3", ("nebius:moonshotai/Kimi-K3", "moonshot:kimi-k3"))
    }
    assert selectors.build_selector_index(rows, identities).models["moonshotai/kimi-k3"] == "moonshot:kimi-k3"


def test_a_real_selector_is_never_rewritten(index: selectors.SelectorIndex) -> None:
    assert selectors.resolve_catalog_selector("cerebras:gpt-oss-120b") is None
    # The legacy slash spelling of a real offering is that offering, not a model id.
    assert selectors.resolve_catalog_selector("cerebras/gpt-oss-120b") is None
    assert selectors.resolve_catalog_selector("nebius:zai-org/GLM-5.3") is None
    assert selectors.resolve_catalog_selector("z-ai/glm-5.3") == "nebius:zai-org/GLM-5.3"
    assert selectors.resolve_catalog_selector("nope") is None


def test_the_resolver_relabels_a_pinned_or_model_selector_like_an_alias(index: selectors.SelectorIndex) -> None:
    config = GatewayConfig(
        master_key="k",
        providers={"nebius": {"api_key": "x"}, "fireworks": {"api_key": "x"}, "cerebras": {"api_key": "x"}},
    )
    pinned = resolve_provider_selector(config, "fireworks:z-ai/glm-5.3")
    assert (pinned.instance, pinned.model, pinned.alias) == (
        "fireworks",
        "accounts/fireworks/models/glm-5p3-fp8",
        "fireworks:z-ai/glm-5.3",
    )
    slug = resolve_provider_selector(config, "z-ai/glm-5.3")
    assert (slug.instance, slug.model, slug.alias) == ("nebius", "zai-org/GLM-5.3", "z-ai/glm-5.3")
    verbatim = resolve_provider_selector(config, "cerebras:gpt-oss-120b")
    assert (verbatim.model, verbatim.alias) == ("gpt-oss-120b", None)


def test_an_empty_index_resolves_nothing() -> None:
    selectors.reset_selector_index()
    assert selectors.resolve_catalog_selector("z-ai/glm-5.3") is None
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") is None


_ORG_A = uuid.uuid4()
_ORG_B = uuid.uuid4()
_WORKSPACE_A = uuid.uuid4()
_WORKSPACE_B = uuid.uuid4()
_BYO_DEEPSEEK = "nebius:deepseek-ai/DeepSeek-V4.1-Flash"


@pytest.fixture
def organization_index() -> Iterator[selectors.SelectorIndex]:
    """Organization A offers DeepSeek on its own nebius key; the deployment serves it nowhere."""
    own = [selectors.OfferingRow(_BYO_DEEPSEEK, "nebius", "nebius", 0.3)]
    identities = {
        **_IDENTITIES,
        "deepseek/deepseek-v4.1-flash": ("deepseekv4.1flash", (_BYO_DEEPSEEK,)),
    }
    view = selectors.build_organization_selectors(_ROWS, own, identities)
    built = selectors.build_selector_index(
        _ROWS,
        _IDENTITIES,
        organizations={_ORG_A: view},
        workspace_organization={_WORKSPACE_A: _ORG_A, _WORKSPACE_B: _ORG_B},
    )
    selectors.set_selector_index(built)
    yield built
    selectors.reset_selector_index()


def test_an_organization_reaches_its_own_offerings_by_every_spelling(
    organization_index: selectors.SelectorIndex,
) -> None:
    view = organization_index.organizations[_ORG_A]
    assert view.full == {_BYO_DEEPSEEK}
    for spelling in ("deepseek/deepseek-v4.1-flash", "nebius:deepseek/deepseek-v4.1-flash"):
        assert selectors.resolve_catalog_selector(spelling, organization_id=_ORG_A) == _BYO_DEEPSEEK
        assert selectors.resolve_catalog_selector(spelling, workspace_id=_WORKSPACE_A) == _BYO_DEEPSEEK
    assert selectors.resolve_catalog_selector(_BYO_DEEPSEEK, organization_id=_ORG_A) is None
    assert selectors.short_selector_for(_BYO_DEEPSEEK, organization_id=_ORG_A) == "nebius:deepseek/deepseek-v4.1-flash"
    assert selectors.model_selector_for_slug("deepseek/deepseek-v4.1-flash", organization_id=_ORG_A) == _BYO_DEEPSEEK
    # The deployment's own spellings still answer inside the organization's view.
    assert selectors.resolve_catalog_selector("z-ai/glm-5.3", organization_id=_ORG_A) == "nebius:zai-org/GLM-5.3"


def test_an_offering_on_one_organizations_key_resolves_for_nobody_else(
    organization_index: selectors.SelectorIndex,
) -> None:
    for spelling in ("deepseek/deepseek-v4.1-flash", "nebius:deepseek/deepseek-v4.1-flash"):
        assert selectors.resolve_catalog_selector(spelling) is None
        assert selectors.resolve_catalog_selector(spelling, organization_id=_ORG_B) is None
        assert selectors.resolve_catalog_selector(spelling, workspace_id=_WORKSPACE_B) is None
    assert selectors.short_selector_for(_BYO_DEEPSEEK) is None
    assert selectors.model_selector_for_slug("deepseek/deepseek-v4.1-flash", organization_id=_ORG_B) is None


def test_an_organizations_view_is_narrowed_to_what_its_offerings_touch() -> None:
    """A model the organization offers nothing of stays the deployment's to resolve."""
    own = [selectors.OfferingRow(_BYO_DEEPSEEK, "nebius", "nebius", 0.3)]
    identities = {**_IDENTITIES, "deepseek/deepseek-v4.1-flash": ("deepseekv4.1flash", (_BYO_DEEPSEEK,))}
    view = selectors.build_organization_selectors(_ROWS, own, identities)
    assert set(view.models) == {"deepseek/deepseek-v4.1-flash"}
    assert set(view.pinned) == {"nebius:deepseek/deepseek-v4.1-flash", "nebius:z-ai/glm-5.3"}
    assert "fireworks:z-ai/glm-5.3" not in view.pinned
    assert "cerebras:gpt-oss-120b" not in view.model_selectors


def test_an_organizations_cheaper_offering_wins_its_own_catalog_id() -> None:
    """The organization's rate decides the pick, so its own key answers where it is cheapest."""
    own = [selectors.OfferingRow("together:zai-org/GLM-5.3", "together", "together", 0.2)]
    identities = {
        **_IDENTITIES,
        "z-ai/glm-5.3": ("glm-5.3", (*_IDENTITIES["z-ai/glm-5.3"][1], "together:zai-org/GLM-5.3")),
    }
    view = selectors.build_organization_selectors(_ROWS, own, identities)
    assert view.models["z-ai/glm-5.3"] == "together:zai-org/GLM-5.3"
    assert view.pinned["together:z-ai/glm-5.3"] == "together:zai-org/GLM-5.3"
    # A pin on an instance the organization has no key for stays the deployment's to answer.
    assert "nebius:z-ai/glm-5.3" not in view.pinned


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

    with (
        patch.object(selector_index_service, "build_merged_catalog", new=AsyncMock(return_value=empty)) as build,
        patch.object(selector_index_service, "resolve_all_organizations_offered_keys", new=AsyncMock(return_value={})),
        patch.object(selector_index_service, "load_models_dev_catalog", new=AsyncMock(return_value=None)),
    ):
        await selector_index_service.rebuild_selector_index(AsyncMock(), config, fetch=fetch)

    assert build.await_args is not None
    assert build.await_args.kwargs["cached_only"] is expected_cached_only
    # The deployment's view carries no organization's offerings; each gets a view of its own.
    assert build.await_args.kwargs["include_offered"] is False
    # No port was handed in, so none is asked; the lifespan worker hands one in.
    assert build.await_args.kwargs["model_provider"] is None
