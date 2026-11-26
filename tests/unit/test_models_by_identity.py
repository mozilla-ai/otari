"""The flat model listing folded into one row per model."""

from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog as selectors
from gateway.services.merged_catalog_service import ALIAS_OWNED_BY, ModelObject, ModelPricingInfo
from gateway.services.model_identity import group_offerings
from gateway.services.selector_index_service import answered_by_listing, models_by_identity, seed_for
from gateway.types.models_dev import ModelsDevCatalog, ModelsDevModel, ModelsDevProvider

_NEBIUS = "nebius:zai-org/GLM-5.3"
_FIREWORKS = "fireworks:accounts/fireworks/models/glm-5p3"


@pytest.fixture
def config() -> GatewayConfig:
    return GatewayConfig(
        master_key="k",
        providers={"nebius": {"api_key": "x"}, "fireworks": {"api_key": "x"}, "openai": {"api_key": "x"}},
    )


@pytest.fixture
def no_index() -> Iterator[None]:
    selectors.reset_selector_index()
    yield
    selectors.reset_selector_index()


def _offering(selector: str, input_price: float | None) -> ModelObject:
    pricing = (
        ModelPricingInfo(input_price_per_million=input_price, output_price_per_million=input_price * 2)
        if input_price is not None
        else None
    )
    return ModelObject(id=selector, created=0, owned_by=selector.partition(":")[0], pricing=pricing)


def test_every_provider_of_a_model_folds_into_one_row(config: GatewayConfig, no_index: None) -> None:
    listed = models_by_identity(
        config, None, [_offering(_NEBIUS, 0.5), _offering(_FIREWORKS, 0.7), _offering("openai:gpt-4o", 2.5)]
    )

    assert [(row.id, row.owned_by) for row in listed] == [("openai/gpt-4o", "openai"), ("z-ai/glm-5.3", "z-ai")]


def test_without_an_index_the_row_describes_the_cheapest_priced_offering(config: GatewayConfig, no_index: None) -> None:
    listed = models_by_identity(config, None, [_offering(_NEBIUS, None), _offering(_FIREWORKS, 0.7)])

    assert listed[0].pricing is not None
    assert listed[0].pricing.input_price_per_million == 0.7


def test_the_row_describes_the_offering_a_request_tries_first(config: GatewayConfig) -> None:
    """The index's order wins over price, so the row states what the request is billed at."""
    rows = [
        selectors.OfferingRow(_NEBIUS, "nebius", "nebius", 0.9),
        selectors.OfferingRow(_FIREWORKS, "fireworks", "fireworks", 0.7),
    ]
    selectors.set_selector_index(
        selectors.build_selector_index(rows, {"z-ai/glm-5.3": ("glm-5.3", (_NEBIUS, _FIREWORKS))})
    )
    try:
        first = selectors.resolve_catalog_offerings("z-ai/glm-5.3")[0]
        listed = models_by_identity(config, None, [_offering(_NEBIUS, 0.9), _offering(_FIREWORKS, 0.7)])
    finally:
        selectors.reset_selector_index()

    assert listed[0].pricing is not None
    assert listed[0].pricing.input_price_per_million == {_NEBIUS: 0.9, _FIREWORKS: 0.7}[first]


def test_an_offering_the_caller_cannot_see_is_never_the_one_described(config: GatewayConfig) -> None:
    """The index knows every offering; the row is chosen among the ones listed for this caller."""
    rows = [
        selectors.OfferingRow(_NEBIUS, "nebius", "nebius", 0.1),
        selectors.OfferingRow(_FIREWORKS, "fireworks", "fireworks", 0.7),
    ]
    selectors.set_selector_index(
        selectors.build_selector_index(rows, {"z-ai/glm-5.3": ("glm-5.3", (_NEBIUS, _FIREWORKS))})
    )
    try:
        listed = models_by_identity(config, None, [_offering(_FIREWORKS, 0.7)])
    finally:
        selectors.reset_selector_index()

    assert [row.id for row in listed] == ["z-ai/glm-5.3"]
    assert listed[0].pricing is not None
    assert listed[0].pricing.input_price_per_million == 0.7


def test_an_alias_target_a_request_tries_first_is_the_one_described(config: GatewayConfig) -> None:
    """The alias hides the target's name, not its place in the catalog id's dispatch order."""
    rows = [
        selectors.OfferingRow(_NEBIUS, "nebius", "nebius", 0.1),
        selectors.OfferingRow(_FIREWORKS, "fireworks", "fireworks", 0.7),
    ]
    selectors.set_selector_index(
        selectors.build_selector_index(rows, {"z-ai/glm-5.3": ("glm-5.3", (_NEBIUS, _FIREWORKS))})
    )
    try:
        first = selectors.resolve_catalog_offerings("z-ai/glm-5.3")[0]
        listed = models_by_identity(
            config, None, [_offering(_NEBIUS, 0.1), _offering(_FIREWORKS, 0.7)], alias_targets={_NEBIUS}
        )
    finally:
        selectors.reset_selector_index()

    assert first == _NEBIUS
    assert [row.id for row in listed] == ["z-ai/glm-5.3"]
    assert listed[0].pricing is not None
    assert listed[0].pricing.input_price_per_million == 0.1


def _limited(selector: str, input_price: float | None, context: int | None, output: int | None = None) -> ModelObject:
    return _offering(selector, input_price).model_copy(update={"context_window": context, "max_output_tokens": output})


def test_a_limit_the_first_offering_does_not_know_comes_from_the_others(config: GatewayConfig, no_index: None) -> None:
    """A self-hosted engine reports its context length while a priced offering tried before it knows none."""
    listed = models_by_identity(
        config,
        None,
        [_limited(_FIREWORKS, 0.7, None), _limited(_NEBIUS, None, 262144), _limited("openai:gpt-4o", 2.5, None)],
    )

    row = next(row for row in listed if row.id == "z-ai/glm-5.3")
    assert row.pricing is not None
    assert row.pricing.input_price_per_million == 0.7
    assert row.context_window == 262144
    assert row.max_output_tokens is None
    assert next(row for row in listed if row.id == "openai/gpt-4o").context_window is None


def test_a_limit_the_first_offering_knows_is_kept(config: GatewayConfig, no_index: None) -> None:
    listed = models_by_identity(
        config, None, [_limited(_FIREWORKS, 0.7, 131072, 16384), _limited(_NEBIUS, None, 262144, 32768)]
    )

    assert (listed[0].context_window, listed[0].max_output_tokens) == (131072, 16384)


def test_a_filled_limit_is_the_smallest_the_others_report_and_never_exceeds_the_window(
    config: GatewayConfig, no_index: None
) -> None:
    """Whichever offering a fallback reaches can hold it, and no output limit is larger than the context."""
    third = "together:zai-org/GLM-5.3"
    config.providers["together"] = {"api_key": "x"}
    listed = models_by_identity(
        config,
        None,
        [_limited(_FIREWORKS, 0.7, 65536), _limited(_NEBIUS, None, 262144, 131072), _limited(third, None, 200000)],
    )

    assert [row.id for row in listed] == ["z-ai/glm-5.3"]
    assert (listed[0].context_window, listed[0].max_output_tokens) == (65536, 65536)


def test_an_output_limit_the_first_offering_knows_is_not_capped_by_a_filled_window(
    config: GatewayConfig, no_index: None
) -> None:
    listed = models_by_identity(config, None, [_limited(_FIREWORKS, 0.7, None, 32768), _limited(_NEBIUS, None, 8192)])

    assert (listed[0].context_window, listed[0].max_output_tokens) == (8192, 32768)


def test_a_model_offered_only_by_alias_targets_is_left_to_the_alias(config: GatewayConfig, no_index: None) -> None:
    alias = ModelObject(id="fast", created=0, owned_by=ALIAS_OWNED_BY)

    listed = models_by_identity(config, None, [alias, _offering("openai:gpt-4o", 2.5)], alias_targets={"openai:gpt-4o"})

    assert [row.id for row in listed] == ["fast"]


def test_aliases_and_policies_are_listed_as_themselves(config: GatewayConfig, no_index: None) -> None:
    alias = ModelObject(id="fast", created=0, owned_by=ALIAS_OWNED_BY)
    policy = ModelObject(id="cheapest", created=0, owned_by="openai", pricing_source="dynamic")

    listed = models_by_identity(config, None, [alias, policy, _offering("openai:gpt-4o", 2.5)])

    assert [row.id for row in listed] == ["cheapest", "fast", "openai/gpt-4o"]


def test_a_name_an_alias_already_takes_is_left_to_the_alias(config: GatewayConfig, no_index: None) -> None:
    alias = ModelObject(id="openai/gpt-4o", created=0, owned_by=ALIAS_OWNED_BY)

    listed = models_by_identity(config, None, [alias, _offering("openai:gpt-4o", 2.5)])

    assert [(row.id, row.owned_by) for row in listed] == [("openai/gpt-4o", ALIAS_OWNED_BY)]


def test_an_organization_key_offering_is_listed_under_the_id_the_index_builds(
    config: GatewayConfig, no_index: None
) -> None:
    """A key slug is no configured instance, so only the slug map says which provider's models.dev entry names it."""
    model = ModelsDevModel(id="deepseek-flash", name="DeepSeek V4.1 Flash 1017")
    catalog = ModelsDevCatalog(
        providers={"deepseek": ModelsDevProvider(id="deepseek", name="DeepSeek", doc=None, models={model.id: model})}
    )
    slug_providers = {"acme-ds": "deepseek"}
    selector = "acme-ds:deepseek-flash"
    seed, _metadata, _provider_type = seed_for(config, catalog, selector, "acme-ds", "deepseek-flash", slug_providers)
    (indexed,) = group_offerings([seed]).values()

    listed = models_by_identity(
        config, catalog, [ModelObject(id=selector, created=0, owned_by="acme-ds")], slug_providers=slug_providers
    )

    assert [row.id for row in listed] == [indexed.id] == ["deepseek/deepseek-v4.1-flash-1017"]


def test_an_organization_key_offering_shows_the_metadata_its_slug_hides(config: GatewayConfig, no_index: None) -> None:
    """The offering is built before the slug map is known, so the folded row reads the model under its provider."""
    deepseek = ModelsDevModel(
        id="deepseek-flash",
        name="DeepSeek V4.1 Flash 1017",
        description="The October 17 snapshot of DeepSeek V4.1 Flash",
        release_date="2026-10-17",
        context_window=1048576,
        tool_call=True,
        reasoning=True,
    )
    described = ModelsDevModel(
        id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        name="DeepSeek V4.1 Flash 1017",
        context_window=1000000,
        max_output_tokens=384000,
        tool_call=True,
        reasoning=True,
    )
    catalog = ModelsDevCatalog(
        providers={
            "deepseek": ModelsDevProvider(id="deepseek", name="DeepSeek", doc=None, models={deepseek.id: deepseek}),
            "deepinfra": ModelsDevProvider(
                id="deepinfra", name="DeepInfra", doc=None, models={described.id: described}
            ),
        }
    )
    # Exactly what the merged catalog builds: each offering's metadata is read
    # under its own instance name, which for the slug names no models.dev provider.
    org = _offering("acme-ds:deepseek-flash", 0.3)
    selectors.apply_model_metadata(config, catalog, org)
    deployment = _offering("deepinfra:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.98)
    selectors.apply_model_metadata(config, catalog, deployment)

    listed = models_by_identity(config, catalog, [org, deployment], slug_providers={"acme-ds": "deepseek"})

    (row,) = listed
    assert row.pricing is not None
    assert row.pricing.input_price_per_million == 0.3
    assert row.name == "DeepSeek V4.1 Flash 1017"
    assert row.type == "chat"
    assert row.supported_parameters == ["tools", "tool_choice", "reasoning"]
    assert (row.description, row.release_date) == ("The October 17 snapshot of DeepSeek V4.1 Flash", "2026-10-17")


def test_an_organization_key_offering_is_read_whichever_ranking_it_gets(config: GatewayConfig, no_index: None) -> None:
    """Read under the slug's provider even when it ranks last, so the earliest date wins and a description is found."""
    deepseek = ModelsDevModel(
        id="deepseek-flash",
        name="DeepSeek V4.1 Flash 1017",
        description="The October 17 snapshot of DeepSeek V4.1 Flash",
        release_date="2026-10-17",
        knowledge="2025-05",
        context_window=1048576,
    )
    described = ModelsDevModel(
        id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        name="DeepSeek V4.1 Flash 1017",
        release_date="2026-11-20",
        context_window=1000000,
        max_output_tokens=384000,
    )
    catalog = ModelsDevCatalog(
        providers={
            "deepseek": ModelsDevProvider(id="deepseek", name="DeepSeek", doc=None, models={deepseek.id: deepseek}),
            "deepinfra": ModelsDevProvider(
                id="deepinfra", name="DeepInfra", doc=None, models={described.id: described}
            ),
        }
    )
    org = _offering("acme-ds:deepseek-flash", 0.9)
    selectors.apply_model_metadata(config, catalog, org)
    deployment = _offering("deepinfra:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3)
    selectors.apply_model_metadata(config, catalog, deployment)
    assert org.release_date is None

    listed = models_by_identity(config, catalog, [org, deployment], slug_providers={"acme-ds": "deepseek"})

    (row,) = listed
    assert row.pricing is not None
    assert row.pricing.input_price_per_million == 0.3
    assert (row.release_date, row.description, row.knowledge_cutoff) == (
        "2026-10-17",
        "The October 17 snapshot of DeepSeek V4.1 Flash",
        "2025-05",
    )
    # The row still takes its limits from the offering it is named after, as the org key's own are unread.
    assert (row.context_window, row.max_output_tokens) == (1000000, 384000)
    assert org.release_date is None


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [("z-ai/glm-5.3", True), ("my-policy", True), ("fast", False), (_NEBIUS, False)],
)
def test_only_a_catalog_id_or_policy_is_read_from_the_listing(model_id: str, expected: bool) -> None:
    """An alias and a selector have cheaper reads of their own, so the catalog is not built for them."""
    assert answered_by_listing(model_id, {"fast": _NEBIUS}) is expected


def test_a_folded_row_derives_its_dates_and_description_across_offerings(config: GatewayConfig, no_index: None) -> None:
    """The release date does not move with the ranking, and an undescribed first offering borrows a description."""
    tried_first = _offering(_NEBIUS, 0.5).model_copy(update={"release_date": "2026-10-20"})
    described = _offering(_FIREWORKS, 0.7).model_copy(
        update={"release_date": "2026-10-17", "description": "A sparse model", "knowledge_cutoff": "2025-05"}
    )

    (row,) = models_by_identity(config, None, [tried_first, described])

    assert (row.release_date, row.description, row.knowledge_cutoff) == ("2026-10-17", "A sparse model", "2025-05")


def test_a_folded_row_keeps_the_description_of_the_offering_it_is_named_after(
    config: GatewayConfig, no_index: None
) -> None:
    tried_first = _offering(_NEBIUS, 0.5).model_copy(update={"description": "Nebius's", "knowledge_cutoff": "2025-04"})
    other = _offering(_FIREWORKS, 0.7).model_copy(update={"description": "Fireworks's", "knowledge_cutoff": "2025-05"})

    (row,) = models_by_identity(config, None, [tried_first, other])

    assert (row.description, row.knowledge_cutoff) == ("Nebius's", "2025-04")
