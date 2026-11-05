"""The flat model listing folded into one row per model."""

from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog as selectors
from gateway.services.merged_catalog_service import ALIAS_OWNED_BY, ModelObject, ModelPricingInfo
from gateway.services.selector_index_service import answered_by_listing, models_by_identity

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


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [("z-ai/glm-5.3", True), ("my-policy", True), ("fast", False), (_NEBIUS, False)],
)
def test_only_a_catalog_id_or_policy_is_read_from_the_listing(model_id: str, expected: bool) -> None:
    """An alias and a selector have cheaper reads of their own, so the catalog is not built for them."""
    assert answered_by_listing(model_id, {"fast": _NEBIUS}) is expected
