"""The provider catalog's options: what only a tool's configuration may set."""

import pytest

import any_fetch
import any_search
from gateway.services.tools._search_provider_catalog import _option_schemas


def _served(kind: str, *, reports_cost: bool = False) -> dict[str, tuple[bool, bool]]:
    update = {"reports_cost": reports_cost}
    metadata: any_search.ProviderMetadata | any_fetch.ProviderMetadata = (
        any_search.AnySearch.get_provider_metadata("fake").model_copy(update=update)
        if kind == "search"
        else any_fetch.AnyFetch.get_provider_metadata("fake").model_copy(update=update)
    )
    return {option.name: (option.cost_tier, option.operator_only) for option in _option_schemas(metadata)}


@pytest.mark.parametrize("kind", ["search", "fetch"])
def test_a_cost_tier_is_served_as_set_only_by_a_tools_configuration_where_no_cost_is_reported(kind: str) -> None:
    served = _served(kind)
    assert served["depth"] == (True, True)
    assert served["account"] == (False, True)
    assert served["delay"] == (False, False)


@pytest.mark.parametrize("kind", ["search", "fetch"])
def test_a_cost_tier_is_served_open_where_the_provider_reports_the_cost(kind: str) -> None:
    served = _served(kind, reports_cost=True)
    assert served["depth"] == (True, False)
    assert served["account"] == (False, True)
