"""The direct fetch service's rules that need no app: which instance, and what an unpriced fetch is told."""

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import (
    DirectFetchNotEnabledError,
    SearchToolMisconfiguredError,
    SearchToolSelectionError,
)
from gateway.schemas.tools import FetchRequest
from gateway.services.tools import pick_direct_fetch_instance, resolve_direct_fetch, unpriced_direct_fetch_detail

_TOOLS = {"fake-fetch": {"provider": "fake"}, "exa-fetch": {"provider": "exa", "api_key": "k"}}


def _config(**updates: Any) -> GatewayConfig:
    return GatewayConfig(web_fetch_enabled=True, fetch_tools=_TOOLS, **updates)


def test_the_pick_order() -> None:
    assert pick_direct_fetch_instance(_config(), "exa-fetch").name == "exa-fetch"
    assert pick_direct_fetch_instance(_config(web_fetch_default_tool="exa-fetch"), None).name == "exa-fetch"
    assert pick_direct_fetch_instance(_config(), None).name == "builtin_fetch"
    # A default naming no instance any-fetch serves counts as unset.
    assert pick_direct_fetch_instance(_config(web_fetch_default_tool="gone"), None).name == "builtin_fetch"
    assert pick_direct_fetch_instance(_config(), "builtin_fetch").provider == "builtin"


def test_a_disabled_fetch_is_refused_before_any_name_is_read() -> None:
    with pytest.raises(DirectFetchNotEnabledError):
        pick_direct_fetch_instance(GatewayConfig(fetch_tools=_TOOLS), "fake-fetch")


def test_an_unknown_name_is_refused_naming_the_available_instances() -> None:
    with pytest.raises(SearchToolSelectionError, match="Available: builtin_fetch, exa-fetch, fake-fetch"):
        pick_direct_fetch_instance(_config(), "nope")


def test_an_instance_without_the_key_its_provider_needs_is_refused() -> None:
    config = GatewayConfig(web_fetch_enabled=True, fetch_tools={"exa-fetch": {"provider": "exa"}})
    with pytest.raises(SearchToolMisconfiguredError, match="Fetch tool 'exa-fetch'"):
        pick_direct_fetch_instance(config, "exa-fetch")


def test_the_unpriced_detail_names_the_tools_own_row_and_offers_the_built_in_fetcher_no_config_entry() -> None:
    config = _config()
    request = FetchRequest(url="https://example.com/a")

    own = resolve_direct_fetch(config, pick_direct_fetch_instance(config, "fake-fetch"), None, request, named=True)
    detail = unpriced_direct_fetch_detail(own)
    assert detail.startswith("A fetch on 'fake-fetch' has no price the deployment's operator set")
    assert "using kind 'fetch' and the tool's name as model_key" in detail
    assert "price_per_call on its entry" in detail
    assert detail.endswith("Or set require_pricing to false to serve fetches unpriced.")

    builtin = resolve_direct_fetch(config, pick_direct_fetch_instance(config, None), None, request, named=False)
    builtin_detail = unpriced_direct_fetch_detail(builtin)
    assert builtin_detail.startswith("A fetch on 'builtin_fetch' has no price the deployment's operator set")
    assert "price_per_call" not in builtin_detail
