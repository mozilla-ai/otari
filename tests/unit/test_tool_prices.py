"""Which pricing rows ``require_pricing`` checks for a request's gateway-run tools, and for the deployment's."""

from dataclasses import replace

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.tools import (
    NO_WEB_TOOLS,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    AdmittedWebTools,
    PricedTool,
    builtin_fetch_resolved,
    configured_tools_to_price,
    resolve_fetch,
    resolve_in_loop_search,
    tools_to_price,
    unpriced_tool_detail,
)
from gateway.services.web_retrieval_policy import DomainPolicy
from gateway.types.tool_charges import instance_rate_key

_INSTANCE = "fake-search"
_FAKE_SEARCH = GatewayConfig(search_tools={_INSTANCE: {"provider": "fake"}})
_SEARCH_ROWS = (instance_rate_key("search", _INSTANCE, "fake"), "otari:web_search")
_SEARCH_ON_ITS_OWN_ROW = PricedTool(WEB_SEARCH_TOOL_NAME, None, ("otari:web_search",))
_BUILTIN_FETCH = PricedTool(
    WEB_FETCH_TOOL_NAME,
    "builtin:builtin_fetch",
    (instance_rate_key("fetch", "builtin_fetch", "builtin"), "otari:web_fetch"),
)
_CODE = PricedTool("code_execution", None, ("otari:code_execution",))


def _web(*, search: bool = False, on_library: bool = True, fetch: bool = False) -> AdmittedWebTools:
    return AdmittedWebTools(
        search_tool_entry={"type": "otari_web_search"} if search else None,
        fetch_tool_entry={"type": "otari_web_fetch"} if fetch else None,
        fetch_policy=DomainPolicy(),
        in_loop_search=resolve_in_loop_search(_FAKE_SEARCH, None) if search and on_library else None,
        fetch=builtin_fetch_resolved() if fetch else None,
    )


def _checked(web: AdmittedWebTools, *, code_execution: bool = False) -> list[PricedTool]:
    return tools_to_price(code_execution=code_execution, web=web, require_pricing=True, budget_exempt=False)


def test_a_search_on_an_instance_is_checked_under_its_label_and_rows() -> None:
    assert _checked(_web(search=True)) == [PricedTool(WEB_SEARCH_TOOL_NAME, f"fake:{_INSTANCE}", _SEARCH_ROWS)]


def test_a_search_the_old_backend_serves_is_checked_on_its_tools_own_row() -> None:
    assert _checked(_web(search=True, on_library=False)) == [_SEARCH_ON_ITS_OWN_ROW]


def test_the_fetch_tool_is_checked_on_the_instance_admission_resolved() -> None:
    assert _checked(_web(fetch=True)) == [_BUILTIN_FETCH]


def test_every_tool_a_request_runs_is_checked_in_the_registrys_order() -> None:
    checked = _checked(_web(search=True, on_library=False, fetch=True), code_execution=True)

    assert checked == [_SEARCH_ON_ITS_OWN_ROW, _BUILTIN_FETCH, _CODE]


@pytest.mark.parametrize(("require_pricing", "budget_exempt"), [(False, False), (True, True)])
def test_nothing_is_checked_without_require_pricing_or_on_a_key_exempt_from_budgets(
    require_pricing: bool, budget_exempt: bool
) -> None:
    web = _web(search=True, fetch=True)

    assert (
        tools_to_price(code_execution=True, web=web, require_pricing=require_pricing, budget_exempt=budget_exempt) == []
    )


def test_a_search_with_no_rate_keys_is_not_checked() -> None:
    web = _web(search=True)
    assert web.in_loop_search is not None
    settled_elsewhere = replace(web.in_loop_search, search=replace(web.in_loop_search.search, rate_keys=()))

    assert _checked(replace(web, in_loop_search=settled_elsewhere)) == []


def test_a_request_with_no_gateway_tools_checks_nothing() -> None:
    assert _checked(NO_WEB_TOOLS) == []


def test_the_deployments_search_and_fetch_are_checked_as_a_call_with_no_workspace_key() -> None:
    config = GatewayConfig(
        search_tools={_INSTANCE: {"provider": "fake"}},
        fetch_tools={"fake-fetch": {"provider": "fake"}},
        web_fetch_default_tool="fake-fetch",
        web_fetch_enabled=True,
    )
    fetch = resolve_fetch(config, None, requested=None, for_search=None)

    assert configured_tools_to_price(config) == [
        PricedTool(WEB_SEARCH_TOOL_NAME, f"fake:{_INSTANCE}", _SEARCH_ROWS),
        PricedTool(WEB_FETCH_TOOL_NAME, fetch.label, fetch.rate_keys),
    ]
    assert fetch.rate_keys == (instance_rate_key("fetch", "fake-fetch", "fake"), "otari:web_fetch")


def test_a_search_on_the_legacy_settings_is_checked_on_its_tools_own_row() -> None:
    config = GatewayConfig(web_search_url="http://searxng:8080")

    assert configured_tools_to_price(config) == [_SEARCH_ON_ITS_OWN_ROW]


def test_a_tool_the_deployment_cannot_run_is_not_checked() -> None:
    assert configured_tools_to_price(GatewayConfig(web_search_default_tool="none")) == []


def test_the_refusal_names_the_row_that_prices_any_of_the_tools_calls() -> None:
    detail = unpriced_tool_detail(WEB_SEARCH_TOOL_NAME)

    assert "'web_search' has no pricing" in detail
    assert "model_key 'otari:web_search'" in detail
    assert "require_pricing" in detail
