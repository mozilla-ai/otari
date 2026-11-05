"""Which pricing rows ``require_pricing`` checks for a request's gateway-run tools, and for the deployment's.

The pages a search reads to enrich its hits are checked as fetches on the
instance that reads them, right after the search.
"""

from dataclasses import replace

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.tools import WebSearchCredential, WebToolCredentialSource
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tools import (
    NO_WEB_TOOLS,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    AdmittedWebTools,
    PricedTool,
    admitted_tools,
    builtin_fetch_resolved,
    configured_tools_to_price,
    enriching_fetch_tool,
    enrichment_for,
    resolve_fetch,
    resolve_in_loop_search,
    tools_to_price,
    unpriced_tool_detail,
)
from gateway.services.web_retrieval_policy import DomainPolicy
from gateway.types.tool_charges import instance_rate_key

_INSTANCE = "fake-search"
_FAKE_SEARCH = GatewayConfig(search_tools={_INSTANCE: {"provider": "fake"}})
# The same search, its hits' pages read through a fetch instance of its own.
_ENRICHED = GatewayConfig(
    search_tools={_INSTANCE: {"provider": "fake", "fetch_tool": "fake-fetch"}},
    fetch_tools={"fake-fetch": {"provider": "fake"}},
)
_SEARCH_ROWS = (instance_rate_key("search", _INSTANCE, "fake"),)
_SEARCH = PricedTool(WEB_SEARCH_TOOL_NAME, f"fake:{_INSTANCE}", _SEARCH_ROWS, True)
_PAGES = PricedTool(
    WEB_SEARCH_TOOL_NAME, "fake:fake-fetch", (instance_rate_key("fetch", "fake-fetch", "fake"),), True, enrichment=True
)
_SEARCH_ON_ITS_OWN_ROW = PricedTool(WEB_SEARCH_TOOL_NAME, None, ("otari:web_search",), True)
_BUILTIN_FETCH = PricedTool(
    WEB_FETCH_TOOL_NAME,
    "builtin:builtin_fetch",
    (instance_rate_key("fetch", "builtin_fetch", "builtin"),),
    True,
)
_CODE = PricedTool("code_execution", None, ("otari:code_execution",), True)
_ORG_KEY = WebSearchCredential(provider="tavily", source=WebToolCredentialSource.ORG, api_key="tvly-test")


def _web(
    *,
    search: bool = False,
    on_library: bool = True,
    fetch: bool = False,
    credential: WebSearchCredential | None = None,
    pages: bool = False,
    config: GatewayConfig = _FAKE_SEARCH,
) -> AdmittedWebTools:
    """A request's admitted web tools; with ``pages``, its searches' hits are enriched as ``config`` says."""
    in_loop = resolve_in_loop_search(config, None) if search and on_library else None
    return AdmittedWebTools(
        search_tool_entry={"type": "otari_web_search"} if search else None,
        fetch_tool_entry={"type": "otari_web_fetch"} if fetch else None,
        fetch_policy=DomainPolicy(),
        search_credential=credential,
        in_loop_search=in_loop,
        enrichment=in_loop.enrichment if pages and in_loop is not None else None,
        fetch=builtin_fetch_resolved() if fetch else None,
    )


def _checked(web: AdmittedWebTools, *, code_execution: bool = False) -> list[PricedTool]:
    return tools_to_price(code_execution=code_execution, web=web, require_pricing=True, budget_exempt=False)


def test_a_search_on_an_instance_is_checked_under_its_label_and_rows() -> None:
    assert _checked(_web(search=True)) == [_SEARCH]


def test_the_pages_a_search_reads_are_checked_after_it_as_fetches_on_the_instance_that_reads_them() -> None:
    assert _checked(_web(search=True, pages=True, config=_ENRICHED)) == [_SEARCH, _PAGES]


def test_the_pages_are_checked_beside_a_fetch_tool_on_another_instance() -> None:
    assert _checked(_web(search=True, pages=True, fetch=True, config=_ENRICHED)) == [_SEARCH, _PAGES, _BUILTIN_FETCH]


def test_the_pages_are_checked_before_a_fetch_tool_on_the_instance_that_reads_them() -> None:
    """Both are listed, the pages first, so an unpriced instance is refused as the search's pages."""
    pages = PricedTool(WEB_SEARCH_TOOL_NAME, "builtin:builtin_fetch", _BUILTIN_FETCH.rate_keys, True, enrichment=True)

    assert _checked(_web(search=True, pages=True, fetch=True)) == [_SEARCH, pages, _BUILTIN_FETCH]


def test_no_pages_are_checked_where_none_is_read() -> None:
    assert _checked(_web(search=True, config=_ENRICHED)) == [_SEARCH]


def test_the_pages_of_a_search_on_the_workspaces_own_key_read_on_the_deployments_fetch_are_deployment_paid() -> None:
    web = _web(search=True, pages=True, config=_ENRICHED)
    assert web.in_loop_search is not None
    on_key = replace(web.in_loop_search, search=replace(web.in_loop_search.search, on_credential=True))

    [search, pages] = _checked(replace(web, in_loop_search=on_key))

    assert (search.deployment_paid, pages.deployment_paid) == (False, True)


def test_pages_read_on_the_workspaces_own_fetch_key_are_not_deployment_paid() -> None:
    web = _web(search=True, pages=True, config=_ENRICHED)
    assert web.enrichment is not None
    on_key = replace(web.enrichment, name=None, label="org:fake", rate_keys=("otari:web_fetch",), on_credential=True)

    [_search, pages] = _checked(replace(web, enrichment=on_key))

    assert pages == PricedTool(WEB_SEARCH_TOOL_NAME, "org:fake", ("otari:web_fetch",), False, enrichment=True)


def test_a_search_reads_pages_through_its_enrichment_unless_enrichment_is_off() -> None:
    in_loop = resolve_in_loop_search(_ENRICHED, None)
    assert in_loop is not None

    assert enrichment_for(_ENRICHED, in_loop) is in_loop.enrichment
    assert enrichment_for(_ENRICHED.model_copy(update={"web_search_extract": False}), in_loop) is None
    assert enrichment_for(_ENRICHED, None) is None, "the old backend's pages are its own"


def test_a_search_the_old_backend_serves_is_checked_on_its_tools_own_row() -> None:
    assert _checked(_web(search=True, on_library=False)) == [_SEARCH_ON_ITS_OWN_ROW]


def test_a_search_the_old_backend_serves_on_the_workspaces_own_key_is_not_deployment_paid() -> None:
    assert _checked(_web(search=True, on_library=False, credential=_ORG_KEY)) == [
        replace(_SEARCH_ON_ITS_OWN_ROW, deployment_paid=False)
    ]


def test_a_search_the_libraries_run_on_the_workspaces_own_key_is_not_deployment_paid() -> None:
    web = _web(search=True)
    assert web.in_loop_search is not None
    on_key = replace(web.in_loop_search, search=replace(web.in_loop_search.search, on_credential=True))

    [search] = _checked(replace(web, in_loop_search=on_key))

    assert search.deployment_paid is False


def test_every_admitted_tool_is_listed_whatever_require_pricing_says() -> None:
    web = _web(search=True, on_library=False, fetch=True)

    assert admitted_tools(code_execution=True, web=web) == [_SEARCH_ON_ITS_OWN_ROW, _BUILTIN_FETCH, _CODE]


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
        _SEARCH,
        PricedTool(WEB_SEARCH_TOOL_NAME, fetch.label, fetch.rate_keys, True, enrichment=True),
        PricedTool(WEB_FETCH_TOOL_NAME, fetch.label, fetch.rate_keys, True),
    ]
    assert fetch.rate_keys == (instance_rate_key("fetch", "fake-fetch", "fake"),)


def test_the_fetch_tool_that_reads_the_deployments_pages_is_named_for_the_startup_warning() -> None:
    assert enriching_fetch_tool(_ENRICHED) == "fake-fetch"
    assert enriching_fetch_tool(_FAKE_SEARCH) == "builtin_fetch"
    assert enriching_fetch_tool(_ENRICHED.model_copy(update={"web_search_extract": False})) is None
    assert enriching_fetch_tool(GatewayConfig(web_search_default_tool="none")) is None


def test_a_search_on_the_legacy_provider_pair_is_checked_on_its_tools_own_row() -> None:
    config = GatewayConfig(web_search_provider="tavily", web_search_provider_api_key="tvly-key")

    assert configured_tools_to_price(config) == [_SEARCH_ON_ITS_OWN_ROW]


def test_a_tool_the_deployment_cannot_run_is_not_checked() -> None:
    assert configured_tools_to_price(GatewayConfig(web_search_default_tool="none")) == []
    assert configured_tools_to_price(GatewayConfig(search_tools={"local": {"provider": "searxng"}})) == []


def test_a_search_on_the_legacy_url_is_checked_on_otari_web_search_and_its_pages_on_the_fetch_tools_row() -> None:
    config = GatewayConfig(web_search_url="http://searxng:8080")

    assert configured_tools_to_price(config) == [
        PricedTool(WEB_SEARCH_TOOL_NAME, "otari:web_search", ("otari:web_search",), True),
        replace(_BUILTIN_FETCH, name=WEB_SEARCH_TOOL_NAME, enrichment=True),
    ]


def test_the_refusal_names_the_instances_own_row() -> None:
    detail = unpriced_tool_detail(WEB_SEARCH_TOOL_NAME, _SEARCH_ROWS[0])

    assert "'web_search' has no pricing" in detail
    assert f"kind 'search' and model_key '{_INSTANCE}'" in detail
    assert "Web search pricing page" in detail
    assert "price_per_call" in detail
    assert "require_pricing" in detail


def test_the_refusal_for_unpriced_pages_names_the_search_and_the_fetch_tools_row() -> None:
    detail = unpriced_tool_detail(WEB_SEARCH_TOOL_NAME, _PAGES.rate_keys[0], enrichment=True)

    assert "'web_search' reads the pages of its results through a fetch tool that has no pricing" in detail
    assert "kind 'fetch' and model_key 'fake-fetch'" in detail
    assert "price_per_call" in detail
    assert "web_search_extract" in detail


def test_the_builtin_fetch_refusal_offers_no_config_entry() -> None:
    detail = unpriced_tool_detail(WEB_FETCH_TOOL_NAME, _BUILTIN_FETCH.rate_keys[0])

    assert "kind 'fetch' and model_key 'builtin_fetch'" in detail
    assert "price_per_call" not in detail


def test_the_refusal_of_a_tool_on_its_own_row_names_that_row() -> None:
    detail = unpriced_tool_detail(WEB_SEARCH_TOOL_NAME)

    assert "model_key 'otari:web_search'" in detail


def test_the_code_execution_refusal_names_its_own_dashboard_page() -> None:
    detail = unpriced_tool_detail(CODE_EXECUTION_TOOL_NAME)

    assert "model_key 'otari:code_execution'" in detail
    assert "Code execution pricing page" in detail
