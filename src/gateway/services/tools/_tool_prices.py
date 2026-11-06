"""What ``require_pricing`` checks before a request's gateway-run tools may run.

A tool call is priced at settlement by its rate keys: an instance's own row
alone, or the tool's ``otari:<tool>`` row for a call on an organization key or
on the old backend, with only the operator's prices counting for a call the
deployment paid for. The check at admission
looks the same rows up the same way, so a request it admits settles priced. The
lookup needs the request's database session, so the caller makes it; this
module says which rows to look up, which requests skip the check, and what the
refusal says.
"""

from dataclasses import dataclass

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.tools import BUILTIN_FETCH, effective_fetch_instances, effective_search_instances
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tools._registry import BUILTIN_TOOLS
from gateway.services.tools._web_admission import AdmittedWebTools, enrichment_for
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME
from gateway.services.tools._web_search import InLoopSearch, ResolvedFetch, resolve_fetch, resolve_in_loop_search
from gateway.types.tool_charges import (
    InstanceRateKey,
    RateKey,
    describe_rate_key,
    gateway_tool_pricing_key,
    instance_rate_key,
)


@dataclass(frozen=True)
class PricedTool:
    """A gateway-run tool, the label its calls are recorded under, and the rows that price them, first match first.

    ``deployment_paid`` says the deployment's account runs the calls rather than a
    key the workspace's organization brought, as each call is recorded at settlement.
    ``enrichment`` marks the pages a search reads its hits from: fetch calls under
    the search's name, priced as the fetch instance that reads them.
    """

    name: str
    label: str | None
    rate_keys: tuple[RateKey, ...]
    deployment_paid: bool
    enrichment: bool = False


def _on_its_own_row(name: str, *, deployment_paid: bool) -> PricedTool:
    """A tool whose calls are recorded with no label, so ``otari:<tool>`` prices them, as the old backend's are."""
    return PricedTool(name, None, (gateway_tool_pricing_key(name),), deployment_paid)


def _priced_search(in_loop: InLoopSearch | None, *, on_credential: bool) -> PricedTool:
    """The search on the libraries under its label and rows, else on the old backend, under ``otari:web_search``.

    ``on_credential`` says the old backend searches with the workspace's own key.
    """
    if in_loop is None:
        return _on_its_own_row(WEB_SEARCH_TOOL_NAME, deployment_paid=not on_credential)
    search = in_loop.search
    return PricedTool(WEB_SEARCH_TOOL_NAME, search.label, search.rate_keys, not search.on_credential)


def _priced_fetch(fetch: ResolvedFetch) -> PricedTool:
    return PricedTool(WEB_FETCH_TOOL_NAME, fetch.label, fetch.rate_keys, not fetch.on_credential)


def _priced_enrichment(enrichment: ResolvedFetch | None) -> PricedTool | None:
    """The pages a search reads, under its name and what the fetch reading them is billed as.

    ``None`` where no page is read. Listed whether or not the request's fetch
    tool runs on the same instance, so an unpriced instance is refused as the
    search's pages whichever tool the request declared.
    """
    if enrichment is None:
        return None
    return PricedTool(
        WEB_SEARCH_TOOL_NAME, enrichment.label, enrichment.rate_keys, not enrichment.on_credential, enrichment=True
    )


def admitted_tools(*, code_execution: bool, web: AdmittedWebTools) -> list[PricedTool]:
    """The gateway-run tools one admitted request may call, in the registry's order, a search's pages after it.

    MCP tools are not among them: they run on a caller's server, so neither the
    deployment's prices nor its account are involved.
    """
    used: dict[str, list[PricedTool]] = {}
    if code_execution:
        # The sandbox is the deployment's, whichever backend runs it.
        used[CODE_EXECUTION_TOOL_NAME] = [_on_its_own_row(CODE_EXECUTION_TOOL_NAME, deployment_paid=True)]
    if web.search_tool_entry is not None:
        search = _priced_search(web.in_loop_search, on_credential=web.search_credential is not None)
        pages = _priced_enrichment(web.enrichment)
        used[WEB_SEARCH_TOOL_NAME] = [search, pages] if pages is not None else [search]
    if web.fetch is not None:
        used[WEB_FETCH_TOOL_NAME] = [_priced_fetch(web.fetch)]
    return [priced for tool in BUILTIN_TOOLS for priced in used.get(tool.name, ())]


def tools_to_price(
    *, code_execution: bool, web: AdmittedWebTools, require_pricing: bool, budget_exempt: bool
) -> list[PricedTool]:
    """The tools one admitted request runs that ``require_pricing`` checks, in the registry's order.

    The posture ``require_pricing`` applies to an unpriced model: the gateway
    does not run work it cannot meter, because a budget cannot restrain a charge
    that is never recorded. So none on a key exempt from budgets, which is never
    debited, as the model gate skips it, and none with ``require_pricing`` off.
    A tool whose calls carry no rate keys is settled by no row here and is left
    out.
    """
    if not require_pricing or budget_exempt:
        return []
    return [tool for tool in admitted_tools(code_execution=code_execution, web=web) if tool.rate_keys]


def configured_tools_to_price(config: GatewayConfig) -> list[PricedTool]:
    """Each tool this deployment can run, priced as a call with no workspace key would be, in the registry's order.

    What the startup warning checks, so it names the tools the check at
    admission would refuse.
    """
    priced: list[PricedTool] = []
    for tool in BUILTIN_TOOLS:
        if not tool.configured(config):
            continue
        if tool.name == WEB_SEARCH_TOOL_NAME:
            in_loop = resolve_in_loop_search(config, None)
            priced.append(_priced_search(in_loop, on_credential=False))
            if (pages := _priced_enrichment(enrichment_for(config, in_loop))) is not None:
                priced.append(pages)
        elif tool.name == WEB_FETCH_TOOL_NAME:
            priced.append(_priced_fetch(resolve_fetch(config, None, requested=None, for_search=None)))
        else:
            priced.append(_on_its_own_row(tool.name, deployment_paid=True))
    return priced


def enriching_fetch_tool(config: GatewayConfig) -> str | None:
    """The fetch tool that reads the pages of the deployment's in-loop search, as a call with no workspace key would.

    ``None`` where no page is read. Its row prices the pages, so the startup
    warning checks it whether or not the fetch tool is enabled.
    """
    enrichment = enrichment_for(config, resolve_in_loop_search(config, None))
    return enrichment.name if enrichment is not None else None


def deployment_tool_rate_keys(config: GatewayConfig) -> list[InstanceRateKey]:
    """The own rate key of each of the deployment's search and fetch tools, ``builtin_fetch`` included.

    Each is the row that prices its tool's calls. The startup warning reads it
    for every tool a start knows, ``config.yml`` and stored alike.
    """
    instances = (*effective_search_instances(config).values(), *effective_fetch_instances(config).values())
    return [instance_rate_key(instance.kind, instance.name, instance.provider) for instance in instances]


def unpriced_tool_detail(name: str, rate_key: RateKey | None = None, *, enrichment: bool = False) -> str:
    """The refusal of a request that would run ``name`` with no price, naming ``rate_key``, the row that prices it.

    With no ``rate_key``, it names the tool's ``otari:<tool>`` row. With
    ``enrichment``, the row is the fetch tool's that reads ``name``'s pages.
    ``price_per_call`` is offered for a tool a ``config.yml`` entry can declare,
    which ``builtin_fetch`` is not.
    """
    row = describe_rate_key(rate_key if rate_key is not None else gateway_tool_pricing_key(name))
    page = "Code execution pricing" if name == CODE_EXECUTION_TOOL_NAME else "Web search pricing"
    where = (
        f"POST {API_ROOT}/pricing, the dashboard's {page} page, or price_per_call on its entry in config.yml"
        if isinstance(rate_key, InstanceRateKey) and rate_key.name != BUILTIN_FETCH
        else f"POST {API_ROOT}/pricing, or the dashboard's {page} page"
    )
    unpriced = (
        f"The gateway tool '{name}' reads the pages of its results through a fetch tool that has no pricing"
        if enrichment
        else f"The gateway tool '{name}' has no pricing"
    )
    return (
        f"{unpriced}, and this gateway runs with "
        "require_pricing enabled, so it will not run work it cannot bill. Set a "
        f"per-request price for {row} ({where}), or set require_pricing to false to serve it unpriced."
        + (" With web_search_extract set to false, a search reads no page." if enrichment else "")
    )
