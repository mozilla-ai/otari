"""What ``require_pricing`` checks before a request's gateway-run tools may run.

A tool call is priced at settlement by its label and rate keys: an instance's
own row first, then the tool's ``otari:<tool>`` row, with only the operator's
prices counting for a call the deployment paid for. The check at admission
looks the same rows up the same way, so a request it admits settles priced. The
lookup needs the request's database session, so the caller makes it; this
module says which rows to look up, which requests skip the check, and what the
refusal says.
"""

from dataclasses import dataclass

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tools._registry import BUILTIN_TOOLS
from gateway.services.tools._web_admission import AdmittedWebTools
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME
from gateway.services.tools._web_search import InLoopSearch, resolve_fetch, resolve_in_loop_search
from gateway.types.tool_charges import RateKey, gateway_tool_pricing_key


@dataclass(frozen=True)
class PricedTool:
    """A gateway-run tool, the label its calls are recorded under, and the rows that price them, first match first."""

    name: str
    label: str | None
    rate_keys: tuple[RateKey, ...]


def _on_its_own_row(name: str) -> PricedTool:
    """A tool whose calls are recorded with no label, so ``otari:<tool>`` prices them, as the old backend's are."""
    return PricedTool(name, None, (gateway_tool_pricing_key(name),))


def _priced_search(in_loop: InLoopSearch | None) -> PricedTool:
    """The search on the libraries under its label and rows, else on the old backend, under ``otari:web_search``."""
    if in_loop is None:
        return _on_its_own_row(WEB_SEARCH_TOOL_NAME)
    return PricedTool(WEB_SEARCH_TOOL_NAME, in_loop.search.label, in_loop.search.rate_keys)


def tools_to_price(
    *, code_execution: bool, web: AdmittedWebTools, require_pricing: bool, budget_exempt: bool
) -> list[PricedTool]:
    """The tools one admitted request runs that ``require_pricing`` checks, in the registry's order.

    The posture ``require_pricing`` applies to an unpriced model: the gateway
    does not run work it cannot meter, because a budget cannot restrain a charge
    that is never recorded. So none on a key exempt from budgets, which is never
    debited, as the model gate skips it, and none with ``require_pricing`` off.
    A tool whose calls carry no rate keys is settled by no row here and is left
    out. MCP tools are not checked: their names come from a caller's server and
    are not something an operator can price in advance.
    """
    if not require_pricing or budget_exempt:
        return []
    used: dict[str, PricedTool] = {}
    if code_execution:
        used[CODE_EXECUTION_TOOL_NAME] = _on_its_own_row(CODE_EXECUTION_TOOL_NAME)
    if web.search_tool_entry is not None:
        used[WEB_SEARCH_TOOL_NAME] = _priced_search(web.in_loop_search)
    if web.fetch is not None:
        used[WEB_FETCH_TOOL_NAME] = PricedTool(WEB_FETCH_TOOL_NAME, web.fetch.label, web.fetch.rate_keys)
    return [used[tool.name] for tool in BUILTIN_TOOLS if tool.name in used and used[tool.name].rate_keys]


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
            priced.append(_priced_search(resolve_in_loop_search(config, None)))
        elif tool.name == WEB_FETCH_TOOL_NAME:
            fetch = resolve_fetch(config, None, requested=None, for_search=None)
            priced.append(PricedTool(tool.name, fetch.label, fetch.rate_keys))
        else:
            priced.append(_on_its_own_row(tool.name))
    return priced


def unpriced_tool_detail(name: str) -> str:
    """The refusal of a request that would run ``name`` with no price, naming the row that prices any of its calls."""
    key = gateway_tool_pricing_key(name)
    return (
        f"The gateway tool '{name}' has no pricing, and this gateway runs with "
        "require_pricing enabled, so it will not run work it cannot bill. Set a "
        f"per-request price for model_key '{key}' (POST {API_ROOT}/pricing, or the dashboard's "
        "Tools & Guardrails screen), or set require_pricing to false to serve it unpriced."
    )
