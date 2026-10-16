"""The direct endpoint's rules: which instance, which key, which count, and the call itself (design J).

``POST /api/v1/search`` reads the request and its dependencies, then asks
these. The instance comes from the path, else the body's
``search_tool_name``, else the deployment's default; with none or several,
contract C6's 400s. The instance supplies the allowlist entry, the rate-limit
rules, the timeout, the usage row's names and the response's ``search_tool``.
A named call runs on that instance with its own key. An unnamed one runs on
the organization key the workspace's policy carries, when it has one, on the
key's own provider and options, priced under ``otari:web_search`` (decisions
33 and 51). Of the workspace's policy, only the on/off switch, which the
route applies, and the ``max_results`` ceiling apply here; its provider
options do not (design G). The call runs on any-search, or on an old client
for a provider any-search has no adapter for yet.
"""

from dataclasses import dataclass, replace
from decimal import Decimal

import httpx

import any_search
from gateway.core.config import API_ROOT
from gateway.core.settings.tools import (
    NO_SEARCH_DEFAULT,
    ToolInstance,
    ToolSettings,
    configured_search_default,
    effective_search_instances,
)
from gateway.exceptions.tools_exceptions import (
    DirectSearchFailedError,
    DirectSearchFieldRefusedError,
    SearchToolMisconfiguredError,
    SearchToolSelectionError,
)
from gateway.models.tools import ResolvedWebSearchConfig, WebSearchCredential
from gateway.schemas.tools import SearchRequest, SearchResponse, SearchResultItem
from gateway.services.tools._web_search._litellm import (
    SearchQuery,
    direct_count,
    litellm_native_options,
    litellm_results,
)
from gateway.services.tools._web_search._resolve import ResolvedSearch, resolve_search
from gateway.services.tools._web_tool_failure import WebToolFailure


@dataclass(frozen=True)
class DirectSearch:
    """One direct search, resolved: what the route checks, bills and runs.

    ``credential`` is the organization key serving the call, ``None`` when the
    instance does. ``query`` carries the count the call asks for.
    """

    instance: ToolInstance
    resolved: ResolvedSearch
    query: SearchQuery
    credential: WebSearchCredential | None
    include_raw: bool

    @property
    def price_key(self) -> tuple[str, str]:
        """The provider and model of the pricing row that prices the call (design L).

        The instance's ``<provider>:<instance>``, or ``otari:web_search`` for an
        organization key, whose own provider a deployment cannot price.
        """
        provider, _, model = self.resolved.rate_keys[0].partition(":")
        return provider, model


@dataclass(frozen=True)
class DirectSearchResult:
    """What a direct search answered, and what the provider said it cost."""

    response: SearchResponse
    cost: Decimal | None


def pick_direct_search_instance(config: ToolSettings, name: str | None) -> ToolInstance:
    """The search instance a direct search runs on, named or the deployment's default.

    Raises :class:`SearchToolSelectionError` with contract C6's texts when no
    instance exists, the name matches none, or no name is given while several
    exist and ``web_search_default_tool`` names none of them; and
    :class:`SearchToolMisconfiguredError` for an instance nothing can run.
    """
    instances = effective_search_instances(config)
    if not instances:
        # Names every route in, because a message that named only config.yml
        # was unfollowable on a deployment that has no config file (issue #601).
        msg = (
            "No search tools are configured. Add one on the dashboard's Tools & Guardrails page "
            f"(or POST {API_ROOT}/search-tools), or declare one under 'search_tools' in your config file, "
            "supplied directly or through OTARI_CONFIG_YAML / OTARI_CONFIG_B64."
        )
        raise SearchToolSelectionError(msg)
    if name is None:
        name = _default_instance(config, instances)
    instance = instances.get(name)
    if instance is None:
        msg = f"Unknown search tool '{name}'. Available: {', '.join(sorted(instances))}."
        raise SearchToolSelectionError(msg)
    # Resolved here, before any workspace check, so an instance nothing can run
    # is refused whatever the workspace's policy says.
    resolve_search(config, None, requested=instance.name, request_options=None)
    if instance.library_backed and not instance.api_key and _requires_key(instance.provider):
        # Only an instance built in-process, or a stored one whose key cannot be decrypted, gets here.
        raise SearchToolMisconfiguredError(instance.name)
    return instance


def _requires_key(provider: str) -> bool:
    return any_search.AnySearch.get_provider_metadata(provider).requires_api_key


def _default_instance(config: ToolSettings, instances: dict[str, ToolInstance]) -> str:
    """``web_search_default_tool`` when it names an instance any-search serves, else the only instance.

    The default as design E, rule 7 reads it: one naming no instance, or one
    any-search does not serve yet, counts as unset. ``none`` concerns the
    in-loop tool only, so the only instance still serves.
    """
    named = configured_search_default(config)
    if named is not None and named != NO_SEARCH_DEFAULT:
        default = instances.get(named)
        if default is not None and default.library_backed:
            return named
    if len(instances) == 1:
        return next(iter(instances))
    msg = (
        "Several search tools are configured; name one in 'search_tool_name' or use "
        f"POST {API_ROOT}/search/{{tool}}. Available: {', '.join(sorted(instances))}."
    )
    raise SearchToolSelectionError(msg)


def resolve_direct_search(
    config: ToolSettings,
    instance: ToolInstance,
    policy: ResolvedWebSearchConfig | None,
    request: SearchRequest,
    *,
    named: bool,
) -> DirectSearch:
    """What a direct search on ``instance`` runs on, with which options and count.

    ``named`` says the caller named the instance, in the path or the body; such
    a call runs on the instance with its own key, never the organization's
    (decision 51). Raises :class:`WebSearchOptionRefusedError` for a request
    option the provider serving the call does not take,
    :class:`DirectSearchFieldRefusedError` for ``include_raw`` or
    ``provider_options`` on a call that does not run on any-search, and
    :class:`WebSearchCredentialRefusedError` for an organization key its
    provider cannot be called with.
    """
    organization_key = policy.credential if policy is not None and not named else None
    resolved = resolve_search(
        config,
        None,
        requested=instance.name,
        request_options=request.provider_options,
        direct_credential=organization_key,
    )
    if resolved is None:
        # Unreachable: a named instance resolves or raises.
        raise SearchToolMisconfiguredError(instance.name)
    if resolved.runner != "library" and (request.include_raw or request.provider_options):
        raise DirectSearchFieldRefusedError(resolved.provider)
    asked = SearchQuery(
        query=request.query,
        max_results=request.max_results,
        domain_filter=tuple(request.search_domain_filter or ()),
        country=request.country,
        max_tokens_per_page=request.max_tokens_per_page,
    )
    count = direct_count(
        resolved.provider,
        asked,
        layered=resolved.max_results,
        options=resolved.options,
        ceiling=policy.max_results if policy is not None else None,
    )
    return DirectSearch(
        instance=instance,
        resolved=resolved,
        query=replace(asked, max_results=count),
        credential=organization_key if resolved.on_credential else None,
        include_raw=request.include_raw,
    )


async def run_direct_search(search: DirectSearch, client: httpx.AsyncClient) -> DirectSearchResult:
    """Run a direct search on what serves it, through the app's web provider client.

    any-search for a library-backed provider; until their adapters exist, the
    old SearXNG client for a ``searxng`` tool and the old Tavily or Brave client
    for an organization key on either. Raises :class:`DirectSearchFailedError`
    when the provider fails the call, an error it signals inside a successful
    answer included. Its message, for the usage row and the log, holds only the
    failure's safe fields: an old client's own text can quote the upstream body,
    or a URL carrying the query.
    """
    # Imported here: the old clients' module imports this package's root.
    from gateway.services.search_backend import SearchProviderError

    try:
        if search.resolved.runner == "library":
            return await _on_any_search(search, client)
        return await _on_old_client(search, client)
    except any_search.ProviderError as exc:
        failure = WebToolFailure.from_provider_error(exc)
    except SearchProviderError as exc:
        failure = WebToolFailure.of(search.resolved.provider, exc.status, exc.tag)
    detail = failure.tag if failure.status is None else f"{failure.tag}, HTTP {failure.status}"
    raise DirectSearchFailedError(
        f"{failure.provider} search failed ({detail})", headers=failure.as_headers(), upstream_status=failure.status
    )


async def _on_any_search(search: DirectSearch, client: httpx.AsyncClient) -> DirectSearchResult:
    # The count is always passed as max_results, so a native count option cannot pass the workspace's ceiling.
    resolved = search.resolved
    options = {**resolved.options, **litellm_native_options(resolved.provider, search.query, resolved.options)}
    async with any_search.AnySearch.create(
        resolved.provider,
        api_key=resolved.api_key,
        api_base=resolved.api_base,
        timeout=resolved.timeout,
        client=client,
    ) as engine:
        result = await engine.search(search.query.query, max_results=search.query.max_results, **options)
    if result.error is not None:
        raise any_search.ProviderError(result.provider, result.error.status, result.error.tag)
    return DirectSearchResult(
        response=SearchResponse(
            search_tool=search.instance.name,
            results=litellm_results(resolved.provider, search.query, result.hits),
            raw=result.raw if search.include_raw else None,
        ),
        cost=result.cost,
    )


async def _on_old_client(search: DirectSearch, client: httpx.AsyncClient) -> DirectSearchResult:
    # Imported here, as in run_direct_search; looked up on each call, so a test that stubs a client still reaches it.
    from gateway.services.search_backend import run_keyed_search, run_search, search_tool

    if search.credential is not None:
        outcome = await run_keyed_search(
            search.credential, search.query, timeout_s=search.resolved.timeout, client=client
        )
    else:
        outcome = await run_search(search_tool(search.resolved), search.query, client=client)
    return DirectSearchResult(
        response=SearchResponse(
            search_tool=search.instance.name,
            results=[
                SearchResultItem(title=hit.title or "", url=hit.url, snippet=hit.snippet or "", date=hit.date)
                for hit in outcome.results
            ],
        ),
        cost=Decimal(str(outcome.cost_usd)) if outcome.cost_usd is not None else None,
    )
