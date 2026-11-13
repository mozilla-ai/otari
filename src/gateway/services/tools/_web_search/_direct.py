"""Direct search: pick the instance, resolve the call, refuse an unpriced one, and run it.

``POST /api/v1/search`` calls the four functions below in that order, between
its own allowlist, budget and usage steps.

- :func:`pick_direct_search_instance` returns the instance named in the path
  or in ``search_tool_name``, else the one ``web_search_default_tool`` names
  when any-search serves it, else the only one. Where a call that names none
  finds no instance to pick, :func:`organization_key_search_instance` stands
  in for one when the workspace has an organization key, else
  :func:`legacy_search_instance` when the legacy settings describe a backend.
  A deployment with no instance serves that backend under its provider's name.
- :func:`resolve_direct_search` returns the :class:`DirectSearch`. A call that
  names its instance runs on the instance's own key. A call that names none
  runs on the organization key in the workspace's policy, when there is one,
  with that key's provider and options, under the rate key
  ``otari:web_search``. The count is capped at the policy's ``max_results``;
  the policy's provider options are not applied.
  :func:`resolve_organization_key_search` does the same for the stand-in.
- :func:`unpriced_direct_search_detail` names the row ``require_pricing``
  needs, for the 402 a direct tool call answers when none prices it.
- :func:`run_direct_search` calls the provider through any-search, or through
  the Tavily or Brave client for a key on those providers.

Whichever key serves the call, the instance supplies the allowlist entry, the
rate-limit rules, the timeout, the usage row's names and the response's
``search_tool``.
"""

from dataclasses import dataclass, replace
from decimal import Decimal

import httpx

import any_search
from gateway.core.config import API_ROOT
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    NO_SEARCH_DEFAULT,
    WEB_SEARCH_PROVIDERS,
    ToolInstance,
    ToolKind,
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
from gateway.services.tools._web_names import WEB_SEARCH_TOOL_NAME
from gateway.services.tools._web_provider_client import web_provider_client_for
from gateway.services.tools._web_search._default import direct_search_fallback
from gateway.services.tools._web_search._litellm import (
    SearchQuery,
    direct_count,
    litellm_native_options,
    litellm_results,
)
from gateway.services.tools._web_search._resolve import ResolvedSearch, resolve_search, resolve_search_on_key
from gateway.services.tools._web_tool_failure import WebToolFailure, failed_text
from gateway.types.tool_charges import InstanceRateKey, RateKey, describe_rate_key


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


@dataclass(frozen=True)
class DirectSearchResult:
    """What a direct search answered, and what the provider said it cost."""

    response: SearchResponse
    cost: Decimal | None


def pick_direct_search_instance(config: ToolSettings, name: str | None) -> ToolInstance:
    """The instance a direct search runs on: the one named, else the deployment's default.

    Raises :class:`SearchToolSelectionError` when no instance exists, the name
    matches none, or no name is given while several exist and
    ``web_search_default_tool`` names none of them; and
    :class:`SearchToolMisconfiguredError` for an instance nothing can run.
    """
    instances = effective_search_instances(config)
    if not instances:
        legacy = legacy_search_instance(config)
        if legacy is not None and name is not None:
            if name != legacy.name:
                raise SearchToolSelectionError(f"Unknown search tool '{name}'. Available: {legacy.name}.")
            return legacy
        # Names every way to add one, since a deployment may have no config file.
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


def legacy_search_instance(config: ToolSettings) -> ToolInstance | None:
    """The legacy settings' backend as the direct endpoint names it, or ``None`` where a search instance exists.

    Named after its provider, so a search instance of that name can later take its place unnoticed.
    """
    fallback = direct_search_fallback(config)
    if fallback is None:
        return None
    library_backed = fallback.provider in any_search.AnySearch.get_supported_providers()
    return ToolInstance(name=fallback.name, kind="search", provider=fallback.provider, library_backed=library_backed)


def unnamed_direct_search_tool(config: ToolSettings) -> str | None:
    """The search tool a direct search that names none runs on, or ``None`` when that search is refused.

    It is refused when there is no instance to pick and no legacy backend, or when the one picked cannot run.
    """
    try:
        return pick_direct_search_instance(config, None).name
    except SearchToolSelectionError:
        legacy = legacy_search_instance(config)
        return legacy.name if legacy is not None else None
    except SearchToolMisconfiguredError:
        return None


def direct_search_rate_key(config: ToolSettings, name: str) -> RateKey:
    """The price row a direct search on the servable tool ``name`` is billed under, for the deployment to pay."""
    resolved = resolve_search(config, None, requested=name, request_options=None)
    if resolved is None:
        # Unreachable: a named tool resolves or raises.
        raise SearchToolMisconfiguredError(name)
    return resolved.rate_keys[0]


def servable_direct_search_instances(config: ToolSettings) -> list[ToolInstance]:
    """The search tools a direct search may name and be served on, by name."""
    if (legacy := legacy_search_instance(config)) is not None:
        return [legacy]
    servable: list[ToolInstance] = []
    for name in sorted(effective_search_instances(config)):
        try:
            servable.append(pick_direct_search_instance(config, name))
        except (SearchToolSelectionError, SearchToolMisconfiguredError):
            continue
    return servable


def _default_instance(config: ToolSettings, instances: dict[str, ToolInstance]) -> str:
    """``web_search_default_tool`` when it names an instance any-search serves, else the only instance.

    A default that names no instance, or one on a provider any-search does not
    serve, counts as unset. ``none`` turns off only the in-loop tool's default,
    so here the only instance still serves.
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
    a call runs on the instance's own key, never the organization's. Raises
    :class:`WebSearchOptionRefusedError` for a request option the provider
    serving the call does not take, :class:`DirectSearchFieldRefusedError` for
    ``include_raw`` or ``provider_options`` on a call that does not run on
    any-search, and :class:`WebSearchCredentialRefusedError` for an
    organization key its provider cannot be called with.
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
    return _direct_search(instance, resolved, policy, request, organization_key)


def organization_key_search_instance(policy: ResolvedWebSearchConfig | None) -> ToolInstance | None:
    """What a direct search that names no tool runs as when there is no instance to pick, or ``None``.

    It stands in for an instance where the workspace has an organization key
    this gateway can search with: named ``web_search``, as the in-loop
    tool is, on the key's provider, so the key's tool list, the rate rules and
    the usage row read it as that tool.
    """
    credential = policy.credential if policy is not None else None
    if credential is None:
        return None
    library_backed = credential.provider in any_search.AnySearch.get_supported_providers()
    if not library_backed and credential.provider not in WEB_SEARCH_PROVIDERS:
        return None
    return ToolInstance(
        name=WEB_SEARCH_TOOL_NAME, kind="search", provider=credential.provider, library_backed=library_backed
    )


def resolve_organization_key_search(
    instance: ToolInstance, policy: ResolvedWebSearchConfig | None, request: SearchRequest
) -> DirectSearch:
    """:func:`resolve_direct_search` for the instance :func:`organization_key_search_instance` stood in."""
    credential = policy.credential if policy is not None else None
    resolved = (
        resolve_search_on_key(credential, name=instance.name, request_options=request.provider_options)
        if credential is not None
        else None
    )
    if resolved is None:
        # Unreachable: the stand-in exists only for a key its provider can be called with.
        raise SearchToolMisconfiguredError(instance.name)
    return _direct_search(instance, resolved, policy, request, credential)


def _direct_search(
    instance: ToolInstance,
    resolved: ResolvedSearch,
    policy: ResolvedWebSearchConfig | None,
    request: SearchRequest,
    organization_key: WebSearchCredential | None,
) -> DirectSearch:
    """The search ``resolved`` runs, with its count capped by the workspace's ``max_results``.

    Raises :class:`DirectSearchFieldRefusedError` for ``include_raw`` or
    ``provider_options`` on a call that does not run on any-search.
    """
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


def unpriced_direct_search_detail(search: DirectSearch) -> str:
    """The 402's message for a search ``require_pricing`` cannot bill: the row to add, and where.

    For an instance's own price, or the ``<provider>:<instance>`` row it was
    priced under before it had a kind, only the operator's price or override
    counts, since the deployment pays for the call. ``otari:web_search``, which
    prices an organization key's search and the legacy backend's, is named as such.
    """
    rate_key = search.resolved.rate_keys[0]
    return direct_unpriced_detail(
        "search",
        search.instance.name,
        rate_key,
        gateway_tool_row=search.resolved.on_credential or not isinstance(rate_key, InstanceRateKey),
    )


def direct_unpriced_detail(kind: ToolKind, tool: str, rate_key: RateKey, *, gateway_tool_row: bool) -> str:
    """The 402's message: the price ``rate_key`` names for a ``kind`` call on ``tool``, and where to set it.

    Both can be set on the dashboard's Web search pricing page or through the
    API, and an instance declared in the configuration file by its
    ``price_per_call``, which ``builtin_fetch`` has no entry for.
    ``gateway_tool_row`` says ``rate_key`` is an ``otari:<tool>`` row, which the message names.
    """
    if gateway_tool_row:
        missing = f"is priced at {describe_rate_key(rate_key)}, which has no pricing"
        how = (
            f"Set a per-request rate for it (POST {API_ROOT}/pricing, the pricing section of config.yml, "
            "or the dashboard's Web search pricing page)"
        )
    else:
        # The deployment pays for the call, so an organization's own override does not count.
        missing = "has no price the deployment's operator set"
        in_config = (
            "" if tool == BUILTIN_FETCH else ", or with price_per_call on its entry if it is declared in config.yml"
        )
        how = (
            f"Set one on the dashboard's Web search pricing page, with POST {API_ROOT}/pricing using kind "
            f"'{kind}' and the tool's name as model_key{in_config}"
        )
    plural = "searches" if kind == "search" else "fetches"
    return (
        f"A {kind} on '{tool}' {missing}, and require_pricing is on, so it is not run. "
        f"{how}; a zero rate is valid. Or set require_pricing to false to serve {plural} unpriced."
    )


async def run_direct_search(search: DirectSearch, client: httpx.AsyncClient) -> DirectSearchResult:
    """Run a direct search through the app's web provider client.

    The call runs on any-search, or on the Tavily or Brave client for a key
    on those providers: an organization's, or the legacy settings'. Raises :class:`DirectSearchFailedError`
    when the provider fails the call, an error inside a successful answer
    included. Its message, for the usage row and the log, holds only the
    failure's safe fields, since the Tavily and Brave clients' own text can
    quote the upstream body.
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
    raise DirectSearchFailedError(
        failed_text(failure.provider, "search", failure.tag, failure.status),
        headers=failure.as_headers(),
        upstream_status=failure.status,
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
        client=web_provider_client_for(client, tenant_endpoint=resolved.tenant_endpoint),
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
    from gateway.services.search_backend import run_keyed_search, run_provider_search

    resolved = search.resolved
    if search.credential is not None:
        outcome = await run_keyed_search(search.credential, search.query, timeout_s=resolved.timeout, client=client)
    else:
        # The legacy settings' Tavily or Brave key, which the deployment pays for.
        outcome = await run_provider_search(
            resolved.provider, resolved.api_key, search.query, timeout_s=resolved.timeout, client=client
        )
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
