"""The resolver: which provider, key and options one web search or fetch runs with (design G).

Search, in order:

1. ``requested``, the instance the direct endpoint picked, which only that
   endpoint sets. The instance runs with its own key, unless
   ``direct_credential``, the workspace's organization key for an unnamed
   call, is given: the key's provider, key and options then serve the call,
   and the instance keeps its name and timeout.
2. Else the policy's credential: the workspace's organization key, or in
   hybrid the control plane's.
3. Else the in-loop default (design E, rule 6).
4. Else nothing, which the caller turns into its refusal.

Fetch, in order: a named fetch instance; for enrichment, the search
instance's ``fetch_tool``; the policy's fetch credential; the fetch default,
which falls back to ``builtin_fetch``, so fetch always resolves.

A credential whose provider this gateway cannot call is ignored with a
warning. A credential never takes what it lacks from the server: the result
carries its own key and base URL, ``""`` where it has none, which the
libraries read as passed rather than fall back to the environment, and a
credential without a key or base URL its provider needs is refused.

Options are layered lowest first: otari's defaults for the provider, the
instance's, the credential's, the workspace's and the request's. The direct
endpoint leaves out the workspace's, and the instance's when an organization
key serves the call. Fetch takes neither a workspace nor a request layer,
since both hold search options. The count the layers name is taken out of
the options, for the caller to apply as its door does. Where a library
runs the call, each option is checked as the instance's are (design E): a
request's option the provider does not take, or takes only from the
deployment, is refused, and the other layers' are dropped with a warning that
names the layer and the key, never the value. What the old code runs takes
the layers unchecked and without otari's defaults, which that code applies
itself.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import any_fetch
import any_search
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    BUILTIN_FETCH_PROVIDER,
    PROVIDER_DEFAULT_OPTIONS,
    SEARCH_PROVIDERS_REQUIRING_API_BASE,
    SEARCH_PROVIDERS_WITHOUT_ADAPTER,
    WEB_SEARCH_PROVIDERS,
    SynthesizedSearchInstance,
    ToolInstance,
    ToolKind,
    ToolSettings,
    configured_fetch_default,
    default_api_base,
    effective_fetch_instances,
    effective_search_instances,
    in_loop_default,
    option_problems,
    validate_search_tool_transport,
)
from gateway.exceptions.tools_exceptions import (
    SearchToolMisconfiguredError,
    SearchToolNotFoundError,
    WebSearchCredentialRefusedError,
    WebSearchOptionRefusedError,
)
from gateway.log_config import logger
from gateway.models.tools import (
    ResolvedWebSearchConfig,
    WebFetchCredential,
    WebSearchCredential,
    WebToolCredential,
    WebToolCredentialSource,
)
from gateway.services.tools._web_names import WEB_FETCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME
from gateway.services.web_retrieval_network import NETWORK_DEADLINE_SECONDS

# What runs a resolved call: one of the libraries, or the old code, which in
# the loop is the old backend and on the direct endpoint the old SearXNG client
# or the organization key's own search.
Runner = Literal["library", "old"]

# How long a call may take when its instance sets no timeout: 30 seconds on
# the direct endpoint and 15 in the loop. The built-in fetcher keeps its own
# deadline.
DIRECT_TIMEOUT_S = 30.0
IN_LOOP_TIMEOUT_S = 15.0

# The rows that price any backend of each tool (design L), under the key the
# pricing service gives a gateway-run tool. Spelled out here rather than imported,
# since the pricing modules import this package.
_SEARCH_FALLBACK_RATE = f"otari:{WEB_SEARCH_TOOL_NAME}"
_FETCH_FALLBACK_RATE = f"otari:{WEB_FETCH_TOOL_NAME}"
# The label of a call an organization key serves, before the provider: never a pricing row.
_ORG_LABEL_PREFIX = "org"
_NO_OPTIONS: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, kw_only=True)
class _Resolved:
    """What both results carry.

    ``name`` is the instance's, ``None`` where a credential serves an in-loop
    call. ``api_key`` and ``api_base`` go to the library as they are, ``""``
    meaning none. ``options`` are the layered options without the count, for
    the caller to apply as its door does (design G, item 3). ``label`` and
    ``rate_keys`` are what billing records (design L). ``on_credential`` says
    a credential serves the call rather than an instance.
    """

    name: str | None
    provider: str
    runner: Runner
    api_key: str = field(repr=False)
    api_base: str = field(repr=False)
    options: Mapping[str, Any] = field(repr=False, hash=False)
    timeout: float
    label: str | None
    rate_keys: tuple[str, ...]
    on_credential: bool = False


@dataclass(frozen=True, kw_only=True)
class ResolvedSearch(_Resolved):
    """What one web search runs on: the count is ``max_results``, and ``fetch_tool`` the instance's enrichment."""

    max_results: int | None
    fetch_tool: str | None = None


@dataclass(frozen=True, kw_only=True)
class ResolvedFetch(_Resolved):
    """What one web fetch runs on, always through any-fetch: the count is ``max_chars``."""

    max_chars: int | None


def resolve_search(
    config: ToolSettings,
    policy: ResolvedWebSearchConfig | None,
    *,
    requested: str | None,
    request_options: Mapping[str, Any] | None,
    direct_credential: WebSearchCredential | None = None,
) -> ResolvedSearch | None:
    """The search one call runs on, or ``None`` when there is nothing to search with.

    The direct endpoint passes the instance it picked as ``requested`` and,
    for an unnamed call, the workspace's organization key as
    ``direct_credential``; the in-loop tool passes neither.

    Raises :class:`WebSearchOptionRefusedError` for a request option the
    provider does not take from a request, :class:`WebSearchCredentialRefusedError`
    for a credential its provider cannot be called with,
    :class:`SearchToolNotFoundError` for a ``requested`` name no search instance
    has, which the direct endpoint answers before it calls this, and
    :class:`SearchToolMisconfiguredError` for a ``requested`` instance nothing can run.
    """
    request = request_options or _NO_OPTIONS
    if requested is not None:
        instance = effective_search_instances(config).get(requested)
        if instance is None:
            raise SearchToolNotFoundError(requested)
        timeout = instance.timeout or DIRECT_TIMEOUT_S
        if direct_credential is not None:
            on_key = _search_on_credential(
                direct_credential, name=instance.name, in_loop=False, timeout=timeout, request=request
            )
            if on_key is not None:
                return on_key
        return _search_on_instance(config, instance, in_loop=False, request=request)
    workspace = (policy.provider_options if policy is not None else None) or _NO_OPTIONS
    if policy is not None and policy.credential is not None:
        on_key = _search_on_credential(
            policy.credential, name=None, in_loop=True, timeout=IN_LOOP_TIMEOUT_S, workspace=workspace, request=request
        )
        if on_key is not None:
            return on_key
    default = in_loop_default(config)
    if isinstance(default, SynthesizedSearchInstance):
        return _search_on_synthesized(default, workspace=workspace, request=request)
    if default is not None:
        return _search_on_instance(config, default, in_loop=True, workspace=workspace, request=request)
    return None


def resolve_fetch(
    config: ToolSettings,
    policy: ResolvedWebSearchConfig | None,
    *,
    requested: str | None,
    for_search: ResolvedSearch | None,
) -> ResolvedFetch:
    """The fetch one call runs on: the fetch tool's when ``for_search`` is ``None``, else that search's enrichment.

    ``requested`` names a fetch instance, which then runs with its own key.
    Raises :class:`WebSearchCredentialRefusedError` for a fetch credential
    without what its provider needs, and :class:`SearchToolNotFoundError` for
    a ``requested`` name no fetch instance has.
    """
    instances = _fetch_instances(config)
    if requested is not None:
        if (named := instances.get(requested)) is None:
            raise SearchToolNotFoundError(requested)
        return _fetch_on_instance(named)
    if for_search is not None and for_search.fetch_tool is not None:
        if (chosen := instances.get(for_search.fetch_tool)) is not None:
            return _fetch_on_instance(chosen)
    if policy is not None and policy.fetch_credential is not None:
        if (on_key := _fetch_on_credential(policy.fetch_credential)) is not None:
            return on_key
    return _fetch_on_instance(instances.get(configured_fetch_default(config)) or instances[BUILTIN_FETCH])


def _fetch_instances(config: ToolSettings) -> dict[str, ToolInstance]:
    """The fetch instances any-fetch can run, each other one left out with a warning, as if it named nothing."""
    served = set(any_fetch.AnyFetch.get_supported_providers())
    instances: dict[str, ToolInstance] = {}
    for name, instance in effective_fetch_instances(config).items():
        if instance.provider in served:
            instances[name] = instance
        else:
            _warn_once(
                "Fetch instance '%s' is left out: any-fetch does not serve its provider %s", name, instance.provider
            )
    return instances


def _search_on_instance(
    config: ToolSettings,
    instance: ToolInstance,
    *,
    in_loop: bool,
    workspace: Mapping[str, Any] = _NO_OPTIONS,
    request: Mapping[str, Any],
) -> ResolvedSearch:
    runner: Runner = "library" if instance.library_backed else "old"
    own = dict(instance.options)
    # Contract C8: a searxng instance that names no engines searches the deployment's.
    if instance.provider in SEARCH_PROVIDERS_REQUIRING_API_BASE and "engines" not in own and config.web_search_engines:
        own["engines"] = config.web_search_engines
    options = _layered(
        "search",
        instance.provider,
        checked=runner == "library",
        defaults=instance.provider_defaults,
        instance=(f"search instance '{instance.name}'", own, instance.dropped_options),
        workspace=workspace,
        request=request,
    )
    api_base = _base(instance.api_base or default_api_base(config, instance.provider))
    if runner == "old" and (instance.provider not in SEARCH_PROVIDERS_WITHOUT_ADAPTER or not api_base):
        # A provider no code here serves any more, or a searxng instance with no address of its own or to inherit.
        raise SearchToolMisconfiguredError(instance.name)
    try:
        # Load checks the instance's own base; an inherited one is checked here.
        validate_search_tool_transport(instance.name, api_base, instance.api_key)
    except ValueError as exc:
        raise SearchToolMisconfiguredError(instance.name, str(exc)) from None
    max_results = _count(options, "max_results")
    label = f"{instance.provider}:{instance.name}"
    return ResolvedSearch(
        name=instance.name,
        provider=instance.provider,
        runner=runner,
        api_key=instance.api_key or "",
        api_base=api_base,
        options=MappingProxyType(options),
        max_results=max_results,
        timeout=instance.timeout or (IN_LOOP_TIMEOUT_S if in_loop else DIRECT_TIMEOUT_S),
        label=label,
        # On the direct endpoint, the instance's own row alone (contract C5).
        rate_keys=(label, _SEARCH_FALLBACK_RATE) if in_loop else (label,),
        fetch_tool=instance.fetch_tool,
    )


def _search_on_synthesized(
    synthesized: SynthesizedSearchInstance, *, workspace: Mapping[str, Any], request: Mapping[str, Any]
) -> ResolvedSearch:
    runner: Runner = "library" if synthesized.provider in any_search.AnySearch.get_supported_providers() else "old"
    legacy = {"engines": ",".join(synthesized.engines)} if synthesized.engines else {}
    options = _layered(
        "search",
        synthesized.provider,
        checked=runner == "library",
        defaults=synthesized.provider_defaults,
        instance=("legacy search settings", legacy, frozenset()),
        workspace=workspace,
        request=request,
    )
    max_results = _count(options, "max_results")
    return ResolvedSearch(
        name=synthesized.name,
        provider=synthesized.provider,
        runner=runner,
        api_key=synthesized.api_key or "",
        api_base=_base(synthesized.api_base),
        options=MappingProxyType(options),
        max_results=max_results,
        timeout=IN_LOOP_TIMEOUT_S,
        label=_SEARCH_FALLBACK_RATE,
        rate_keys=(_SEARCH_FALLBACK_RATE,),
    )


def _search_on_credential(
    credential: WebSearchCredential,
    *,
    name: str | None,
    in_loop: bool,
    timeout: float,
    workspace: Mapping[str, Any] = _NO_OPTIONS,
    request: Mapping[str, Any],
) -> ResolvedSearch | None:
    """The search a credential serves, or ``None`` for one whose provider this gateway cannot call."""
    provider = credential.provider
    runner: Runner
    if provider in any_search.AnySearch.get_supported_providers():
        runner = "library"
        metadata = any_search.AnySearch.get_provider_metadata(provider)
        requires_key, default_base = metadata.requires_api_key, metadata.default_api_base
        requires_base = metadata.requires_api_base and not default_base
    elif provider in (*WEB_SEARCH_PROVIDERS, *SEARCH_PROVIDERS_WITHOUT_ADAPTER):
        # Tavily, Brave and SearXNG until their adapters exist (decision 28).
        runner = "old"
        requires_key, default_base = provider in WEB_SEARCH_PROVIDERS, None
        requires_base = provider in SEARCH_PROVIDERS_REQUIRING_API_BASE
    else:
        _ignore(credential, "search")
        return None
    _check(credential, requires_key=requires_key, requires_base=requires_base)
    if runner == "old" and provider in WEB_SEARCH_PROVIDERS and _base(credential.api_base):
        # The old code calls Tavily and Brave at their public endpoints, where the key would go instead.
        raise WebSearchCredentialRefusedError(
            provider, f"names a base URL, which this gateway's {provider} client cannot use"
        )
    options = _layered(
        "search",
        provider,
        checked=runner == "library",
        defaults=PROVIDER_DEFAULT_OPTIONS.get(provider, _NO_OPTIONS),
        credential=(_credential_layer(credential), credential.options),
        workspace=workspace,
        request=request,
    )
    max_results = _count(options, "max_results")
    label, rate_keys = _credential_billing(credential, _SEARCH_FALLBACK_RATE)
    if in_loop and runner == "old":
        # Design L: in the loop, the old backend records an organization key's calls with no label.
        label = None
    return ResolvedSearch(
        name=name,
        provider=provider,
        runner=runner,
        api_key=credential.api_key or "",
        api_base=_base(credential.api_base or default_base),
        options=MappingProxyType(options),
        max_results=max_results,
        timeout=timeout,
        label=label,
        rate_keys=rate_keys,
        on_credential=True,
    )


def _fetch_on_instance(instance: ToolInstance) -> ResolvedFetch:
    options = _layered(
        "fetch",
        instance.provider,
        checked=True,
        defaults=instance.provider_defaults,
        instance=(f"fetch instance '{instance.name}'", instance.options, instance.dropped_options),
    )
    max_chars = _count(options, "max_chars")
    label = f"{instance.provider}:{instance.name}"
    return ResolvedFetch(
        name=instance.name,
        provider=instance.provider,
        runner="library",
        api_key=instance.api_key or "",
        api_base=_base(
            instance.api_base or any_fetch.AnyFetch.get_provider_metadata(instance.provider).default_api_base
        ),
        options=MappingProxyType(options),
        max_chars=max_chars,
        timeout=instance.timeout
        or (NETWORK_DEADLINE_SECONDS if instance.provider == BUILTIN_FETCH_PROVIDER else IN_LOOP_TIMEOUT_S),
        label=label,
        rate_keys=(label, _FETCH_FALLBACK_RATE),
    )


def _fetch_on_credential(credential: WebFetchCredential) -> ResolvedFetch | None:
    """The fetch a credential serves, or ``None`` for one whose provider this gateway cannot call."""
    provider = credential.provider
    if provider == BUILTIN_FETCH_PROVIDER or provider not in any_fetch.AnyFetch.get_supported_providers():
        _ignore(credential, "fetch")
        return None
    metadata = any_fetch.AnyFetch.get_provider_metadata(provider)
    _check(
        credential,
        requires_key=metadata.requires_api_key,
        requires_base=metadata.requires_api_base and not metadata.default_api_base,
    )
    options = _layered(
        "fetch",
        provider,
        checked=True,
        defaults=PROVIDER_DEFAULT_OPTIONS.get(provider, _NO_OPTIONS),
        credential=(_credential_layer(credential), credential.options),
    )
    max_chars = _count(options, "max_chars")
    label, rate_keys = _credential_billing(credential, _FETCH_FALLBACK_RATE)
    return ResolvedFetch(
        name=None,
        provider=provider,
        runner="library",
        api_key=credential.api_key or "",
        api_base=_base(credential.api_base or metadata.default_api_base),
        options=MappingProxyType(options),
        max_chars=max_chars,
        timeout=IN_LOOP_TIMEOUT_S,
        label=label,
        rate_keys=rate_keys,
        on_credential=True,
    )


def _ignore(credential: WebToolCredential, kind: ToolKind) -> None:
    _warn_once(
        "Ignoring %s for %s, a provider this gateway cannot %s with",
        _credential_layer(credential),
        credential.provider,
        kind,
    )


def _check(credential: WebToolCredential, *, requires_key: bool, requires_base: bool) -> None:
    """Refuse a credential the library or the old code would complete from the server's own settings."""
    if requires_key and not credential.api_key:
        raise WebSearchCredentialRefusedError(credential.provider, "carries no API key, which the provider requires")
    if requires_base and not _base(credential.api_base):
        raise WebSearchCredentialRefusedError(credential.provider, "carries no base URL, which the provider requires")


def _credential_layer(credential: WebToolCredential) -> str:
    if credential.source == WebToolCredentialSource.HYBRID:
        return "the control plane's credential"
    return "the organization's key"


def _credential_billing(credential: WebToolCredential, fallback: str) -> tuple[str | None, tuple[str, ...]]:
    """The label and rate keys of a call a credential serves (design L).

    A hybrid data plane settles nothing itself, so the control plane's
    credential carries neither (design K).
    """
    if credential.source == WebToolCredentialSource.HYBRID:
        return None, ()
    return f"{_ORG_LABEL_PREFIX}:{credential.provider}", (fallback,)


def _layered(
    kind: ToolKind,
    provider: str,
    *,
    checked: bool,
    defaults: Mapping[str, Any],
    instance: tuple[str, Mapping[str, Any], frozenset[str]] | None = None,
    credential: tuple[str, Mapping[str, Any]] | None = None,
    workspace: Mapping[str, Any] = _NO_OPTIONS,
    request: Mapping[str, Any] = _NO_OPTIONS,
) -> dict[str, Any]:
    """The options a call runs with, layered lowest first.

    ``instance`` is the instance's layer with the keys its load recorded as
    breaking the rules; ``credential`` is a credential's. Unless ``checked``,
    the layers pass as they are, without otari's defaults. Each dropped option
    is warned about once, not on every call.
    """
    instance_name, instance_options, broken = instance or ("", _NO_OPTIONS, frozenset())
    credential_name, credential_options = credential or ("", _NO_OPTIONS)
    if not checked:
        return {**instance_options, **credential_options, **workspace, **request}
    operator_only = _operator_only(kind, provider)
    if refused := _tenant_problems(kind, provider, request, operator_only):
        key, problem = next(iter(refused.items()))
        raise WebSearchOptionRefusedError(key, problem)
    return {
        **defaults,
        **_kept(
            instance_options,
            instance_name,
            provider,
            {key: f"option '{key}' broke the rules at load" for key in broken},
        ),
        **_kept(credential_options, credential_name, provider, option_problems(kind, provider, credential_options)),
        **_kept(
            workspace,
            "the workspace's provider_options",
            provider,
            _tenant_problems(kind, provider, workspace, operator_only),
        ),
        **request,
    }


def _operator_only(kind: ToolKind, provider: str) -> frozenset[str]:
    metadata = (
        any_search.AnySearch.get_provider_metadata(provider)
        if kind == "search"
        else any_fetch.AnyFetch.get_provider_metadata(provider)
    )
    return frozenset(option.name for option in metadata.options if option.operator_only)


def _tenant_problems(
    kind: ToolKind, provider: str, options: Mapping[str, Any], operator_only: frozenset[str]
) -> dict[str, str]:
    """Why each of a workspace's or a request's options cannot reach the provider.

    The provider does not take it or its value, or only the deployment sets it.
    """
    problems = option_problems(kind, provider, options)
    for key in options:
        if key in operator_only and key not in problems:
            problems[key] = f"option '{key}' is set only by the deployment"
    return problems


def _kept(options: Mapping[str, Any], layer: str, provider: str, problems: Mapping[str, str]) -> dict[str, Any]:
    """The layer's options without those that have a problem, each dropped with a warning that never shows its value."""
    kept: dict[str, Any] = {}
    for key, value in options.items():
        if key in problems:
            _warn_once("Dropping an option of %s for %s: %s.", layer, provider, problems[key])
        else:
            kept[key] = value
    return kept


@functools.lru_cache(maxsize=1024)
def _warn_once(template: str, *args: str) -> None:
    """Log a warning the first time it is due, rather than on every call that meets the same problem."""
    logger.warning(template, *args)


def _base(api_base: str | None) -> str:
    """A base URL as the library takes it: without surrounding blanks or a trailing slash, ``""`` for none."""
    return (api_base or "").strip().rstrip("/")


def _count(options: dict[str, Any], key: str) -> int | None:
    """Take the count out of the options: a positive whole number, else nothing."""
    value = options.pop(key, None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
