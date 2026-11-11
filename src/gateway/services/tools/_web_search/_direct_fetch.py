"""Direct fetch: pick the instance, resolve the call, name an unpriced one's row, and run it.

:meth:`DirectToolService.fetch` calls the four functions below in that order,
and gates, bills and logs the call.

- :func:`pick_direct_fetch_instance` returns the instance the request names,
  else the one ``web_fetch_default_tool`` names,
  else ``builtin_fetch``. A deployment with ``web_fetch_enabled`` off serves
  none, as it offers no ``otari_web_fetch``.
- :func:`resolve_direct_fetch` returns the :class:`DirectFetch`. A call that
  names its instance runs on the instance's own key. A call that names none
  runs on the organization fetch key in the workspace's policy, when there is
  one, with that key's provider and options, under the rate key
  ``otari:web_fetch``. The URL is checked against the workspace's domain lists
  and the built-in fetcher's address rules here, before anything is held for
  the call; the fetch holds where the page was found to the lists too.
- :func:`unpriced_direct_fetch_detail` names the row ``require_pricing`` needs.
- :func:`run_direct_fetch` fetches the page through any-fetch: the built-in
  fetcher on a retrieval client of its own, a provider on the app's web
  provider client.

Whichever key serves the call, the instance supplies the allowlist entry, the
rate-limit rules, the timeout, the usage row's names and the response's
``fetch_tool``.
"""

from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from decimal import Decimal

import httpx

import any_fetch
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    BUILTIN_FETCH_PROVIDER,
    ToolInstance,
    ToolSettings,
    configured_fetch_default,
)
from gateway.exceptions.tools_exceptions import (
    DirectFetchDomainRefusedError,
    DirectFetchFailedError,
    DirectFetchNotEnabledError,
    DirectFetchUrlRefusedError,
    SearchToolMisconfiguredError,
    SearchToolSelectionError,
)
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.schemas.tools import FetchRequest, FetchResponse
from gateway.services.tools._web_access import workspace_domain_policy
from gateway.services.tools._web_fetch import FetchFailure, fetch_page, refused_url, retrieval_client
from gateway.services.tools._web_fetch._builtin import DOMAIN_POLICY_TAG, URL_INVALID_TAG
from gateway.services.tools._web_search._direct import direct_unpriced_detail
from gateway.services.tools._web_search._resolve import ResolvedFetch, resolve_fetch, served_fetch_instances
from gateway.services.tools._web_tool_failure import failed_text
from gateway.services.web_retrieval_policy import DomainPolicy


@dataclass(frozen=True)
class DirectFetch:
    """One direct fetch, resolved: what the route checks, bills and runs.

    ``url`` is the URL asked for in its canonical form, which the gateway's own
    checks passed. ``policy`` is the workspace's domain lists, which the fetch
    holds the page's final URL to as well. ``resolved`` says what serves the
    call: the instance, or the organization key in its place.
    """

    instance: ToolInstance
    resolved: ResolvedFetch
    url: str
    max_chars: int | None
    policy: DomainPolicy
    include_raw: bool


@dataclass(frozen=True)
class DirectFetchResult:
    """What a direct fetch answered, and what the provider said it cost."""

    response: FetchResponse
    cost: Decimal | None


def pick_direct_fetch_instance(config: ToolSettings, name: str | None) -> ToolInstance:
    """The instance a direct fetch runs on: the one named, else the fetch default, else ``builtin_fetch``.

    Raises :class:`DirectFetchNotEnabledError` where ``web_fetch_enabled`` is
    off, :class:`SearchToolSelectionError` for a name no fetch instance has,
    and :class:`SearchToolMisconfiguredError` for an instance without the key
    its provider needs, which only a stored one whose key cannot be read is.
    """
    if not config.web_fetch_enabled:
        raise DirectFetchNotEnabledError()
    instances = served_fetch_instances(config)
    if name is None:
        default = configured_fetch_default(config)
        name = default if default in instances else BUILTIN_FETCH
    instance = instances.get(name)
    if instance is None:
        msg = f"Unknown fetch tool '{name}'. Available: {', '.join(sorted(instances))}."
        raise SearchToolSelectionError(msg)
    if instance.provider != BUILTIN_FETCH_PROVIDER and not instance.api_key and _requires_key(instance.provider):
        raise SearchToolMisconfiguredError(instance.name, f"Fetch tool '{instance.name}' is not configured correctly.")
    return instance


def _requires_key(provider: str) -> bool:
    return any_fetch.AnyFetch.get_provider_metadata(provider).requires_api_key


def resolve_direct_fetch(
    config: ToolSettings,
    instance: ToolInstance,
    policy: ResolvedWebSearchConfig | None,
    request: FetchRequest,
    *,
    named: bool,
) -> DirectFetch:
    """What a direct fetch of ``request.url`` on ``instance`` runs on, with which options and domain lists.

    ``named`` says the caller named the instance, in the path or the body; such
    a call runs on the instance's own key, never the organization's. Raises
    :class:`WebSearchOptionRefusedError` for a request option the provider
    serving the call does not take, :class:`WebSearchCredentialRefusedError`
    for an organization key its provider cannot be called with,
    :class:`DirectFetchUrlRefusedError` for a URL that is not a public http or
    https one or names a private address, and
    :class:`DirectFetchDomainRefusedError` for one the workspace's lists disallow.
    """
    organization_key = policy.fetch_credential if policy is not None and not named else None
    resolved = resolve_fetch(
        config,
        None,
        requested=instance.name,
        for_search=None,
        request_options=request.provider_options,
        direct_credential=organization_key,
    )
    domains = workspace_domain_policy(policy)
    tag, checked = refused_url(request.url, domains)
    if tag is not None or checked is None:
        raise _refusal(tag or URL_INVALID_TAG)
    return DirectFetch(
        instance=instance,
        resolved=resolved,
        url=str(checked.url),
        max_chars=request.max_chars if request.max_chars is not None else resolved.max_chars,
        policy=domains,
        include_raw=request.include_raw,
    )


def unpriced_direct_fetch_detail(fetch: DirectFetch) -> str:
    """The 402's message for a fetch ``require_pricing`` cannot bill: the row to add, and where."""
    return direct_unpriced_detail(
        "fetch", fetch.instance.name, fetch.resolved.rate_keys[0], organization_key=fetch.resolved.on_credential
    )


async def run_direct_fetch(
    fetch: DirectFetch, client: httpx.AsyncClient, *, trust_env_proxy: bool
) -> DirectFetchResult:
    """Fetch the page: the built-in fetcher on its own client, a provider instance through ``client``.

    Never reaches a private host, as the fetch tool never does. Raises
    :class:`DirectFetchDomainRefusedError` for a page found outside the
    workspace's lists, :class:`DirectFetchUrlRefusedError` for a provider's
    final URL the gateway refuses, and :class:`DirectFetchFailedError` when the
    fetch fails, the built-in fetcher's own failures included: a host it cannot
    resolve, or that resolves to a private address, is one of them, tagged
    ``url_invalid``. The message holds only the failure's safe fields, never the URL.
    """
    resolved = fetch.resolved
    on: AbstractAsyncContextManager[httpx.AsyncClient] = (
        retrieval_client(trust_env_proxy=trust_env_proxy)
        if resolved.provider == BUILTIN_FETCH_PROVIDER
        else nullcontext(client)
    )
    async with on as fetch_client:
        fetched = await fetch_page(
            resolved,
            fetch.url,
            client=fetch_client,
            policy=fetch.policy,
            allow_private=False,
            max_chars=fetch.max_chars,
        )
    if isinstance(fetched, FetchFailure):
        raise _failed(resolved, fetched)
    return DirectFetchResult(
        response=FetchResponse(
            fetch_tool=fetch.instance.name,
            url=fetch.url,
            final_url=fetched.final_url,
            title=fetched.title,
            # One line: a content type can carry its parameters over several.
            content_type=" ".join(fetched.content_type.split()),
            text=fetched.text,
            published=fetched.published,
            truncated=fetched.source_truncated or fetched.text_truncated,
            raw=fetched.raw if fetch.include_raw else None,
        ),
        cost=fetched.cost,
    )


def _failed(resolved: ResolvedFetch, fetched: FetchFailure) -> Exception:
    """The error a failed fetch raises: the gateway's own refusal of a URL, else the fetcher's failure.

    The gateway refuses a provider's final URL itself (no provider on the
    failure), and the built-in fetcher refuses a hop outside the domain lists
    on the gateway's behalf. Its ``url_invalid`` after the call is a
    resolution outcome, since the URL's form and a literal address were
    checked before the call, so it stays a failure of the fetch.
    """
    failure = fetched.failure
    if failure.provider is None or (resolved.provider == BUILTIN_FETCH_PROVIDER and failure.tag == DOMAIN_POLICY_TAG):
        return _refusal(failure.tag)
    return DirectFetchFailedError(
        failed_text(failure.provider, "fetch", failure.tag, fetched.status),
        headers=failure.as_headers(),
        upstream_status=fetched.status,
    )


def _refusal(tag: str) -> Exception:
    """The refusal a check's ``tag`` names: the domain lists' is a 403, any other check's a 400."""
    if tag == DOMAIN_POLICY_TAG:
        return DirectFetchDomainRefusedError()
    return DirectFetchUrlRefusedError()
