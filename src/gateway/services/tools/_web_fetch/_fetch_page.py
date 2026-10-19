"""One page fetched on a fetch instance, for the fetch tool and for enriching a search's results.

The built-in fetcher checks the URL and every redirect against the domain policy
itself. A provider follows redirects and resolves hostnames on its own side, so
the gateway checks only the URL it asks for and the URL the provider reports
finding the page at: the host must pass the domain policy, and must not name an
address the built-in fetcher refuses, as an IP address in any form or as
``localhost``. A provider that does not report its final URL, as Exa and
Tabstack do not, answers with the URL as asked, so a redirect it follows escapes
the policy.

A failure is returned, never raised, with its safe fields: the provider that
failed, ``None`` for a URL the gateway's own checks refused, the HTTP status, a
tag and whether a retry may help. An exception a provider raises other than its
own failure is logged by its class alone, since its text can carry the URL.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

from any_fetch import AnyFetch, FetchedPage, MissingCredentialError, ProviderError, UnsupportedParameterError
from gateway.core.settings.tools import BUILTIN_FETCH_PROVIDER
from gateway.log_config import logger
from gateway.services.tools._web_display import fetched_url
from gateway.services.tools._web_fetch._builtin import DOMAIN_POLICY_TAG, URL_INVALID_TAG
from gateway.services.tools._web_tool_failure import (
    INTERNAL_ERROR_TAG,
    MISSING_CREDENTIAL_TAG,
    UNSUPPORTED_OPTION_TAG,
    WebToolFailure,
)
from gateway.services.web_retrieval_network import blocked_address_reason
from gateway.services.web_retrieval_policy import (
    CanonicalHost,
    CanonicalWebURL,
    DomainPolicy,
    IPAddress,
    WebURLValidationError,
    canonicalize_web_url,
)

if TYPE_CHECKING:
    import httpx

    from gateway.services.tools._web_search import ResolvedFetch


@dataclass(frozen=True)
class FetchFailure:
    """Why a fetch brought no page: its safe fields, and the HTTP status as the provider gave it."""

    failure: WebToolFailure
    status: int | None = None


async def fetch_page(
    resolved: ResolvedFetch,
    url: str,
    *,
    client: httpx.AsyncClient | None,
    policy: DomainPolicy,
    allow_private: bool,
    max_chars: int | None,
) -> FetchedPage | FetchFailure:
    """The page at ``url`` on ``resolved``, or why there is none.

    ``client`` is the built-in fetcher's retrieval client for ``builtin``, else the
    app's web provider client. ``allow_private`` lets only the built-in fetcher
    reach a private host.
    """
    builtin = resolved.provider == BUILTIN_FETCH_PROVIDER
    if not builtin:
        # A provider is asked for the canonical form of the URL checked here.
        try:
            checked = canonicalize_web_url(url)
        except WebURLValidationError:
            return _refused(URL_INVALID_TAG)
        if (tag := _refusal(checked, policy)) is not None:
            return _refused(tag)
        url = str(checked.url)
    try:
        fetcher = (
            AnyFetch.create(resolved.provider, client=client, policy=policy, allow_private=allow_private)
            if builtin
            else AnyFetch.create(
                resolved.provider,
                api_key=resolved.api_key,
                api_base=resolved.api_base,
                timeout=resolved.timeout,
                client=client,
            )
        )
        async with fetcher:
            page = await fetcher.fetch(url, max_chars=max_chars, **resolved.options)
    except ProviderError as exc:
        return FetchFailure(WebToolFailure.from_provider_error(exc), exc.status)
    except MissingCredentialError as exc:
        return _raised(resolved, exc, MISSING_CREDENTIAL_TAG)
    except UnsupportedParameterError as exc:
        return _raised(resolved, exc, UNSUPPORTED_OPTION_TAG)
    except Exception as exc:
        return _raised(resolved, exc, INTERNAL_ERROR_TAG)
    if (error := page.error) is not None:
        return FetchFailure(WebToolFailure.of(resolved.provider, error.status, error.tag), error.status)
    if not builtin:
        # Where the provider found the page is checked as the URL asked for was, and a
        # URL it cannot be checked by is refused rather than trusted.
        found = fetched_url(page.final_url)
        if found is None:
            return _refused(URL_INVALID_TAG)
        if (tag := _refusal(found, policy)) is not None:
            return _refused(tag)
    return page


def _refusal(url: CanonicalWebURL, policy: DomainPolicy) -> str | None:
    """The tag that refuses ``url`` to a provider fetch instance, ``None`` when it may be fetched."""
    host = url.origin.host
    if not policy.permits(host):
        return DOMAIN_POLICY_TAG
    # The provider resolves a hostname itself, so only an address the URL names, and
    # localhost, can be checked here, by the built-in fetcher's rule for an address.
    address = _named_address(host)
    if _is_localhost(host) or (address is not None and blocked_address_reason(address) is not None):
        return URL_INVALID_TAG
    return None


def _refused(tag: str) -> FetchFailure:
    """A URL the gateway's own checks refused, before or after the provider was called."""
    return FetchFailure(WebToolFailure.of(None, None, tag))


def _raised(resolved: ResolvedFetch, exc: Exception, tag: str) -> FetchFailure:
    # The class alone: an exception's text can carry the URL.
    logger.warning("Fetch on instance '%s' (%s) failed: %s", resolved.name, resolved.provider, type(exc).__name__)
    return FetchFailure(WebToolFailure.of(resolved.provider, None, tag))


def _named_address(host: CanonicalHost) -> IPAddress | None:
    """The address a host names: an IP literal, or an IPv4 shorthand a resolver reads too, such as ``127.1``."""
    if host.ip is not None:
        return host.ip
    try:
        return ipaddress.ip_address(socket.inet_aton(host.value))
    except OSError:
        return None


def _is_localhost(host: CanonicalHost) -> bool:
    """``localhost`` or a name under it, which resolves to loopback (RFC 6761)."""
    return host.value == "localhost" or host.value.endswith(".localhost")
