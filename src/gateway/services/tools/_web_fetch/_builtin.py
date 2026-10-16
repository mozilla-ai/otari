"""otari's hardened fetcher as any-fetch's ``builtin`` provider (decision 54).

any-fetch declares the provider and leaves its implementation to the host.
``register_builtin_fetcher`` registers the factory below, which ``create_app``
calls at startup, never on import, so ``AnyFetch.create("builtin", ...)``
builds a :class:`HardenedFetcher` inside the library's own wrapper, which keeps
the URL out of httpx's log records like any other provider's.

The fetcher takes, per request, the client it sends through, the domain policy
it checks every hop against, and whether private hosts may be reached: never for
a fetch tool call, and for enriching a search result only when
``web_search_allow_private_hosts`` is on. Its failures are raised as
``ProviderError`` tags, which carry no URL; the fetch backend turns each back
into the tool's fixed text.
"""

import httpx

from any_fetch import AnyFetch, FetchedPage, ProviderError
from gateway.core.settings.tools import BUILTIN_FETCH_PROVIDER
from gateway.services.web_extraction import ExtractionError
from gateway.services.web_fetch_service import (
    UnsupportedContentTypeError,
    WebFetchError,
    WebFetchHTTPStatusError,
    WebFetchService,
)
from gateway.services.web_retrieval_network import (
    ContentDecodingError,
    NetworkDeadlineExceeded,
    PinnedAsyncHTTPTransport,
    RedirectValidationError,
    RetrievalAddressError,
    RetrievalDomainPolicyError,
    RetrievalTargetError,
    TrustedProxyAsyncHTTPTransport,
)
from gateway.services.web_retrieval_policy import DomainPolicy, WebURLValidationError

# The tags the fetcher's failures are raised with, which the fetch backend maps back to its texts.
DOMAIN_POLICY_TAG = "domain_policy"
REDIRECT_TAG = "redirect"
TIMEOUT_TAG = "timeout"
HTTP_STATUS_TAG = "http_status"
CONTENT_TYPE_TAG = "content_type"
EXTRACTION_TAG = "extraction"
URL_INVALID_TAG = "url_invalid"
NETWORK_TAG = "network"
CONTENT_DECODING_TAG = "content_decoding"


def register_builtin_fetcher() -> None:
    """Make otari's hardened fetcher any-fetch's ``builtin`` provider; a second call changes nothing."""
    AnyFetch.register_builtin(HardenedFetcher)


def retrieval_client(*, trust_env_proxy: bool) -> httpx.AsyncClient:
    """The client the built-in fetcher sends through, for the caller to open and close.

    Connections are pinned to the addresses the fetcher validated, unless the
    operator trusts the environment's proxy to enforce address safety
    (``web_retrieval_trust_env_proxy``). Redirects are the fetcher's to follow,
    one checked hop at a time.
    """
    transport = TrustedProxyAsyncHTTPTransport() if trust_env_proxy else PinnedAsyncHTTPTransport()
    return httpx.AsyncClient(transport=transport, timeout=None, follow_redirects=False, trust_env=False)


class HardenedFetcher:
    """One request's built-in fetcher, over a client the caller owns and closes."""

    def __init__(
        self, *, client: httpx.AsyncClient, policy: DomainPolicy | None = None, allow_private: bool = False
    ) -> None:
        self._service = WebFetchService(client)
        self._policy = policy or DomainPolicy()
        self._allow_private = allow_private

    async def fetch(self, url: str, *, max_chars: int | None) -> FetchedPage:
        """Fetch one page under the fetcher's own bounds, cut to ``max_chars`` when given."""
        try:
            result = await self._service.fetch(url, policy=self._policy, allow_private=self._allow_private)
        except RetrievalDomainPolicyError:
            raise _failed(DOMAIN_POLICY_TAG) from None
        except RedirectValidationError:
            raise _failed(REDIRECT_TAG) from None
        except (NetworkDeadlineExceeded, httpx.TimeoutException):
            raise _failed(TIMEOUT_TAG) from None
        except WebFetchHTTPStatusError as exc:
            raise _failed(HTTP_STATUS_TAG, exc.status_code) from None
        except UnsupportedContentTypeError:
            raise _failed(CONTENT_TYPE_TAG) from None
        except ExtractionError:
            raise _failed(EXTRACTION_TAG) from None
        except (WebURLValidationError, RetrievalTargetError, RetrievalAddressError):
            raise _failed(URL_INVALID_TAG) from None
        except (httpx.HTTPError, WebFetchError):
            raise _failed(NETWORK_TAG) from None
        except ContentDecodingError:
            raise _failed(CONTENT_DECODING_TAG) from None
        cut = max_chars is not None and len(result.text) > max_chars
        return FetchedPage(
            url=str(result.requested_url.url),
            final_url=str(result.final_url.url),
            text=result.text[:max_chars] if cut else result.text,
            content_type=result.content_type,
            cost_source="none",
            source_truncated=result.source_truncated,
            text_truncated=result.extraction_truncated or cut,
            raw={"content_kind": result.content_kind, "redirect_count": result.redirect_count},
        )

    async def aclose(self) -> None:
        """Nothing to close: the client is the caller's."""


def _failed(tag: str, status: int | None = None) -> ProviderError:
    return ProviderError(BUILTIN_FETCH_PROVIDER, status, tag)
