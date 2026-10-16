"""What a failed web tool call may show the caller (decision 67).

Only fields the logging rule lets an error carry (design section 7): the
provider, its HTTP status, a tag, and whether a retry may help; never a body,
a URL or exception text. The direct endpoint's 502 carries them as headers,
and a failed entry in Chat Completions' ``otari_tool_calls`` carries them as its
``error`` when the call's backend kept them. Today only a call refused past its
use cap does; the in-loop web backends do not keep a failure yet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import any_search
from gateway.core.error_codes import (
    ERROR_PROVIDER_HEADER,
    ERROR_RETRYABLE_HEADER,
    ERROR_STATUS_HEADER,
    ERROR_TAG_HEADER,
)

# The tag of a call refused because the request used up its calls, which reaches no provider.
CALL_LIMIT_TAG = "call_limit"

# A retry may help a timeout, a rate limit or a fault on the provider's side.
_RETRYABLE_TAGS = frozenset({"timeout"})
_CLIENT_ERROR = 400
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR = 500


@dataclass(frozen=True)
class WebToolFailure:
    """A failed web tool call's safe fields. ``provider`` is ``None`` for a call refused before it reached one."""

    provider: str | None
    status: int | None
    tag: str
    retryable: bool

    @classmethod
    def of(cls, provider: str | None, status: int | None, tag: str) -> "WebToolFailure":
        """The failure of ``provider`` with this status and tag, its retryability read from them.

        Only an error status is kept: an answer that came back 200 and could not
        be read failed with no status of its own.
        """
        status = status if status is not None and status >= _CLIENT_ERROR else None
        retryable = tag in _RETRYABLE_TAGS or status == _TOO_MANY_REQUESTS or (status or 0) >= _SERVER_ERROR
        return cls(provider=provider, status=status, tag=tag, retryable=retryable)

    @classmethod
    def from_provider_error(cls, error: any_search.ProviderError) -> "WebToolFailure":
        """The failure any-search raised: its provider, status and tag."""
        return cls.of(error.provider, error.status, error.tag)

    def as_dict(self) -> dict[str, Any]:
        """The four fields as Chat Completions' ``otari_tool_calls`` entries carry them, ``None`` kept."""
        return asdict(self)

    def as_headers(self) -> dict[str, str]:
        """The four fields as the direct endpoint's 502 headers; ``retryable`` is ``true`` or ``false``."""
        headers = {ERROR_TAG_HEADER: self.tag, ERROR_RETRYABLE_HEADER: "true" if self.retryable else "false"}
        if self.provider is not None:
            headers[ERROR_PROVIDER_HEADER] = self.provider
        if self.status is not None:
            headers[ERROR_STATUS_HEADER] = str(self.status)
        return headers
