"""The machine-readable parts of a refusal: its code, and the category a dialect names.

A code is sent as the ``Otari-Error-Code`` response header and as ``code`` in
the error body, beside the human-readable ``detail``, so a client maps a refusal
by its code rather than by matching text that may be reworded. A streamed error
event carries it as ``error.code``. A code, once sent, keeps its meaning.
"""

from collections.abc import Mapping
from enum import Enum, auto

from fastapi import HTTPException, status

ERROR_CODE_HEADER = "Otari-Error-Code"
# Which budget refused: ``user`` for the billed user's own budget, otherwise the
# scope of the ceiling (one of ``models.budgets.ScopeType``).
BUDGET_SCOPE_HEADER = "Otari-Budget-Scope"
# The ``rate_limits`` rule a 429 names.
RATE_LIMIT_RULE_HEADER = "Otari-Rate-Limit-Rule"
# The ``mode: report`` rules a served request was over, comma-separated: not a refusal.
RATE_LIMIT_EXCEEDED_HEADER = "Otari-Rate-Limit-Exceeded"
# What a direct search's 502 shows of a provider's failure (decision 67): the
# provider, its own HTTP status, a tag naming the failure, and whether a retry
# may help (``true`` or ``false``). A field with no value sends no header.
ERROR_PROVIDER_HEADER = "Otari-Error-Provider"
ERROR_STATUS_HEADER = "Otari-Error-Status"
ERROR_TAG_HEADER = "Otari-Error-Tag"
ERROR_RETRYABLE_HEADER = "Otari-Error-Retryable"

BUDGET_EXCEEDED = "budget_exceeded"
# A guardrail refused the request's input or the model's answer.
GUARDRAIL_VIOLATION = "guardrail_violation"
USER_BLOCKED = "user_blocked"
USER_NOT_FOUND = "user_not_found"
RATE_LIMITED = "rate_limited"
UPSTREAM_RATE_LIMITED = "upstream_rate_limited"
INVALID_MODEL = "invalid_model"
MODEL_NOT_ALLOWED = "model_not_allowed"
# The organization offers the model on a provider key, with its serving switch off.
MODEL_NOT_SERVING = "model_not_serving"
# A provider key of the organization serves the provider, and none offers the model.
MODEL_NOT_FOUND = "model_not_found"
# ``provider_supported_countries`` closes the model's provider to the country the request comes from.
COUNTRY_NOT_SUPPORTED = "country_not_supported"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
# The provider refused the request for its size (an upstream 413), such as a prompt over a tokens-per-minute limit.
REQUEST_TOO_LARGE = "request_too_large"
# The model the request reached does not accept a kind of content the request carries, such as an image.
UNSUPPORTED_CONTENT = "unsupported_content"
# The provider cannot express a request parameter for the model, such as ``response_format`` on one with no JSON mode.
UNSUPPORTED_PARAMETER = "unsupported_parameter"
# The model failed to produce output matching the requested JSON schema; resending the request may succeed.
STRUCTURED_OUTPUT_FAILED = "structured_output_failed"
# The model is a kind the endpoint cannot serve, such as an embedding model sent a chat completion.
WRONG_MODEL_TYPE = "wrong_model_type"
PRICING_REQUIRED = "pricing_required"
# The organization has no funds paid in to cover a request the deployment pays for.
INSUFFICIENT_FUNDS = "insufficient_funds"
END_USER_BUDGET_NOT_ALLOWED = "end_user_budget_not_allowed"
# Otari-End-User-Budget on a request that names no end user of a service key, so no budget would apply.
END_USER_BUDGET_WITHOUT_END_USER = "end_user_budget_without_end_user"
# A service key already has as many end users as its ``max_end_users`` allows.
END_USER_LIMIT_REACHED = "end_user_limit_reached"
ALL_CANDIDATES_REJECTED = "all_candidates_rejected"
PROVIDER_NOT_CONFIGURED = "provider_not_configured"
# The provider refused the credential the gateway holds for it (an upstream 401 or 403).
PROVIDER_CREDENTIAL_REJECTED = "provider_credential_rejected"
# A request body or parameter that failed schema validation (a 422).
INVALID_REQUEST = "invalid_request"
# The provider failed after the stream started, for a reason no more specific
# code names (a stream's error event), or answered a chat completion with no
# choices (a 502).
PROVIDER_ERROR = "provider_error"
# Sent only in a stream's error event: the provider's stream ended before it
# finished, by closing without its final event or cutting the connection.
PROVIDER_STREAM_INTERRUPTED = "provider_stream_interrupted"
# Creating or first claiming an account needs the terms of service the deployment publishes.
TERMS_NOT_ACCEPTED = "terms_not_accepted"
# A workspace with web access (fetch or search) turned off refused a request that declared a web tool.
WEB_ACCESS_DISABLED = "web_access_disabled"


class ErrorKind(Enum):
    """Coarse error category, for a dialect that names one on an HTTP error.

    A dialect that distinguishes more categories once its stream has committed
    maps those itself.
    """

    API = auto()
    AUTHENTICATION = auto()
    INVALID_REQUEST = auto()
    NOT_FOUND = auto()
    PERMISSION = auto()
    RATE_LIMIT = auto()
    REQUEST_TOO_LARGE = auto()

    @classmethod
    def from_status(cls, status_code: int) -> "ErrorKind":
        """The kind a bare status implies, falling back to ``API``.

        An error flattened into an ``HTTPException`` no longer carries its kind,
        so a status stands in for one.
        """
        return _STATUS_ERROR_KINDS.get(status_code, cls.API)


_STATUS_ERROR_KINDS = {
    status.HTTP_400_BAD_REQUEST: ErrorKind.INVALID_REQUEST,
    status.HTTP_401_UNAUTHORIZED: ErrorKind.AUTHENTICATION,
    status.HTTP_403_FORBIDDEN: ErrorKind.PERMISSION,
    status.HTTP_404_NOT_FOUND: ErrorKind.NOT_FOUND,
    status.HTTP_413_CONTENT_TOO_LARGE: ErrorKind.REQUEST_TOO_LARGE,
    status.HTTP_429_TOO_MANY_REQUESTS: ErrorKind.RATE_LIMIT,
}


def error_headers(
    code: str,
    *,
    budget_scope: str | None = None,
    rule: str | None = None,
    retry_after: int | None = None,
) -> dict[str, str]:
    """``Otari-Error-Code`` plus any of the extra headers that have a value."""
    headers = {ERROR_CODE_HEADER: code}
    if budget_scope is not None:
        headers[BUDGET_SCOPE_HEADER] = budget_scope
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    if rule is not None:
        headers[RATE_LIMIT_RULE_HEADER] = rule
    return headers


def error_code_of(headers: Mapping[str, str] | None) -> str | None:
    """The code a refusal's headers carry, or None."""
    return (headers or {}).get(ERROR_CODE_HEADER)


class RefusalException(HTTPException):
    """A refusal whose error body carries ``fields`` beside ``detail`` and ``code``.

    A field goes in the body only, never in a header: its value may be caller
    text, such as an end user's id, that a header cannot encode.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        headers: Mapping[str, str] | None = None,
        fields: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=dict(headers) if headers else None)
        self.fields = dict(fields or {})


def body_fields_of(exc: BaseException) -> dict[str, str]:
    """The fields ``exc`` adds to its error body, empty for anything but a :class:`RefusalException`."""
    return exc.fields if isinstance(exc, RefusalException) else {}
