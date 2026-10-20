"""Stable, machine-readable codes for the refusals a caller is expected to act on.

Sent as the ``Otari-Error-Code`` response header and as ``code`` in the error
body, beside the human-readable ``detail``, so a client maps a refusal by its
code rather than by matching text that may be reworded. A streamed error event
carries it as ``error.code``. A code, once sent, keeps its meaning.
"""

from collections.abc import Mapping

ERROR_CODE_HEADER = "Otari-Error-Code"
# Which budget refused: ``user`` for the billed user's own budget, otherwise the
# scope of the ceiling (one of ``models.budgets.ScopeType``).
BUDGET_SCOPE_HEADER = "Otari-Budget-Scope"
# The ``rate_limits`` rule a 429 names.
RATE_LIMIT_RULE_HEADER = "Otari-Rate-Limit-Rule"
# What a direct search's 502 shows of a provider's failure (decision 67): the
# provider, its own HTTP status, a tag naming the failure, and whether a retry
# may help (``true`` or ``false``). A field with no value sends no header.
ERROR_PROVIDER_HEADER = "Otari-Error-Provider"
ERROR_STATUS_HEADER = "Otari-Error-Status"
ERROR_TAG_HEADER = "Otari-Error-Tag"
ERROR_RETRYABLE_HEADER = "Otari-Error-Retryable"

BUDGET_EXCEEDED = "budget_exceeded"
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
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
# The model the request reached does not accept a kind of content the request carries, such as an image.
UNSUPPORTED_CONTENT = "unsupported_content"
PRICING_REQUIRED = "pricing_required"
# The organization has no funds paid in to cover a request the deployment pays for.
INSUFFICIENT_FUNDS = "insufficient_funds"
END_USER_BUDGET_NOT_ALLOWED = "end_user_budget_not_allowed"
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
