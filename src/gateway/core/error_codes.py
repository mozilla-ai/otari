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

BUDGET_EXCEEDED = "budget_exceeded"
USER_BLOCKED = "user_blocked"
USER_NOT_FOUND = "user_not_found"
RATE_LIMITED = "rate_limited"
UPSTREAM_RATE_LIMITED = "upstream_rate_limited"
INVALID_MODEL = "invalid_model"
MODEL_NOT_ALLOWED = "model_not_allowed"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
PRICING_REQUIRED = "pricing_required"


def error_headers(code: str, *, budget_scope: str | None = None, rule: str | None = None) -> dict[str, str]:
    """``Otari-Error-Code`` plus any of the extra headers that have a value."""
    headers = {ERROR_CODE_HEADER: code}
    if budget_scope is not None:
        headers[BUDGET_SCOPE_HEADER] = budget_scope
    if rule is not None:
        headers[RATE_LIMIT_RULE_HEADER] = rule
    return headers


def error_code_of(headers: Mapping[str, str] | None) -> str | None:
    """The code a refusal's headers carry, or None."""
    return (headers or {}).get(ERROR_CODE_HEADER)
