"""Stable, machine-readable codes for the refusals a caller is expected to act on.

Sent as the ``Otari-Error-Code`` response header beside the human-readable
``detail``, so a client maps a refusal by its code rather than by matching text
that may be reworded. A code, once sent, keeps its meaning.
"""

ERROR_CODE_HEADER = "Otari-Error-Code"
# Which budget refused: ``user`` for the billed user's own budget, otherwise the
# scope of the ceiling (``api_token``, ``workspace``, ``organization``, ...).
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


def error_headers(code: str, **extra: str | None) -> dict[str, str]:
    """``Otari-Error-Code`` plus any of the extra headers that have a value."""
    headers = {ERROR_CODE_HEADER: code}
    names = {"budget_scope": BUDGET_SCOPE_HEADER, "rule": RATE_LIMIT_RULE_HEADER}
    headers.update({names[name]: value for name, value in extra.items() if value is not None})
    return headers
