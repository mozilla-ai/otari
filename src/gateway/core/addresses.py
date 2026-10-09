"""What this codebase treats as an email address it could deliver to.

In ``core`` rather than beside the mail transport because both layers need it
and only one direction is allowed: ``services`` imports ``core.config``, so
config validating a from-address cannot reach into ``services.mail`` without a
cycle. Same reasoning as ``core/sql.py``, which exists so a route and a service
can share a condition without either importing the other.
"""

import re

_WHITESPACE = re.compile(r"\s")


# Deliberately permissive (there is no useful regex for RFC 5322, and the SMTP
# server is the real authority): it rejects the shapes that are certainly not
# addresses, which is what an operator typing into a form needs, and nothing
# more. Shared with tenancy's member and invitation addresses rather than kept
# per caller, so "an address Otari will accept" has one answer.
# Not a regex: this runs on unauthenticated input, and a pattern for "a dot
# inside the domain" backtracks quadratically on a long run of dots.
def normalized_address(value: str) -> str | None:
    """Lower-case and trim an address, or return ``None`` if it cannot be one."""
    candidate = value.strip().lower()
    local, at, domain = candidate.partition("@")
    if not local or not at or "@" in domain or "." not in domain[1:-1]:
        return None
    return None if _WHITESPACE.search(candidate) else candidate


__all__ = ["normalized_address"]
