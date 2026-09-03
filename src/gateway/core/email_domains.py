"""Email domains: the apex an address belongs to, and whether one may be claimed.

Beside `core.addresses` and for the same reason: an organization claiming a
domain (`services.tenancy.organization_domain_service`) and a sign-in matching
one (`services.tenancy.organization_service`) have to agree on what the domain
of an address *is*, and a disagreement between them is not a cosmetic one. The
claim is stored once and matched on every sign-in afterwards, so a normalizer
that differs by a trailing dot or a case fold would store a claim that never
matches, or match one it should not.
"""

import re

# Free and consumer providers an organization must never be able to claim: a
# match there would sweep in unrelated strangers rather than colleagues, and
# the DNS proof is no defence because nobody claiming these controls them.
PUBLIC_EMAIL_DOMAINS = frozenset(
    {
        "126.com",
        "163.com",
        "aol.com",
        "fastmail.com",
        "gmail.com",
        "gmx.com",
        "gmx.net",
        "googlemail.com",
        "hey.com",
        "hotmail.com",
        "icloud.com",
        "live.com",
        "mac.com",
        "mail.com",
        "me.com",
        "msn.com",
        "outlook.com",
        "pm.me",
        "proton.me",
        "protonmail.com",
        "qq.com",
        "yahoo.co.uk",
        "yahoo.com",
        "yandex.com",
        "ymail.com",
        "zoho.com",
    }
)

# A registrable domain: dot-separated labels with an alphabetic TLD. Stricter
# than `core.addresses`' address pattern on purpose. That one bounds what a
# person may sign in as and leans permissive; this one bounds what an admin
# stores as a claim, and the stored value has to match the normalizer's output
# exactly on every later sign-in, so a scheme, a path, or a stray label would
# store a claim that can never match.
_DOMAIN_PATTERN = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$")


def email_domain(email: str) -> str | None:
    """Return the apex domain of an address, or ``None`` when it has none.

    Splits on the last ``@``, so plus-addressing and quoted local parts are
    irrelevant. The trailing dot of an RFC-legal absolute form
    (``person@example.com.``) is stripped, because an admin claims the domain
    without it and the two must match.
    """
    if "@" not in email:
        return None
    domain = email.rsplit("@", 1)[1].strip().lower().rstrip(".")
    return domain or None


def normalized_domain(value: str) -> str:
    """Normalize an admin's typed claim to the form a stored domain takes.

    Accepts a bare domain or a whole address, so pasting one's own address into
    the field claims the right thing rather than storing something that matches
    nobody.
    """
    candidate = value.strip().lower().rstrip(".")
    if "@" in candidate:
        candidate = candidate.rsplit("@", 1)[1]
    return candidate


def is_registrable_domain(domain: str) -> bool:
    """Whether ``domain`` has the shape of a domain that could be registered."""
    return bool(_DOMAIN_PATTERN.match(domain))


def is_public_email_domain(domain: str) -> bool:
    """Whether ``domain`` is a free provider, including under a subdomain.

    Every trailing label window is tested, so ``mail.gmail.com`` is refused
    along with ``gmail.com``: without that, prefixing a label would be enough to
    claim a provider's whole population.
    """
    labels = domain.lower().strip(".").split(".")
    return any(".".join(labels[start:]) in PUBLIC_EMAIL_DOMAINS for start in range(len(labels) - 1))


__all__ = [
    "PUBLIC_EMAIL_DOMAINS",
    "email_domain",
    "is_public_email_domain",
    "is_registrable_domain",
    "normalized_domain",
]
