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

import idna

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

def _ascii(domain: str) -> str:
    """Punycode an internationalized domain, or hand back what came in.

    Both normalizers below run this, which is the whole point: a claim and a
    sign-in have to land on the same spelling. Without it the two disagree
    exactly where it is hardest to see. ``münchen.de`` fails the ASCII-only
    shape check, so an admin retypes it as ``xn--mnchen-3ya.de``, which is
    claimable and provable by DNS; every colleague then signs in at
    ``münchen.de``, which normalizes to itself and matches nothing. The result
    is a claim shown as verified and active that admits nobody, with no error
    raised anywhere to explain it.

    **IDNA2008 (``idna``), not the standard library's ``str.encode("idna")``.**
    That codec implements IDNA2003, whose mapping step folds ``ß`` to ``ss`` and
    final sigma to sigma, so ``faß.de`` and ``fass.de`` both arrive as
    ``fass.de``. Those are two registrable domains owned by two different
    people, and collapsing them means proving one admits addresses at the other:
    the DNS proof would no longer bind to the domain it was taken from, which is
    the single guarantee this feature rests on. ``uts46=True`` keeps the case
    folding and normalization a typed claim needs; ``transitional=False`` is
    what holds ``ß`` apart from ``ss``.

    A value the library refuses (an empty label, a leading hyphen, an over-long
    label) is returned unchanged rather than raised on. For a claim,
    ``is_registrable_domain`` rejects it a moment later with a message about the
    domain; for a sign-in it simply matches nothing, which is what an
    unclaimable domain should do.
    """
    try:
        return idna.encode(domain, uts46=True, transitional=False).decode("ascii")
    except (idna.IDNAError, UnicodeError):
        return domain


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
    without it and the two must match, and an internationalized domain is
    punycoded for the same reason (see ``_ascii``).
    """
    if "@" not in email:
        return None
    domain = email.rsplit("@", 1)[1].strip().lower().rstrip(".")
    return _ascii(domain) or None


def normalized_domain(value: str) -> str:
    """Normalize an admin's typed claim to the form a stored domain takes.

    Accepts a bare domain or a whole address, so pasting one's own address into
    the field claims the right thing rather than storing something that matches
    nobody. Internationalized domains are punycoded, so ``münchen.de`` and
    ``xn--mnchen-3ya.de`` are one claim rather than two.
    """
    candidate = value.strip().lower().rstrip(".")
    if "@" in candidate:
        candidate = candidate.rsplit("@", 1)[1]
    return _ascii(candidate)


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
