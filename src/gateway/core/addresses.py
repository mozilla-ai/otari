"""What this codebase treats as an email address it could deliver to.

In ``core`` rather than beside the mail transport because both layers need it
and only one direction is allowed: ``services`` imports ``core.config``, so
config validating a from-address cannot reach into ``services.mail`` without a
cycle. Same reasoning as ``core/sql.py``, which exists so a route and a service
can share a condition without either importing the other.
"""

# The two characters that start an encoded word, which a header reader decodes into other text.
_ENCODED_WORD = "=?"
_NOT_IN_A_PLAIN_ADDRESS = frozenset(' "(),:;<>[\\]')


# NOTE: Every check must stay linear in the length of the text, because the text can be any length.
# A regular expression that backtracks is not linear, so the shape of the domain is checked without one.
def normalized_address(value: str) -> str | None:
    """Return ``value`` trimmed and in lower case if it is a plain email address, or ``None`` if it is not.

    A plain address is one address and nothing else.
    It has no display name, comment, quoted text, list, domain literal, encoded word, space or invisible character.
    Its local part is ASCII, because the SMTP transport does not ask for SMTPUTF8 (RFC 6531).
    Its domain has two or more labels, and no label is empty.
    """
    candidate = value.strip()
    # Checked before lower-casing, which turns a character such as the Kelvin sign (U+212A) into ASCII "k".
    if not candidate.partition("@")[0].isascii():
        return None
    candidate = candidate.lower()
    if not candidate.isprintable() or not _NOT_IN_A_PLAIN_ADDRESS.isdisjoint(candidate) or _ENCODED_WORD in candidate:
        return None
    local, at, domain = candidate.partition("@")
    if not local or not at or "@" in domain or "." not in domain[1:-1]:
        return None
    # An empty label ("example.com.", "a..b") is refused: a mail server will not
    # take one, and the send would fail where nobody sees it.
    if "" in domain.split("."):
        return None
    return candidate


# This is how long a normalized address may be and still fit in the column it is stored in.
# NOTE: "İ" (U+0130) lower-cases to two codepoints, so a 255-character address ending in one
# normalizes to 256, over width. Checked after normalization, because that is the value actually stored.
MAX_STORED_ADDRESS_LENGTH = 255


def stored_address(value: str) -> str | None:
    """Return ``value`` normalized as it will be stored, or ``None`` when it cannot be stored.

    Refuses on shape, through ``normalized_address``, and on stored length.
    """
    candidate = normalized_address(value)
    if candidate is None or len(candidate) > MAX_STORED_ADDRESS_LENGTH:
        return None
    return candidate


def same_address(first: str, second: str) -> bool:
    """Return whether these two values are the same address.

    A difference of case or surrounding space is the same address. A value that is not an address
    matches nothing, including another value that is not an address.
    """
    normalized = normalized_address(first)
    return normalized is not None and normalized_address(second) == normalized


__all__ = ["MAX_STORED_ADDRESS_LENGTH", "normalized_address", "same_address", "stored_address"]
