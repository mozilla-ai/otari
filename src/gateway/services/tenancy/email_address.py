"""The one shape check an email address gets in this edition.

Shared by every surface that takes an address as a handle: adding an
organization member, inviting one, and claiming an operator identity for
password sign-in. One module rather than a private copy per service, because
two checks that disagree would accept an address on one surface and refuse the
same address on another, and the address is the key those surfaces match each
other on.

Deliberately a shape check and nothing more; see ``InvalidEmailError`` for why.
"""

from gateway.core.addresses import normalized_address
from gateway.services.tenancy.errors import InvalidEmailError

# ``user.email`` is ``varchar(255)``, so this is the column's width and not a
# policy. Request schemas carry it too, but theirs bounds the *raw* value and
# this one bounds the stored one, which is not the same number: lower-casing can
# make a string longer. "İ" (U+0130) lower-cases to two codepoints, so a raw
# address of exactly 255 characters ending in one normalizes to 256 and reaches
# the column over-width, which the driver answers with an error the caller
# cannot read. Checked here, after normalization, because here is the only place
# that sees the value that will actually be stored.
MAX_EMAIL_LENGTH = 255


def validated_email(email: str) -> str:
    """Normalize an address to lower case, refusing one that cannot be a handle.

    The shape check is ``core.addresses``', because "an address Otari would
    deliver to" is one question whether it is being invited, emailed, or signed
    in with, and that module is where the answer lives. What this adds is the
    stored-length rule, which belongs here rather than there: it is the width of
    ``user.email``, and only the surfaces that write that column care.

    Refuses on shape and on stored length. Both raise ``InvalidEmailError``: from
    the caller's side "this cannot be your address" is one answer, and splitting
    it would publish which of the two rules a probe tripped.
    """
    candidate = normalized_address(email)
    if candidate is None or len(candidate) > MAX_EMAIL_LENGTH:
        raise InvalidEmailError(email)
    return candidate


__all__ = ["MAX_EMAIL_LENGTH", "validated_email"]
