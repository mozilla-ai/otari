"""Minting the token for one send of an invitation."""

from dataclasses import dataclass
from datetime import datetime

from gateway.core.tokens import generate_shared_token, generate_token, hash_token
from gateway.services.mail import Mailer
from gateway.services.organizations._accept_link import accept_link


@dataclass(frozen=True, kw_only=True)
class Mint:
    """A token minted for a send, and the link to hand back for it when this deployment cannot email it.

    NOTE: ``accept_link`` is set exactly when ``next_send_at`` is not:
    a link that is handed back is never on a row with a send due.
    """

    accept_link: str | None
    next_send_at: datetime | None
    token_hash: str


def mint_send(mailer: Mailer, now: datetime) -> Mint:
    """Mint a token for one send of an invitation, due now or carried by hand."""
    can_deliver = mailer.can_deliver_links
    token = generate_token() if can_deliver else generate_shared_token()
    return Mint(
        accept_link=None if can_deliver else accept_link(mailer, token),
        next_send_at=now if can_deliver else None,
        token_hash=hash_token(token),
    )
