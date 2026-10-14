"""Shared plumbing for the public, unauthenticated auth routes.

``auth_signup.py`` and ``auth_password_reset.py`` both expose a caller who
holds neither the master key nor a session, the token or address in the
request being their whole proof of anything here, the same shape
``invitations.py`` already established for its own two routes. This module is
what keeps their two identical needs (a shared throttle, a shared mail-not-
configured rendering) from drifting: previously each route file carried its
own copy of both.
"""

from fastapi import HTTPException, Request, status

from gateway.services.mail import MailNotConfiguredError


def throttle_public_auth(request: Request) -> None:
    """Throttle calls to a public auth route per client IP.

    Unconditional, not just on failure: there is no legitimate caller here at a
    rate worth exempting, only a client with an address or a code it can retry
    from. The budget is ``public_auth_rate_limiter``, not the failed sign-in
    one, so a few mistyped passwords do not lock the same person (or everyone
    behind the same NAT) out of the link that would let them in.
    """
    limiter = getattr(request.app.state, "public_auth_rate_limiter", None)
    if limiter is None:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    limiter.check(client_ip)


def refuse_exhausted_verification_attempts(request: Request) -> None:
    """Refuse a verification from an IP that has used up its failed attempts, before any password is hashed.

    A verification link is checked against the password chosen at signup, so
    without this a held token would let a caller guess that password, and make
    the gateway run bcrypt, at whatever rate it can manage. Only failures are
    counted (``count_failed_verification``), so a person who opens their link
    and types the right password is never held back by anybody else's traffic.
    """
    limiter = getattr(request.app.state, "verification_failure_rate_limiter", None)
    client_ip = request.client.host if request.client else None
    if limiter is None or client_ip is None:
        return
    limiter.require_room(client_ip)


def count_failed_verification(request: Request) -> None:
    """Count one refused verification against the calling IP."""
    limiter = getattr(request.app.state, "verification_failure_rate_limiter", None)
    client_ip = request.client.host if request.client else None
    if limiter is None or client_ip is None:
        return
    try:
        limiter.check(client_ip)
    except HTTPException:
        # The refusal being counted is the answer this call owes; the 429 is for the next one.
        return


def mail_unavailable(exc: MailNotConfiguredError) -> HTTPException:
    """Render a mail-gated refusal the way ``mail.py``'s own ``send_test_mail`` does.

    Not a ``TenancyError``: the central handler in ``gateway.main`` renders
    every status of 500 or above with a generic "Internal server error" body,
    on purpose, for the errors that already live in that family (an operator
    problem the caller cannot act on). This one is the opposite: the missing
    settings are exactly what the caller, or the dashboard reading them,
    needs to see.
    """
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


__all__ = [
    "count_failed_verification",
    "mail_unavailable",
    "refuse_exhausted_verification_attempts",
    "throttle_public_auth",
]
