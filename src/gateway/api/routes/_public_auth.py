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

    Unconditional, not just on failure, the same reasoning
    ``invitations._throttle`` gives its own two routes: there is no legitimate
    caller here at a rate worth exempting, only a client with an address or a
    token it can retry from. Reuses the sign-in route's limiter/budget rather
    than a separate one.
    """
    limiter = getattr(request.app.state, "login_rate_limiter", None)
    if limiter is None:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    limiter.check(client_ip)


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


__all__ = ["mail_unavailable", "throttle_public_auth"]
