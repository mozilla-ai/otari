"""Accepting an organization invitation (standalone mode only).

Deliberately public: the person following an emailed link holds no master
key and no session, and the token in the link is their whole proof of
anything here. Both routes therefore take no ``CurrentIdentity`` and are
scoped to exactly the one invitation the token names.

No session is minted on accept: accepting resolves the membership to
``active``, the same place ``POST /me/members`` already lands a member added
directly. An invitee who has never signed in can set a password in the same
call, which is the only way in on a deployment that sends no mail; without one,
the identity stays password-less until ``POST /api/v1/auth/signup`` or a
provider sign-in claims it.
"""

from fastapi import APIRouter, Request

from gateway.api.deps import OrganizationMembershipServiceDep
from gateway.models.tenancy import (
    AcceptInvitationRequest,
    AcceptInvitationResultPublic,
    InvitationPreviewPublic,
    ValidateInvitationRequest,
)

router = APIRouter(prefix="/invitations", tags=["invitations"])


def _throttle(request: Request) -> None:
    """Throttle calls to these routes per client IP.

    ``POST /api/v1/auth/session`` is the only other unauthenticated route that
    takes a credential, and it is IP-limited (``auth_session._check_login_rate_limit``,
    via ``app.state.login_rate_limiter``); these two were not, and ``accept``
    writes. The token's entropy (``secrets.token_urlsafe(32)``) already rules
    out guessing, so this isn't about brute-forcing a token: it's that these
    are the app's only unauthenticated write surface, reachable at whatever
    rate a client can manage, each call costing a handful of reads (``accept``
    several writes). Reuses the sign-in route's limiter/budget rather than a
    separate one, unconditionally (not just on failure, unlike sign-in): there
    is no legitimate caller here to avoid locking out, only a client with an
    address it can retry from.
    """
    limiter = getattr(request.app.state, "login_rate_limiter", None)
    if limiter is None:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    limiter.check(client_ip)


@router.post("/validate")
async def validate_invitation(
    request: Request,
    service: OrganizationMembershipServiceDep,
    body: ValidateInvitationRequest,
) -> InvitationPreviewPublic:
    """Look up a pending invitation by its token, for the accept page to render before committing.

    A ``POST`` with the token in the body, not a ``GET`` with it in the URL:
    the token is a bearer credential, and a URL path is what an access log or
    an intermediate proxy routinely retains.
    """
    _throttle(request)
    return await service.get_invitation_preview(body.token)


@router.post("/accept")
async def accept_invitation(
    request: Request,
    service: OrganizationMembershipServiceDep,
    body: AcceptInvitationRequest,
) -> AcceptInvitationResultPublic:
    """Accept a pending invitation, resolving it to an active membership and optionally setting a first password."""
    _throttle(request)
    return await service.accept_invitation(
        body.token,
        password=body.password,
        full_name=body.full_name,
        terms_accepted=body.terms_accepted,
    )
