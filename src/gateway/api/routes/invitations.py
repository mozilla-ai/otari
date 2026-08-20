"""Accepting an organization invitation (standalone mode only).

Deliberately public, unlike every other route under ``/v1``: the person
following an emailed link holds no master key and no session, and the token
in the link is their whole proof of anything here. Both routes therefore take
no ``CurrentIdentity`` and are scoped to exactly the one invitation the token
names.

No session is minted on accept. Otari has no per-user sign-in yet (every
authenticated request resolves to the one bootstrap operator identity), so
there is nothing to sign this visitor into; accepting only resolves their
membership to ``active``, the same place ``POST /me/members`` already lands a
member added directly. They see the sign-in screen next, same as anyone else
added to an organization today.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db
from gateway.models.tenancy import (
    AcceptInvitationRequest,
    AcceptInvitationResultPublic,
    InvitationPreviewPublic,
    ValidateInvitationRequest,
)
from gateway.services.tenancy import OrganizationService

router = APIRouter(prefix="/v1/invitations", tags=["invitations"])


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the organization service on the request's session.

    A separate copy of the same tiny factory `organizations.py` declares: the
    two route modules deliberately don't import from each other (one is
    public, one is master-key gated), and importing the dependency alone
    across that boundary is not worth it for one function.
    """
    return OrganizationService(db)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


def _throttle(request: Request) -> None:
    """Throttle calls to these routes per client IP.

    ``POST /v1/auth/session`` is the only other unauthenticated route that
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
    service: OrganizationServiceDep,
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
    service: OrganizationServiceDep,
    body: AcceptInvitationRequest,
) -> AcceptInvitationResultPublic:
    """Accept a pending invitation, resolving it to an active membership."""
    _throttle(request)
    return await service.accept_invitation(body.token)
