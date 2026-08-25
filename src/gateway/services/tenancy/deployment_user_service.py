"""Deployment-wide account administration, for whoever operates the deployment.

Every other identity surface in this tree is scoped to one organization: the
workspace roster, the organization's Members & roles page, and the invitation
flow all read through a membership and cannot see past it. That is right for a
tenant and leaves an operator with nothing. Before this module the only
deployment-wide concepts were the ``is_superuser`` flag and the
``tenancy_bootstrap_user_id`` marker, neither of which had an API, so the
recourse for a stuck or abusive account was SQL (mozilla-ai/otari#797).

Three operations, and deliberately only three: list every identity, deactivate
or reactivate one, and flip its ``is_superuser`` flag. Creating and inviting stay
with the organization surface, which is where the membership that makes an
identity useful is created, and deleting is not here at all for the reason
``routes/organizations.py`` gives about deleting an organization: historical
attribution resolves through rows that hang off the identity.

**Who may reach it.** A superuser, or the identity the ``tenancy_bootstrap_user_id``
marker names. The second is not redundant with the first even though provisioning
makes the bootstrap operator a superuser: the marker is what still admits an
operator whose flag was cleared, by hand or by another operator, and it is the
reason this surface cannot lock a deployment out of itself. Everyone else is
refused with 404 rather than 403; see
:class:`~gateway.services.tenancy.errors.DeploymentAdministrationUnavailableError`
for why the status is the one it is.

**What it refuses.** Two lockout guards, both narrow and both directional, so a
repair is still possible where a foot-gun is not:

- an operator may not deactivate *themselves* or clear their *own* superuser
  flag, because either ends the session they are holding with nothing on the
  page to undo it;
- neither may be turned off on the bootstrap operator, because master-key
  sign-in mints a session for that identity and ``resolve_dashboard_session``
  refuses a deactivated one, so deactivating it turns the deployment's fallback
  credential into a session that dies on arrival.

Turning either flag *on* is unguarded in both cases, which is what makes
granting yourself back a cleared flag, or reactivating the bootstrap identity,
something an operator can still do here.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    DeploymentUserOrganizationPublic,
    DeploymentUserPublic,
    DeploymentUsersPublic,
    DeploymentUserUpdateRequest,
    Organization,
    OrganizationMember,
    User,
)
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.services.dashboard_session_service import revoke_user_dashboard_sessions
from gateway.services.tenancy.errors import (
    BootstrapOperatorProtectedError,
    DeploymentAdministrationUnavailableError,
    DeploymentUserNotFoundError,
    DeploymentUserSelfChangeError,
    EmptyDeploymentUserUpdateError,
)
from gateway.services.tenancy.provisioning_service import load_bootstrap_identity


class DeploymentUserService:
    """Reads and writes identities across the whole deployment."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.users = UserRepository(db)
        self.members = OrganizationMemberRepository(db)

    async def has_administration_access(self, actor: User) -> bool:
        """Whether this caller may reach the surface at all.

        The predicate form exists for the one endpoint that has to *report* the
        answer rather than act on it, the same split
        ``authorization.has_workspace_management_access`` makes: the dashboard
        asks so it can hide a destination, and hiding it is a convenience that
        never grants anything.

        The marker lookup is skipped for a superuser, which is every caller on a
        deployment nobody has changed, so the common path stays one comparison
        against a row the request already resolved.
        """
        if actor.is_superuser:
            return True
        bootstrap = await load_bootstrap_identity(self.db)
        return bootstrap is not None and bootstrap.id == actor.id

    async def list_users(
        self,
        *,
        actor: User,
        skip: int = 0,
        limit: int = 100,
    ) -> DeploymentUsersPublic:
        """List every identity on the deployment, with the organizations it belongs to."""
        await self._require_administration_access(actor)

        rows, count = await self.users.list_all(skip=skip, limit=limit)
        # One query for the page's memberships rather than one per row, the same
        # shape the organization roster uses to resolve its own joins.
        organizations = await self.members.get_by_users_with_organizations(user.id for user in rows)
        bootstrap = await load_bootstrap_identity(self.db)
        bootstrap_id = bootstrap.id if bootstrap is not None else None
        return DeploymentUsersPublic(
            data=[
                _to_public(
                    user,
                    organizations.get(user.id, []),
                    bootstrap_id=bootstrap_id,
                    actor_id=actor.id,
                )
                for user in rows
            ],
            count=count,
        )

    async def update_user(
        self,
        *,
        actor: User,
        user_id: uuid.UUID,
        request: DeploymentUserUpdateRequest,
    ) -> DeploymentUserPublic:
        """Deactivate, reactivate, or change what one identity may administer.

        Deactivating also ends that identity's dashboard sessions here, in this
        transaction, rather than leaving them to be refused when a cookie next
        comes back: ``resolve_dashboard_session`` does refuse them, but it only
        runs when the browser asks, so until then the rows are alive and
        reactivating would hand back every cookie the account held.
        """
        await self._require_administration_access(actor)
        if request.is_active is None and request.is_superuser is None:
            raise EmptyDeploymentUserUpdateError

        target = await self.users.get(user_id)
        if target is None:
            raise DeploymentUserNotFoundError(user_id)

        # Only the directions that can lock somebody out are guarded; see the
        # module docstring for why turning either flag on is not.
        removes_access = request.is_active is False or request.is_superuser is False
        if removes_access:
            if target.id == actor.id:
                raise DeploymentUserSelfChangeError(
                    "You cannot deactivate your own account or remove your own operator access"
                )
            bootstrap = await load_bootstrap_identity(self.db)
            if bootstrap is not None and bootstrap.id == target.id:
                raise BootstrapOperatorProtectedError(
                    "The deployment's bootstrap operator cannot be deactivated or have its operator access removed"
                )

        if request.is_active is not None:
            target.is_active = request.is_active
        if request.is_superuser is not None:
            target.is_superuser = request.is_superuser
        self.db.add(target)
        if request.is_active is False:
            await revoke_user_dashboard_sessions(self.db, target.id)
        await self.db.commit()
        await self.db.refresh(target)

        memberships = await self.members.get_by_users_with_organizations([target.id])
        bootstrap = await load_bootstrap_identity(self.db)
        return _to_public(
            target,
            memberships.get(target.id, []),
            bootstrap_id=bootstrap.id if bootstrap is not None else None,
            actor_id=actor.id,
        )

    async def _require_administration_access(self, actor: User) -> None:
        """Raise the 404 unless this caller operates the deployment."""
        if not await self.has_administration_access(actor):
            raise DeploymentAdministrationUnavailableError


def _to_public(
    user: User,
    memberships: list[tuple[OrganizationMember, Organization]],
    *,
    bootstrap_id: uuid.UUID | None,
    actor_id: uuid.UUID,
) -> DeploymentUserPublic:
    """Render one identity and its memberships as the surface returns them."""
    return DeploymentUserPublic(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        is_bootstrap_operator=bootstrap_id is not None and bootstrap_id == user.id,
        is_self=user.id == actor_id,
        last_sign_in_at=user.last_sign_in_at,
        created_at=user.created_at,
        organizations=[
            DeploymentUserOrganizationPublic(
                organization_id=organization.id,
                name=organization.name,
                slug=organization.slug,
                role=membership.role,
                status=membership.status,
            )
            for membership, organization in memberships
        ],
    )


__all__ = ["DeploymentUserService"]
