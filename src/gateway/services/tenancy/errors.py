"""Domain errors for the tenancy services, and the HTTP status each carries.

The platform maps roughly 25 exception modules to statuses in one central table
(`otari-ai` `backend/app/api/exception_handlers.py`). The gateway has no such
table and its routes raise ``HTTPException`` directly, which would mean a
try/except around every one of the tenancy handlers. Instead each error names
its own status here and one handler, registered in `gateway.main`, renders it as
FastAPI's own ``{"detail": ...}`` body, so a rehomed service keeps raising
domain errors and the routes stay thin.

The status is on the class rather than at the raise site because it is a
property of the condition, not of the endpoint that hit it: a workspace the
caller may not see is a 404 whichever route asked for it.
"""

from fastapi import status


class TenancyError(Exception):
    """Base class for a tenancy operation that cannot be completed."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TenancyNotFoundError(TenancyError):
    """A resource does not exist, or does not exist *for this caller*.

    The two are deliberately one status. A workspace in another organization
    must not be distinguishable from a workspace that was never created, or the
    404 becomes an existence oracle for other tenants' data.
    """

    status_code = status.HTTP_404_NOT_FOUND


class TenancyForbiddenError(TenancyError):
    """The caller is known, the resource is visible, and the action is not theirs."""

    status_code = status.HTTP_403_FORBIDDEN


class TenancyConflictError(TenancyError):
    """The request collides with something that already exists."""

    status_code = status.HTTP_409_CONFLICT


class TenancyValidationError(TenancyError):
    """The request is well-formed but would leave the tenancy model inconsistent."""

    status_code = status.HTTP_400_BAD_REQUEST


class OrganizationNotFoundError(TenancyNotFoundError):
    def __init__(self, organization_id: object):
        super().__init__(f"Organization {organization_id} not found")


class OrganizationMemberNotFoundError(TenancyNotFoundError):
    def __init__(self, organization_member_id: object):
        super().__init__(f"Organization member {organization_member_id} not found")


class WorkspaceNotFoundError(TenancyNotFoundError):
    def __init__(self, workspace_id: object):
        super().__init__(f"Workspace {workspace_id} not found")


class WorkspaceMemberNotFoundError(TenancyNotFoundError):
    def __init__(self, workspace_id: object, user_id: object):
        super().__init__(f"User {user_id} is not a member of workspace {workspace_id}")


class WorkspaceAlreadyExistsError(TenancyConflictError):
    def __init__(self, name: str):
        super().__init__(f"A workspace named '{name}' already exists in this organization")


class WorkspaceMemberAlreadyExistsError(TenancyConflictError):
    def __init__(self, user_id: object):
        super().__init__(f"User {user_id} is already a member of this workspace")


class NotAuthorizedError(TenancyForbiddenError):
    def __init__(self, message: str = "Not enough privileges to perform this action"):
        super().__init__(message)


class MembershipUpdateError(TenancyValidationError):
    """A membership change the organization's own rules refuse (e.g. the last owner)."""


class NotAnOrganizationMemberError(TenancyValidationError):
    def __init__(self, user_id: object):
        super().__init__(f"User {user_id} is not an active member of this organization")


class InvalidRoleError(TenancyValidationError):
    def __init__(self, role: str, allowed: set[str]):
        super().__init__(f"Invalid role '{role}'; expected one of {', '.join(sorted(allowed))}")


class OrganizationMemberAlreadyExistsError(TenancyConflictError):
    def __init__(self, identifier: object):
        super().__init__(f"{identifier} is already an active member of this organization")


class InvitationAlreadyPendingError(TenancyConflictError):
    """The address already has a live, unexpired invitation, so a fresh one is refused rather than piled on.

    Distinct from ``OrganizationMemberAlreadyExistsError``: that message says
    "already an active member", which is false for an address that is only
    ``invited``. Resending is revoke (which cancels the pending invitation and
    suspends the membership) followed by a fresh invite; once the existing
    invitation's own expiry has passed, a fresh invite supersedes it directly
    instead of raising this.
    """

    def __init__(self, identifier: object):
        super().__init__(f"{identifier} already has a pending invitation")


class InvalidEmailError(TenancyValidationError):
    """An address that could not be a claim handle.

    Deliberately a shape check and nothing more. The address is not delivered to
    by this edition, and ownership of it is proven by the claim flow that
    arrives with sign-in, so anything stricter here would be theater.
    """

    def __init__(self, email: str):
        super().__init__(f"'{email}' is not a valid email address")


class WorkspaceNameRequiredError(TenancyValidationError):
    """A workspace name that is absent, null, or blank once trimmed.

    ``Workspace.name`` is NOT NULL and carries no minimum length, and SQLModel
    skips validation when constructing a table instance, so without this a
    ``{"name": null}`` update reaches the column as a NOT NULL violation and a
    ``{"name": ""}`` create stores a nameless workspace.
    """

    def __init__(self) -> None:
        super().__init__("A workspace name is required")


class ForeignTenancyError(TenancyError):
    """The database holds organizations this deployment did not provision.

    A 500 rather than a client error, because nothing the caller sent is wrong:
    the deployment is pointed at a database it cannot serve, and that is an
    operator's problem to fix before any request can succeed.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class OrganizationNameRequiredError(TenancyValidationError):
    """An organization name that is absent, null, or blank once trimmed.

    The request's ``min_length=1`` admits a single space, so this is what stops a
    whitespace-only rename from reaching the column, in the same way
    ``WorkspaceNameRequiredError`` does for a workspace.
    """

    def __init__(self) -> None:
        super().__init__("An organization name is required")


class WorkspaceInUseError(TenancyConflictError):
    """A workspace still holds request-plane rows, which are ON DELETE RESTRICT.

    Keys, usage, aliases and policies are restricted rather than cascaded on
    purpose: a workspace is a billing scope, and deleting one should not take
    the record of what was spent in it with it. So this is a real refusal with a
    real reason, not the integrity error escaping as a 500.
    """

    status_code = status.HTTP_409_CONFLICT

    def __init__(self) -> None:
        super().__init__(
            "This workspace still holds API keys, usage, aliases or routing policies. "
            "Move or delete those first; they are kept rather than cascaded so a "
            "workspace's spend history survives it."
        )


class LastWorkspaceError(TenancyValidationError):
    """Deleting this workspace would leave the organization without one."""

    def __init__(self) -> None:
        super().__init__(
            "An organization keeps at least one workspace; create another before deleting this one"
        )


# The two below are pricing errors in a tenancy module, because the status
# mapping is what decides where an error class lives here: one handler is
# registered for ``TenancyError`` (see `gateway.main`), so an organization-scoped
# error that wants a status rather than a 500 has to descend from it. Naming them
# for what they are keeps that visible.
class OrganizationPricingNotFoundError(TenancyNotFoundError):
    """No pricing override with this id in the caller's organization.

    One status for "never existed" and "belongs to another organization", the same
    rule the base class states: a distinguishable 404 would tell one tenant which
    ids exist in another.
    """

    def __init__(self, pricing_id: object):
        super().__init__(f"Pricing override {pricing_id} not found")


class OrganizationPricingOverlapError(TenancyConflictError):
    """A period that would overlap one this organization already priced.

    ``model_pricing`` lets a later row shadow an earlier one, because a catalog is
    re-imported wholesale. An override is a commitment for a period, so two
    periods covering one instant is an unanswerable question rather than a newest
    wins rule, and it is refused with both periods named.
    """

    def __init__(self, model_key: str, existing_period: str):
        super().__init__(
            f"An override for '{model_key}' already covers part of that period ({existing_period}). "
            "Change this period, or edit the existing override instead."
        )
class InvitationNotFoundError(TenancyNotFoundError):
    """No invitation matches the token or id given.

    One status for "wrong token", "unknown id", and "someone else's
    invitation", for the same reason ``TenancyNotFoundError`` gives every
    cross-tenant lookup one status: distinguishing them would let a caller
    probe for which is true.
    """

    def __init__(self, identifier: object = "invitation") -> None:
        super().__init__(f"{identifier} not found or already used")


class InvitationExpiredError(TenancyValidationError):
    """The invitation's ``expires_at`` has passed."""

    def __init__(self) -> None:
        super().__init__("This invitation has expired")


class InvitationAlreadyUsedError(TenancyValidationError):
    """The invitation is not ``pending`` (already accepted, cancelled, or expired)."""

    def __init__(self) -> None:
        super().__init__("This invitation has already been used or is no longer valid")


class WorkspaceBudgetDefaultNotFoundError(TenancyNotFoundError):
    def __init__(self, default_id: object):
        super().__init__(f"Workspace budget default {default_id} not found")


class WorkspaceBudgetDefaultAlreadyExistsError(TenancyConflictError):
    def __init__(self, workspace_id: object, provider_key_id: object):
        scope = "every provider" if provider_key_id is None else f"provider '{provider_key_id}'"
        super().__init__(f"Workspace {workspace_id} already has a budget default for {scope}")


__all__ = [
    "ForeignTenancyError",
    "InvalidEmailError",
    "InvalidRoleError",
    "InvitationAlreadyPendingError",
    "InvitationAlreadyUsedError",
    "InvitationExpiredError",
    "InvitationNotFoundError",
    "LastWorkspaceError",
    "MembershipUpdateError",
    "NotAnOrganizationMemberError",
    "NotAuthorizedError",
    "OrganizationMemberAlreadyExistsError",
    "OrganizationMemberNotFoundError",
    "OrganizationNameRequiredError",
    "OrganizationNotFoundError",
    "OrganizationPricingNotFoundError",
    "OrganizationPricingOverlapError",
    "TenancyConflictError",
    "TenancyError",
    "TenancyForbiddenError",
    "TenancyNotFoundError",
    "TenancyValidationError",
    "WorkspaceAlreadyExistsError",
    "WorkspaceBudgetDefaultAlreadyExistsError",
    "WorkspaceBudgetDefaultNotFoundError",
    "WorkspaceInUseError",
    "WorkspaceMemberAlreadyExistsError",
    "WorkspaceMemberNotFoundError",
    "WorkspaceNameRequiredError",
    "WorkspaceNotFoundError",
]
