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


class LastWorkspaceError(TenancyValidationError):
    """Deleting this workspace would leave the organization without one."""

    def __init__(self) -> None:
        super().__init__(
            "An organization keeps at least one workspace; create another before deleting this one"
        )


__all__ = [
    "ForeignTenancyError",
    "InvalidEmailError",
    "InvalidRoleError",
    "LastWorkspaceError",
    "MembershipUpdateError",
    "NotAnOrganizationMemberError",
    "NotAuthorizedError",
    "OrganizationMemberAlreadyExistsError",
    "OrganizationMemberNotFoundError",
    "OrganizationNotFoundError",
    "TenancyConflictError",
    "TenancyError",
    "TenancyForbiddenError",
    "TenancyNotFoundError",
    "TenancyValidationError",
    "WorkspaceAlreadyExistsError",
    "WorkspaceMemberAlreadyExistsError",
    "WorkspaceMemberNotFoundError",
    "WorkspaceNameRequiredError",
    "WorkspaceNotFoundError",
]
