"""Errors the organization and workspace surfaces raise, and the HTTP status each carries."""

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


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


class OrganizationDomainNotFoundError(TenancyNotFoundError):
    def __init__(self, organization_domain_id: object):
        super().__init__(f"Organization domain {organization_domain_id} not found")


class OrganizationDomainAlreadyClaimedError(TenancyConflictError):
    """The domain is already claimed, and the message does not say by whom.

    Deliberately silent about the holder. A claim is unique deployment-wide, so
    a message naming the other organization would let anyone who can create one
    probe for which domains their neighbours have registered.
    """

    def __init__(self, domain: str):
        super().__init__(f"'{domain}' is already claimed")


class OrganizationDomainClaimedHereError(TenancyConflictError):
    """This organization already holds a claim on the domain.

    Distinct from ``OrganizationDomainAlreadyClaimedError``, which is about
    somebody else's proven claim and deliberately says nothing about who: this
    one is about the caller's own row, which they can already see and act on, so
    it names the situation plainly.
    """

    def __init__(self, domain: str):
        super().__init__(f"'{domain}' is already claimed by this organization")


class TooManyOrganizationDomainsError(TenancyValidationError):
    """The organization is at its claim ceiling.

    A ceiling exists because every unverified claim is a name this deployment
    will run an outbound DNS query against on demand.
    """

    def __init__(self, limit: int):
        super().__init__(f"An organization may claim at most {limit} email domains")


class OrganizationDomainNotVerifiedError(TenancyValidationError):
    """The expected TXT record was not found at the claimed domain."""

    def __init__(self, domain: str):
        super().__init__(
            f"No matching TXT record was found at {domain}. Publish the record shown for this "
            "domain at its apex and try again; DNS changes can take a while to propagate."
        )


class UnregistrableDomainError(TenancyValidationError):
    def __init__(self, value: str):
        super().__init__(f"'{value}' is not a valid email domain")


class PublicEmailDomainError(TenancyValidationError):
    """A free provider cannot be claimed, whoever proves what.

    Separate from ``UnregistrableDomainError`` because the two need different
    things from the admin: one is a typo to correct, the other is a domain that
    will never be allowed however it is spelled.
    """

    def __init__(self, domain: str):
        super().__init__(
            f"'{domain}' is a public email provider and cannot be claimed for auto-join. "
            "Use a domain your organization controls."
        )


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


class OrganizationSlugUnavailableError(TenancyConflictError):
    """A created organization's derived slug collided with one already stored.

    The slug is the name reduced to a stem plus four random bytes, so two
    organizations may share a name and a collision needs the same stem *and*
    the same suffix. Reported rather than retried because retrying means
    rolling back the whole unit of work (the organization, the owner
    membership, and the first workspace) to change one column, and the caller
    can simply send the request again.
    """

    def __init__(self) -> None:
        super().__init__("Could not allocate a unique organization slug; retry the request")


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
        super().__init__("An organization keeps at least one workspace; create another before deleting this one")


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


class InvitationPasswordNotAcceptedError(TenancyValidationError):
    """A password sent with an accept for an address that already has a way to sign in.

    Refused rather than applied: an invitation link can be handed over by hand,
    so letting it replace an existing credential would let whoever holds a
    forwarded link take over the account it names.
    """

    def __init__(self) -> None:
        super().__init__("This address can already sign in; accept without a password and sign in as usual")


class WorkspaceActivationUnavailableError(TenancyConflictError):
    """The first-request setup guide is not on offer for this workspace.

    The deployment turned it off, the workspace is classified out of it, or
    someone dismissed it. A conflict rather than a 403: the caller is allowed to
    manage this workspace, the flow they are asking to act on is simply retired,
    and the message says which of the three it was.
    """


class WorkspaceAlreadyActivatedError(TenancyConflictError):
    """A request in this workspace has already succeeded, so the guide is finished.

    Reached by a browser tab left open across the first successful request, which
    is the one caller likely to ask a retired guide for a credential.
    """

    def __init__(self) -> None:
        super().__init__("This workspace has already served a successful request")


__all__ = [
    "ForeignTenancyError",
    "InvalidRoleError",
    "InvitationAlreadyPendingError",
    "InvitationAlreadyUsedError",
    "InvitationExpiredError",
    "InvitationNotFoundError",
    "InvitationPasswordNotAcceptedError",
    "LastWorkspaceError",
    "MembershipUpdateError",
    "NotAnOrganizationMemberError",
    "NotAuthorizedError",
    "OrganizationDomainAlreadyClaimedError",
    "OrganizationDomainClaimedHereError",
    "OrganizationDomainNotFoundError",
    "OrganizationDomainNotVerifiedError",
    "OrganizationMemberAlreadyExistsError",
    "OrganizationMemberNotFoundError",
    "OrganizationNameRequiredError",
    "OrganizationNotFoundError",
    "OrganizationSlugUnavailableError",
    "PublicEmailDomainError",
    "TooManyOrganizationDomainsError",
    "UnregistrableDomainError",
    "WorkspaceActivationUnavailableError",
    "WorkspaceAlreadyActivatedError",
    "WorkspaceAlreadyExistsError",
    "WorkspaceInUseError",
    "WorkspaceMemberAlreadyExistsError",
    "WorkspaceMemberNotFoundError",
    "WorkspaceNameRequiredError",
    "WorkspaceNotFoundError",
]
