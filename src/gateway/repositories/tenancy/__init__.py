"""Data access for the reconciled control plane's tenancy tables.

A package rather than four more files beside `users_repository.py`: the tenancy
slice arrives as one graph (`gateway.models.tenancy`) and its repositories are
read together. `users_repository.py` next door serves the gateway's own legacy
string-keyed user table, which is a different thing with a confusingly similar
name during the strangle.

**Every column reference goes through ``sqlmodel.col()``.** On a SQLModel class
the attribute's static type is the annotation, so ``Organization.slug == slug``
reads as ``bool`` and mypy rejects it where a ``ColumnElement[bool]`` is
expected. ``col()`` recovers the column expression, and using it everywhere
keeps the slice free of the blanket ``attr-defined``/``arg-type`` exemptions the
platform's own mypy config needs.
"""

from gateway.repositories.tenancy.invitation_repository import InvitationRepository
from gateway.repositories.tenancy.organization_member_repository import OrganizationMemberRepository
from gateway.repositories.tenancy.organization_repository import OrganizationRepository
from gateway.repositories.tenancy.user_repository import UserRepository, user_alphabetical_order
from gateway.repositories.tenancy.workspace_repository import WorkspaceMemberRepository, WorkspaceRepository

__all__ = [
    "InvitationRepository",
    "OrganizationMemberRepository",
    "OrganizationRepository",
    "UserRepository",
    "WorkspaceMemberRepository",
    "WorkspaceRepository",
    "user_alphabetical_order",
]
