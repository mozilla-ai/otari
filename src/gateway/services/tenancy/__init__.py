"""The reconciled control plane's tenancy services.

A feature module: organizations, workspaces, memberships, and the first-boot
provisioning that gives a standalone deployment an identity to act as. The route
files under `gateway.api.routes` stay thin composition over these services.

Nothing here may reach model discovery. `workspace_scope` imports this package
through `provisioning_service`, and discovery dials back through
`provider_kwargs` and `alias_service` to `workspace_scope`, so a name in that
ring listed here makes every module in it unimportable. The offered-models
service is in `services/providers/` for that reason among others.
"""

from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService
from gateway.services.tenancy.organization_domain_service import DomainAutoJoinListener, OrganizationDomainService
from gateway.services.tenancy.organization_service import InvitationClaimSettings, OrganizationService
from gateway.services.tenancy.password_reset_email import render_password_reset_email
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.tokens import generate_shared_token, generate_token, hash_token
from gateway.services.tenancy.user_service import (
    describe_caller,
    forget_unproven_access,
    report_new_account,
    unverified_identity_is_claimable,
    update_full_name,
)
from gateway.services.tenancy.workspace_service import WorkspaceService

__all__ = [
    "DeploymentUserService",
    "DomainAutoJoinListener",
    "InvitationClaimSettings",
    "OrgProviderKeyService",
    "OrganizationDomainService",
    "OrganizationService",
    "WorkspaceService",
    "describe_caller",
    "ensure_bootstrap_identity",
    "forget_unproven_access",
    "generate_shared_token",
    "generate_token",
    "hash_token",
    "render_password_reset_email",
    "report_new_account",
    "unverified_identity_is_claimable",
    "update_full_name",
]
