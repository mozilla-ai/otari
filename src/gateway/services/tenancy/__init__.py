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
from gateway.services.tenancy.organization_domain_service import OrganizationDomainService
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.workspace_service import WorkspaceService

__all__ = [
    "DeploymentUserService",
    "OrgProviderKeyService",
    "OrganizationDomainService",
    "OrganizationService",
    "WorkspaceService",
    "ensure_bootstrap_identity",
]
