"""The reconciled control plane's tenancy services.

A feature module: organizations, workspaces, memberships, and the first-boot
provisioning that gives a standalone deployment an identity to act as. The route
files under `gateway.api.routes` stay thin composition over these services.

`org_provider_model_service` is deliberately **not** re-exported here, and
nothing else that reaches model discovery should be either. `workspace_scope`
imports this package (through `provisioning_service`), and that module dials
back through `model_discovery_service` to `provider_kwargs` to `alias_service`
to `workspace_scope`, so listing it here makes every module in that ring
unimportable first. `tests/unit/test_service_module_imports.py` is what says so.
Import it from its own module.
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
