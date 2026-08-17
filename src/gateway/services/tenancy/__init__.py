"""The reconciled control plane's tenancy services.

A feature module: organizations, workspaces, memberships, and the first-boot
provisioning that gives a standalone deployment an identity to act as. The route
files under `gateway.api.routes` stay thin composition over these services.
"""

from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.workspace_service import WorkspaceService

__all__ = [
    "OrganizationService",
    "WorkspaceService",
    "ensure_bootstrap_identity",
]
