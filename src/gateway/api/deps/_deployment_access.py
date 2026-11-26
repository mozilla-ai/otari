"""Building the reader that answers whether a caller operates the deployment.

Its own module because the authentication gates and the identity builders both
need it, and those two import in one direction.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.repositories.organizations import BootstrapMarkerRepository
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.services.identity import DeploymentAdministrationAccess
from gateway.services.organizations import BootstrapOperatorMarker


def deployment_administration_access(db: AsyncSession) -> DeploymentAdministrationAccess:
    """Build the deployment administration reader on one session.

    Shared by every gate and read that asks whether a caller operates the
    deployment, so they cannot come to build it from different collaborators.
    """
    return DeploymentAdministrationAccess(
        UserRepository(db),
        OrganizationMemberRepository(db),
        BootstrapOperatorMarker(BootstrapMarkerRepository(db)),
    )
