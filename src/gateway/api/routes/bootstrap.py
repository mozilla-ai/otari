"""Deployment bootstrap the dashboard shell reads before it renders.

One server-derived payload that selects the runtime context: which deployment is
serving this URL, what kind of session it issues, which management surfaces it
offers, and, for a hybrid gateway, where its control plane actually lives. The
shell reads it once and gates navigation on it, so no page component has to ask
which mode the gateway is in.

Registered in both modes, and unauthenticated by necessity: this is what tells a
browser whether a sign-in screen is even the right thing to show. It therefore
carries no secret. In particular it never carries the platform token, and
``management_url`` is a link target an operator configured, not a credential.

The contract is shared with otari.ai, which serves the same shape for its hosted
deployment (mozilla-ai/otari-ai#1591). That is why ``deployment_type`` and
``session_type`` name three values each while this server only ever produces two
of them: the enum is the contract, not this deployment's inventory.
"""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from gateway.api.deps import get_config
from gateway.core.config import GatewayConfig

router = APIRouter(prefix="/v1/bootstrap", tags=["bootstrap"])

DeploymentType = Literal["standalone", "hosted", "hybrid"]
SessionType = Literal["local_operator", "hosted_user", "none"]

# The management API groups a standalone gateway serves, one name per ``/v1/``
# router the dashboard's surfaces are built on. Naming the groups rather than the
# pages keeps the list checkable: ``test_deployment_bootstrap`` asserts every
# name here is a route this app actually mounts, so a capability cannot outlive
# the API behind it. Several pages share one (Activity and Usage are both views
# over ``/v1/usage``), and the Overview index needs none.
#
# A hybrid gateway serves none of them: its control plane is otari.ai, and a
# second management UI beside it is what the deployment contract rules out.
# The set is therefore all-or-nothing here, because this gateway's management API
# is: `register_routers` mounts the whole of it in standalone and none of it in
# hybrid. It travels as a set rather than as a second reading of the mode because
# the shared shell also runs against a hosted control plane, where the two come
# apart.
STANDALONE_CAPABILITIES: tuple[str, ...] = (
    "budgets",
    "keys",
    "models",
    "providers",
    "routing",
    "settings",
    "tools",
    "usage",
    "users",
)


class DeploymentBootstrap(BaseModel):
    """What the dashboard shell needs before it can render anything."""

    deployment_type: DeploymentType = Field(
        description=(
            "Which deployment serves this URL. 'standalone' owns its own data; "
            "'hosted' is otari.ai; 'hybrid' is a gateway attached to otari.ai, which is "
            "data-plane only and holds no management surface of its own."
        )
    )
    session_type: SessionType = Field(
        description=(
            "The kind of session this deployment issues, not whether the caller holds one. "
            "'local_operator' is the standalone master-key sign-in, 'hosted_user' an otari.ai "
            "account, and 'none' a deployment that issues no management session at all."
        )
    )
    capabilities: list[str] = Field(
        description=(
            "Management API groups this deployment serves, sorted, which is what its dashboard "
            "surfaces gate on. Empty for a hybrid gateway."
        )
    )
    management_url: str | None = Field(
        description=(
            "Where the authoritative control plane lives when it is not this deployment. "
            "Set for a hybrid gateway so its landing page can link to otari.ai; null otherwise."
        )
    )


@router.get("", response_model=DeploymentBootstrap)
async def get_bootstrap(config: GatewayConfig = Depends(get_config)) -> DeploymentBootstrap:
    """Return the deployment context the dashboard shell renders from.

    Public: the shell fetches this before it knows whether it can authenticate.
    """
    if config.is_hybrid_mode:
        return DeploymentBootstrap(
            deployment_type="hybrid",
            session_type="none",
            capabilities=[],
            management_url=config.platform_management_url,
        )
    return DeploymentBootstrap(
        deployment_type="standalone",
        session_type="local_operator",
        capabilities=sorted(STANDALONE_CAPABILITIES),
        management_url=None,
    )
