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

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.tenancy.user_service import operator_has_password

router = APIRouter(prefix="/v1/bootstrap", tags=["bootstrap"])

DeploymentType = Literal["standalone", "hosted", "hybrid"]
SessionType = Literal["local_operator", "hosted_user", "none"]
# How a caller may sign in to this deployment. ``master_key`` is the first-boot
# credential and ``password`` the steady-state one, and a standalone gateway
# offers exactly one of them: the master key until the operator claims the
# deployment with a password, and the password from then on
# (mozilla-ai/otari-ai#1716). A list rather than a single value because #651 and
# #652 add methods that coexist with the password rather than replacing it, and
# because a hybrid gateway offers none.
SignInMethod = Literal["master_key", "password"]

# The management API groups a standalone gateway serves, one name per ``/v1/``
# router the dashboard's surfaces are built on. Naming the groups rather than the
# pages keeps the list checkable: ``test_deployment_bootstrap`` asserts every
# name here is a route this app actually mounts, so a surface cannot outlive the
# API behind it. Several pages share one (Activity and Usage are both views over
# ``/v1/usage``), and the Overview index needs none.
#
# Deliberately *not* called capabilities. otari.ai already spends that word on
# the entitlement axis, down to a nav item's ``capability`` field and a
# ``routing`` entry that means "this org is licensed for routing". This axis
# answers something else entirely, "does this process host the surface at all",
# and the two vocabularies meet in one shell at M5. See ARCHITECTURE.md.
#
# A hybrid gateway serves none of them: its control plane is otari.ai, and a
# second management UI beside it is what the deployment contract rules out.
# The set is therefore all-or-nothing here, because this gateway's management API
# is: `register_routers` mounts the whole of it in standalone and none of it in
# hybrid. It travels as a set rather than as a second reading of the mode because
# the shared shell also runs against a hosted control plane, where the two come
# apart.
STANDALONE_SURFACES: tuple[str, ...] = (
    "budgets",
    "keys",
    "models",
    "organizations",
    "pricing",
    "providers",
    "routing",
    "settings",
    "tools",
    "usage",
    "users",
    "workspaces",
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
            "'local_operator' is the standalone operator sign-in (see sign_in_methods for which "
            "credential it currently accepts), 'hosted_user' an otari.ai "
            "account, and 'none' a deployment that issues no management session at all."
        )
    )
    surfaces: list[str] = Field(
        description=(
            "Management API groups this deployment serves, sorted, which is what its dashboard "
            "pages gate on. Named surfaces, not capabilities: capability is otari.ai's word for "
            "the entitlement (licensing) axis, and this is the deployment (topology) axis. "
            "Empty for a hybrid gateway."
        )
    )
    management_url: str | None = Field(
        description=(
            "Where the authoritative control plane lives when it is not this deployment. "
            "Set for a hybrid gateway so its landing page can link to otari.ai; null otherwise."
        )
    )
    sign_in_methods: list[SignInMethod] = Field(
        description=(
            "How POST /v1/auth/session may be authenticated right now, sorted. 'master_key' is the "
            "first-boot credential and is offered until some identity on this deployment has a "
            "password; 'password' replaces it from then on, and the master key stays the credential "
            "for the management API. Empty for a hybrid gateway, which issues no session. The login "
            "page renders from this rather than trying a credential to find out."
        )
    )
    mail_ready: bool = Field(
        description=(
            "Whether this deployment can deliver a message carrying a link back to itself "
            "(an invitation's accept link, and the verification and reset links to come), "
            "not merely whether a transport is configured: it also needs to know its own "
            "public URL to put in one. Lets the dashboard disable or hide a mail-dependent "
            "affordance instead of offering one that would fail at send time. Every "
            "message this control plane sends carries such a link, which is why this is "
            "one flag and not one per feature. False for a hybrid gateway, whose control "
            "plane is otari.ai and which sends no mail of its own."
        )
    )


@router.get("", response_model=DeploymentBootstrap)
async def get_bootstrap(
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> DeploymentBootstrap:
    """Return the deployment context the dashboard shell renders from.

    Public: the shell fetches this before it knows whether it can authenticate.
    That is also why ``sign_in_methods`` is answered here rather than behind a
    credential, and it publishes nothing an unauthenticated caller could not
    already learn by trying both credentials against the sign-in endpoint.

    The one database read is a ``LIMIT 1`` probe for any identity holding a
    password, over a table a standalone deployment keeps one row per person in.
    It runs only in standalone mode: a hybrid gateway has no session to describe,
    and ``get_db_if_needed`` hands it no session to read one from.
    """
    if config.is_hybrid_mode:
        return DeploymentBootstrap(
            deployment_type="hybrid",
            session_type="none",
            surfaces=[],
            sign_in_methods=[],
            management_url=config.platform_management_url,
            mail_ready=False,
        )
    assert db is not None  # get_db_if_needed yields a session outside hybrid mode
    return DeploymentBootstrap(
        deployment_type="standalone",
        session_type="local_operator",
        surfaces=sorted(STANDALONE_SURFACES),
        sign_in_methods=await _sign_in_methods(db),
        management_url=None,
        mail_ready=config.mail_ready,
    )


async def _sign_in_methods(db: AsyncSession) -> list[SignInMethod]:
    """Which credential ``POST /v1/auth/session`` accepts on this deployment.

    A database failure answers "none" rather than propagating. This route is the
    first thing the dashboard shell fetches, so a 500 here is a blank page
    instead of a login screen, and it would be a blank page for the one outage
    where an operator most wants the dashboard to say something. "None" is also
    the truth while the database is unreachable: no session can be minted either
    way, since minting one writes a row.
    """
    try:
        claimed = await operator_has_password(db)
    except SQLAlchemyError:
        logger.warning("Could not read which sign-in methods this deployment offers", exc_info=True)
        return []
    return ["password"] if claimed else ["master_key"]
