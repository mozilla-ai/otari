"""Deployment bootstrap the dashboard shell reads before it renders.

One server-derived payload that selects the runtime context: which deployment is
serving this URL, what kind of session it issues, which management surfaces it
offers, and, for a hybrid gateway, where its control plane actually lives. The
shell reads it once and gates navigation on it, so no page component has to ask
which mode the gateway is in.

Registered in both modes, and unauthenticated by necessity: this is what tells a
browser whether a sign-in screen is even the right thing to show. It therefore
carries no secret. In particular it never carries the platform token, and
``management_url``, ``data_plane_url`` and ``docs_url`` are addresses an operator
configured, not credentials.

The contract is shared with otari.ai, which serves the same shape for its hosted
deployment (mozilla-ai/otari-ai#1591). ``deployment_type`` and ``session_type``
name three values each, and this server produces all three of the first and two
of the second: ``hosted_user`` is a session minted by somebody else's account
system, which no build here does. The enum is the contract, not this
deployment's inventory.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.maintenance_mode_service import is_maintenance_mode
from gateway.services.tenancy.user_service import operator_has_password
from gateway.services.tenancy.webauthn_service import has_any_credential

router = APIRouter(prefix="/v1/bootstrap", tags=["bootstrap"])

DeploymentType = Literal["standalone", "hosted", "hybrid"]
SessionType = Literal["local_operator", "hosted_user", "none"]
# How a caller may sign in to this deployment. ``master_key`` is the first-boot
# credential and ``password`` the steady-state one, and a standalone gateway
# offers exactly one of them: the master key until the operator claims the
# deployment with a password, and the password from then on
# (mozilla-ai/otari-ai#1716). A list rather than a single value because #651 and
# #652 add methods that coexist with the password rather than replacing it, and
# because a hybrid gateway offers none. "passkey" is the first of those to land,
# and it is genuinely additive: it appears beside whichever of the two
# credentials is current, never instead of one.
#
# #651's OAuth sign-in is deliberately *not* a fourth value here. A method name
# cannot say which provider, and "oauth" plus a separate list of providers would
# be one fact published twice, so it travels as ``oauth_providers`` below and
# this list stays the set of methods that need no further qualification.
SignInMethod = Literal["master_key", "password", "passkey"]

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
    # The deployment-wide account administration prefix (/v1/admin). The
    # deployment axis only: it says this process hosts the surface, not that the
    # caller may use it, which is `GET /v1/admin/access`'s question and the
    # service's to enforce.
    "admin",
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

# The same list for a *hosted* deployment: one control plane serving many
# organizations, rather than one operator's own gateway. Two rows differ, and
# both differences are the same fact seen from either side, that a credential
# here belongs to a tenant rather than to the process.
#
# ``providers`` drops. It is the deployment-instance surface over
# ``provider_credentials``, whose primary key is the instance name alone, so an
# instance added there is served to every organization and shadows that
# organization's own BYO key for the provider (#818). On the single-tenant
# product that page is correct, and it stays; on a multi-tenant one it is a
# control whose blast radius nobody looking at it can see.
#
# ``organization_providers`` appears, and is the per-tenant surface that
# replaces it: ``/v1/organizations/me/provider-keys``, the organization-scoped
# BYO keys #670 shipped. It is the one name here that is not its router's path
# prefix, because the router is nested under ``organizations``; ``organizations``
# stays a separate surface, since the roster and the credential set are
# different pages with different access.
#
# Dropping the surface is not itself a guard over the table: ``config.yml``'s
# ``providers:`` block and ``/v1/provider-credentials`` still populate it with no
# page in front of them, which is #818's to close.
HOSTED_SURFACES: tuple[str, ...] = (
    *(surface for surface in STANDALONE_SURFACES if surface != "providers"),
    "organization_providers",
)


class DeploymentBootstrap(BaseModel):
    """What the dashboard shell needs before it can render anything."""

    deployment_type: DeploymentType = Field(
        description=(
            "Which deployment serves this URL. 'standalone' owns its own data and serves one "
            "tenant; 'hosted' owns its own data and serves many (otari.ai, or any deployment "
            "run as a control plane), which is why its management surfaces are the "
            "per-organization ones; 'hybrid' is a gateway attached to otari.ai, which is "
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
    data_plane_url: str | None = Field(
        description=(
            "Where this deployment's inference traffic belongs, when it is not served here. "
            "The mirror of management_url: that one says where management lives when this "
            "deployment is not the control plane, this one says where the data plane is when "
            "this deployment is not it. Set only by a hosted control plane, which serves the "
            "dashboard but not inference (otari#822); null for standalone and hybrid, both of "
            "which serve inference at the address that reached this page. Not a human link "
            "target like management_url: it is the gateway's bare address, which the dashboard "
            "suffixes with /v1 to build its request snippets. So it must carry no /v1 path "
            "segment anywhere (a value ending in /v1, or a whole endpoint like "
            "/v1/chat/completions, renders that path twice) and no credential, since this "
            "response is unauthenticated. This gateway refuses both at startup; any deployment "
            "serving this contract should publish the same shape. Null on a hosted "
            "deployment means unconfigured, and the dashboard then shows no snippet rather "
            "than one naming this host."
        )
    )
    docs_url: str | None = Field(
        description=(
            "Where this deployment's documentation lives, when it is not the operator guide "
            "bundled with the gateway. Set, the dashboard's Documentation links open it in a "
            "new tab; null, they go to the bundled guide at /#/docs, which stays served either "
            "way. A link target an operator configured, validated at startup as an absolute "
            "http(s) URL."
        )
    )
    sign_in_methods: list[SignInMethod] = Field(
        description=(
            "How POST /v1/auth/session may be authenticated right now, sorted. 'master_key' is the "
            "first-boot credential and is offered until the operator identity has a password, which "
            "is what claiming the deployment means; 'password' replaces it from then on, and the "
            "master key stays the credential for the management API. 'passkey' appears alongside "
            "either one when this deployment is configured for WebAuthn and holds at least one "
            "passkey that its current relying-party ID can assert. Empty for a hybrid gateway, "
            "which issues no session. The login page renders from this rather than trying a "
            "credential to find out."
        )
    )
    maintenance_mode: bool = Field(
        description=(
            "Whether this deployment is refusing new dashboard sign-ins while an operator "
            "redeploys it. The sign-in screen says so rather than presenting a form whose only "
            "outcome is a 503. Sessions already issued keep working, and the management API and "
            "the data plane are unaffected. False for a hybrid gateway, which issues no session."
        )
    )
    passkeys_ready: bool = Field(
        description=(
            "Whether this deployment can run a passkey ceremony at all: it has a relying-party ID "
            "(webauthn_rp_id, or derived from public_base_url) and an origin to serve one from. "
            "Distinct from 'passkey' in sign_in_methods, which is narrower and answers whether a "
            "registered passkey could sign somebody in *right now*: an operator with none yet needs "
            "this one, or the page that registers the first would be hidden from them. False for a "
            "hybrid gateway, which issues no session of its own."
        )
    )
    oauth_providers: list[str] = Field(
        description=(
            "OAuth providers this deployment can sign somebody in with, sorted, one entry per "
            "provider with a client ID, a client secret and a public_base_url to build a redirect "
            "URI from. The sign-in screen renders a button per entry and none at all when the list "
            "is empty, so a provider nobody configured is absent rather than offered and then "
            "refused. Additive to sign_in_methods rather than part of it: an OAuth sign-in coexists "
            "with whichever typed credential is current, the way a passkey does. Empty for a hybrid "
            "gateway, which issues no session."
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

    The database read is two primary-key lookups: the ``tenancy_bootstrap_user_id``
    marker, and the identity it names, to answer whether *that* identity holds a
    password (#702). It runs only in standalone mode: a hybrid gateway has no session to describe,
    and ``get_db_if_needed`` hands it no session to read one from.
    """
    if config.is_hybrid_mode:
        return DeploymentBootstrap(
            deployment_type="hybrid",
            session_type="none",
            surfaces=[],
            sign_in_methods=[],
            management_url=config.platform_management_url,
            # This gateway *is* the data plane, so the address that reached this
            # page is the address that reaches the API.
            data_plane_url=None,
            docs_url=config.docs_url,
            maintenance_mode=False,
            passkeys_ready=False,
            oauth_providers=[],
            mail_ready=False,
        )
    assert db is not None  # get_db_if_needed yields a session outside hybrid mode
    # Hosted is standalone's multi-tenant sibling and differs here in exactly two
    # fields. Everything below them is answered identically, because a hosted
    # deployment holds its own database, mounts the same management API and mints
    # its own sessions: ``session_type`` stays ``local_operator`` because this
    # build's sign-in is this build's, not an account minted by somebody else's
    # control plane, which is what ``hosted_user`` names.
    hosted = config.is_hosted_mode
    return DeploymentBootstrap(
        deployment_type="hosted" if hosted else "standalone",
        session_type="local_operator",
        surfaces=sorted(HOSTED_SURFACES if hosted else STANDALONE_SURFACES),
        sign_in_methods=await _sign_in_methods(db, config),
        management_url=None,
        # Standalone is its own data plane and answers null; a hosted control
        # plane is not, and publishes wherever its operator says the gateway is.
        data_plane_url=config.data_plane_url if hosted else None,
        docs_url=config.docs_url,
        maintenance_mode=await _maintenance_mode(db),
        passkeys_ready=config.webauthn_enabled,
        oauth_providers=list(config.oauth_providers),
        mail_ready=config.mail_ready,
    )


async def _sign_in_methods(db: AsyncSession, config: GatewayConfig) -> list[SignInMethod]:
    """How this deployment may be signed in to right now, sorted.

    Two independent questions. Which of the two *typed* credentials
    ``POST /v1/auth/session`` accepts is the first, and they are mutually
    exclusive: the master key until the operator identity holds a password
    (otari#702), that password from then on. Whether a passkey can sign somebody in is the second, and it
    is additive, because ``POST /v1/auth/webauthn/authenticate`` is a separate
    endpoint that does not displace either.

    A passkey is published only when one could actually answer: the deployment
    has a relying-party ID *and* holds at least one credential registered under
    it. Advertising the method on a deployment with no passkeys would put a
    button on the login page whose only outcome is the browser reporting that it
    found nothing, which is the same trap the master-key box would be on a
    claimed deployment.

    A database failure answers "none" rather than propagating. This route is the
    first thing the dashboard shell fetches, so a 500 here is a blank page
    instead of a login screen, and it would be a blank page for the one outage
    where an operator most wants the dashboard to say something. "None" is also
    the truth while the database is unreachable: no session can be minted either
    way, since minting one writes a row.
    """
    try:
        claimed = await operator_has_password(db)
        passkeys = await has_any_credential(db, config)
    except SQLAlchemyError:
        logger.warning("Could not read which sign-in methods this deployment offers", exc_info=True)
        return []
    typed: SignInMethod = "password" if claimed else "master_key"
    methods: list[SignInMethod] = [typed]
    if passkeys:
        methods.append("passkey")
    return sorted(methods)


async def _maintenance_mode(db: AsyncSession) -> bool:
    """Whether this deployment is currently refusing new dashboard sign-ins.

    A database failure answers "not frozen", for the same reason ``_sign_in_methods``
    answers "none": this payload must render a page rather than propagate a 500.
    The two degradations agree, because that failure already empties
    ``sign_in_methods``, and the screen that emptiness selects says the gateway
    cannot start a session at all, which is both true and more specific than a
    maintenance notice would be.
    """
    try:
        return await is_maintenance_mode(db)
    except SQLAlchemyError:
        logger.warning("Could not read whether this deployment is in maintenance mode", exc_info=True)
        return False
