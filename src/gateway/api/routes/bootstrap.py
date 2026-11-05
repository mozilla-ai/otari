"""Deployment bootstrap the dashboard shell reads before it renders.

One server-derived payload: which management surfaces this deployment offers
and how a caller may sign in. The shell reads it once and gates navigation on
it.

Unauthenticated by necessity: this is what tells a browser whether a sign-in
screen is even the right thing to show. It therefore carries no secret:
``docs_url``, ``terms_url``, ``privacy_url`` and ``site_url`` are addresses an
operator configured, not credentials.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_enabled_features
from gateway.api.routes import (
    admin,
    agent_recommenders,
    budgets,
    hosted_providers,
    keys,
    models,
    org_provider_keys,
    organization_guardrails,
    organization_usage,
    organizations,
    playground,
    pricing,
    routing,
    settings,
    tools,
    usage,
    users,
    workspaces,
)
from gateway.core.config import GatewayConfig
from gateway.core.database import describe_database_error
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.log_config import logger
from gateway.services.maintenance_mode_service import is_maintenance_mode
from gateway.services.tenancy.user_service import operator_has_password, password_sign_in_possible
from gateway.services.tenancy.webauthn_service import has_any_credential

router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])

# How a caller may sign in to this deployment. ``master_key`` is the first-boot
# credential and ``password`` the steady-state one, and a gateway
# offers exactly one of them: the master key until the operator claims the
# deployment with a password, and the password from then on
# (mozilla-ai/otari-ai#1716). A list rather than a single value because #651 and
# #652 add methods that coexist with the password rather than replacing it.
# "passkey" is the first of those to land,
# and it is genuinely additive: it appears beside whichever of the two
# credentials is current, never instead of one.
#
# #651's OAuth sign-in is deliberately *not* a fourth value here. A method name
# cannot say which provider, and "oauth" plus a separate list of providers would
# be one fact published twice, so it travels as ``oauth_providers`` below and
# this list stays the set of methods that need no further qualification.
SignInMethod = Literal["master_key", "password", "passkey"]

# A route module's ``SURFACE`` is published only once listed here.
_DECLARED_SURFACES: tuple[Surface, ...] = (
    admin.SURFACE,
    agent_recommenders.SURFACE,
    budgets.SURFACE,
    hosted_providers.SURFACE,
    keys.SURFACE,
    models.SURFACE,
    org_provider_keys.SURFACE,
    organization_guardrails.SURFACE,
    organization_usage.SURFACE,
    organizations.SURFACE,
    playground.SURFACE,
    pricing.SURFACE,
    routing.SURFACE,
    settings.SURFACE,
    tools.SURFACE,
    usage.SURFACE,
    users.SURFACE,
    workspaces.SURFACE,
)

SURFACES: tuple[str, ...] = tuple(surface.name for surface in _DECLARED_SURFACES)


def published_surfaces(enabled_features: tuple[CoreFeature, ...]) -> list[str]:
    """The surfaces this deployment publishes, sorted: the fixed set and each enabled feature's surface."""
    featured = [feature.surface for feature in enabled_features if feature.surface is not None]
    surfaces = (*_DECLARED_SURFACES, *featured)
    return sorted({surface.name for surface in surfaces})


class DeploymentBootstrap(BaseModel):
    """What the dashboard shell needs before it can render anything."""

    surfaces: list[str] = Field(
        description=(
            "Management API groups this deployment serves, sorted, which is what its dashboard "
            "pages gate on: a fixed set plus one for each enabled feature. Named surfaces, not "
            "capabilities: capability is otari.ai's word for the entitlement (licensing) axis."
        )
    )
    docs_url: str | None = Field(
        description=(
            "Where this deployment's documentation lives, when it is not docs.otari.ai. Set, "
            "the dashboard's Documentation links open it in a new tab; null, they open "
            "https://docs.otari.ai/. A link target an operator configured, validated at startup as an absolute "
            "http(s) URL carrying no credential, since this response is unauthenticated."
        )
    )
    terms_url: str | None = Field(
        description=(
            "Where this deployment's terms of service live. Set, the account menu carries a "
            "Terms of service row pointing at them; null, no address is configured and the "
            "menu carries no such row. A link target an operator configured, validated at "
            "startup as an absolute http(s) URL carrying no credential, since this response "
            "is unauthenticated."
        )
    )
    privacy_url: str | None = Field(
        description=(
            "Where this deployment's privacy notice lives. Set, the account menu's Data & "
            "Privacy row links to it; null, no address is configured and that row stays "
            "disabled, carrying the standing note that there is nothing to configure there "
            "yet. A link target an operator configured, validated at startup as an absolute "
            "http(s) URL carrying no credential, since this response is unauthenticated."
        )
    )
    site_url: str | None = Field(
        description=(
            "Where this deployment's public website lives. Set, the logo on the pages a visitor "
            "reaches without an account links to it; null, it links to the public catalog where "
            "public_catalog is true, and is not a link otherwise. A link target an operator "
            "configured, validated at startup as an absolute http(s) URL carrying no credential, "
            "since this response is unauthenticated."
        ),
    )
    sign_in_methods: list[SignInMethod] = Field(
        description=(
            "How POST /api/v1/auth/session may be authenticated right now, sorted. 'master_key' is "
            "the first-boot credential and is offered until the operator identity has a password, "
            "which is what claiming the deployment means; past that it stays the credential for "
            "the management API but is no longer a dashboard login. 'password' is offered while "
            "any active identity holds one, which is not the same question and not always the "
            "later half of it: a member can hold a password on a deployment whose operator never "
            "claimed it, so both typed credentials can appear together. 'passkey' appears "
            "alongside either when this deployment is configured for WebAuthn and holds at least "
            "one passkey that its current relying-party ID can assert. The login page renders "
            "from this rather than trying a credential to find out."
        )
    )
    maintenance_mode: bool = Field(
        description=(
            "Whether this deployment is refusing new dashboard sign-ins while an operator "
            "redeploys it. The sign-in screen says so rather than presenting a form whose only "
            "outcome is a 503. Sessions already issued keep working, and the API is unaffected."
        )
    )
    passkeys_enabled: bool = Field(
        description=(
            "Whether this deployment offers passkeys at all (the passkeys_enabled setting). False "
            "means the passkey routes are not mounted and the dashboard hides its passkey page, "
            "which is a different answer from passkeys_ready: a deployment that merely has not "
            "configured a relying party is ready for nobody yet but still shows the page that says "
            "so."
        )
    )
    passkeys_ready: bool = Field(
        description=(
            "Whether this deployment can run a passkey ceremony at all: it has a relying-party ID "
            "(webauthn_rp_id, or derived from public_base_url) and an origin to serve one from. "
            "Distinct from 'passkey' in sign_in_methods, which is narrower and answers whether a "
            "registered passkey could sign somebody in *right now*: an operator with none yet needs "
            "this one, or the page that registers the first would be hidden from them."
        )
    )
    oauth_providers: list[str] = Field(
        description=(
            "OAuth providers this deployment can sign somebody in with, sorted, one entry per "
            "provider with a client ID, a client secret and a public_base_url to build a redirect "
            "URI from. The sign-in screen renders a button per entry and none at all when the list "
            "is empty, so a provider nobody configured is absent rather than offered and then "
            "refused. Additive to sign_in_methods rather than part of it: an OAuth sign-in coexists "
            "with whichever typed credential is current, the way a passkey does."
        )
    )
    public_catalog: bool = Field(
        default=False,
        description=(
            "Whether the model catalog is served to a visitor with no session: GET /api/v1/catalog/models "
            "answers anonymously and the dashboard renders Models ahead of sign-in."
        ),
    )
    open_signup: bool = Field(
        description=(
            "Whether POST /api/v1/auth/signup creates an account for an address nobody has "
            "added yet, each with an organization of its own, or only lets an address an admin "
            "already put on the roster set its password. The signup page reads as registration "
            "or as claiming an invitation accordingly, and the sign-in screen links to it with "
            "the wording that matches."
        )
    )
    feedback_enabled: bool = Field(description="Whether this deployment accepts deliberate feedback submissions.")
    federation_member_rules: bool = Field(
        description=(
            "Whether an organization's owners and admins may give their own API keys federation rules (the "
            "federation_member_rules setting), so the dashboard offers workload identity on their keys."
        )
    )
    mail_ready: bool = Field(
        description=(
            "Whether this deployment can deliver a message carrying a link back to itself "
            "(an invitation's accept link, and the verification and reset links to come), "
            "not merely whether a transport is configured: it also needs to know its own "
            "public URL to put in one. Lets the dashboard disable or hide a mail-dependent "
            "affordance instead of offering one that would fail at send time. Every "
            "message this deployment sends carries such a link, which is why this is "
            "one flag and not one per feature."
        )
    )


@router.get("", response_model=DeploymentBootstrap)
async def get_bootstrap(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    enabled_features: Annotated[tuple[CoreFeature, ...], Depends(get_enabled_features)],
) -> DeploymentBootstrap:
    """Return the deployment context the dashboard shell renders from.

    Public: the shell fetches this before it knows whether it can authenticate.
    That is also why ``sign_in_methods`` is answered here rather than behind a
    credential, and it publishes nothing an unauthenticated caller could not
    already learn by trying both credentials against the sign-in endpoint.

    The database read is two primary-key lookups: the ``tenancy_bootstrap_user_id``
    marker, and the identity it names, to answer whether *that* identity holds a
    password (#702).
    """
    return DeploymentBootstrap(
        surfaces=published_surfaces(enabled_features),
        sign_in_methods=await _sign_in_methods(db, config),
        docs_url=config.docs_url,
        terms_url=config.terms_url,
        privacy_url=config.privacy_url,
        site_url=config.site_url,
        maintenance_mode=await _maintenance_mode(db),
        public_catalog=bool(config.public_catalog),
        passkeys_enabled=config.passkeys_enabled,
        passkeys_ready=config.webauthn_enabled,
        oauth_providers=list(config.oauth_providers),
        feedback_enabled=any(feature.name == "feedback" for feature in enabled_features),
        federation_member_rules=config.federation_member_rules,
        mail_ready=config.mail_ready,
        # Gated on mail as well as on the setting, and not only because the
        # route refuses without it: an operator who turned open signup on and
        # has no transport would otherwise get a page inviting registrations
        # that every visitor's submit answers with a 503.
        open_signup=config.open_signup and config.mail_ready,
    )


async def _sign_in_methods(db: AsyncSession, config: GatewayConfig) -> list[SignInMethod]:
    """How this deployment may be signed in to right now, sorted.

    Three independent questions, and every method here is published only when it
    could actually answer.

    ``master_key`` is the first-boot credential, and it is offered until the
    operator identity holds a password (otari#702), which is what claiming the
    deployment means. Past that point ``POST /api/v1/auth/session`` refuses it
    with a 403, so offering it would be offering a refusal.

    ``password`` is offered while some active identity holds one. Not the
    operator's alone (otari-ai#2100): that endpoint verifies a password against
    whichever identity the address resolves to and has never consulted the
    operator's row, so a member who signed up on a deployment its operator never
    claimed can sign in, and used to be shown the master-key box instead of the
    form that would have worked. The two typed credentials are therefore
    additive rather than exclusive, and an unclaimed deployment with members on
    it publishes both.

    ``passkey`` is the third, additive for a different reason:
    ``POST /api/v1/auth/webauthn/authenticate`` is a separate endpoint that
    displaces neither. It needs the deployment to have a relying-party ID *and*
    to hold at least one credential registered under it, or the button's only
    outcome is the browser reporting that it found nothing.

    A database failure answers "none" rather than propagating. This route is the
    first thing the dashboard shell fetches, so a 500 here is a blank page
    instead of a login screen, and it would be a blank page for the one outage
    where an operator most wants the dashboard to say something. "None" is also
    the truth while the database is unreachable: no session can be minted either
    way, since minting one writes a row.
    """
    try:
        claimed = await operator_has_password(db)
        passwords = await password_sign_in_possible(db)
        passkeys = await has_any_credential(db, config)
    except SQLAlchemyError as exc:
        logger.warning("Could not read which sign-in methods this deployment offers: %s", describe_database_error(exc))
        return []
    methods: list[SignInMethod] = []
    if not claimed:
        methods.append("master_key")
    if passwords:
        methods.append("password")
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
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not read whether this deployment is in maintenance mode: %s", describe_database_error(exc)
        )
        return False
