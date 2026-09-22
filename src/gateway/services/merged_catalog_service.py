"""The merged catalog: every selector one caller may be shown, before presentation.

Discovery, stored prices, the genai-prices defaults, aliases and routing
policies folded into one view, scoped to the caller who asked. Two surfaces
read it and must not disagree about which selectors exist or which of them a
caller may see: the flat, OpenAI-shaped listing, and the catalog grouped by
model.

The wire shapes live here with the build because both surfaces serve them.
"""

import calendar
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.money import as_float
from gateway.models.pricing import ModelPricing, PriceSource
from gateway.models.pricing_schemas import PricingTier
from gateway.models.routing import PolicySpec
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.services.alias_service import effective_aliases
from gateway.services.model_access import is_model_allowed, resolve_request_allowlist
from gateway.services.model_discovery_service import background_discovery_enabled, discover_all_models
from gateway.services.policy_store import effective_policies
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    OverridePeriod,
    default_model_pricing,
    default_pricing_enabled,
    default_pricing_reference,
    model_context_window,
    normalize_effective_at,
    pricing_key_forms,
    resolve_organization_override,
)
from gateway.services.provider_kwargs import is_deployment_instance_key, normalize_pricing_key, split_selector
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.organization_model_access import (
    resolve_default_workspace_offered_keys,
    resolve_organization_offered_keys,
    resolve_session_catalog_scope,
    resolve_workspace_offered_keys,
)

if TYPE_CHECKING:
    from any_llm.types.model import Model


# ``owned_by`` reported for alias entries. Aliases intentionally hide the
# underlying provider, so the gateway itself is named as the owner rather than
# the real upstream.
ALIAS_OWNED_BY = "otari"


class ModelPricingInfo(BaseModel):
    """Pricing information for a model."""

    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    cache_write_1h_price_per_million: float | None = None
    # The same tiers SetPricingRequest accepts, so the response says what the
    # request already promised. The permissive arm stays for the same reason as
    # the billing shapes on a usage row, with one addition specific to here:
    # dropping it would make Pydantic validate every stored tier against
    # PricingTier's rules on read, which is validation this field never used to
    # do, so a rule tightened later would turn old rows into a 500 on the listing.
    #
    # Deliberately left on pydantic's smart union, unlike the charge lines in
    # _billing_schemas, which pin `union_mode="left_to_right"`. Smart mode takes
    # the dict arm for a stored tier, which is what keeps the wire format clean.
    # Do not "make it consistent" by adding left_to_right here: PricingTier is
    # not one of the defaulted shapes, so it would write four null price keys per
    # tier, and it would start rejecting the catalog-derived tiers that
    # pricing_service._pricing_tiers emits (a `min_input_tokens` of 0 fails
    # `gt=0`, and a tier with no rate override fails validate_has_rate_override).
    pricing_tiers: Sequence[PricingTier | dict[str, float | int]] = Field(default_factory=list)
    # What the rates are per (``PRICING_UNITS``). A gateway-run tool's row is per
    # million requests, and a reader that assumed tokens would be wrong by the
    # whole unit; the catalog never lists one, but the field travels with the
    # price so a caller reading one model is told too.
    unit: str = "tokens"


def pricing_info(pricing: ModelPricing) -> ModelPricingInfo:
    """The wire shape of one stored, default, or override price row."""
    return ModelPricingInfo(
        input_price_per_million=float(pricing.input_price_per_million),
        output_price_per_million=float(pricing.output_price_per_million),
        cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
        cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
        cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
        pricing_tiers=pricing.pricing_tiers or [],
        # A transient row (a default, an override) is built without the column
        # default that an insert would apply, so it reads back None here.
        unit=pricing.unit or "tokens",
    )


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    pricing: ModelPricingInfo | None = None
    # Where ``pricing`` came from: "configured" (DB), "default" (genai-prices
    # fallback, only when default_pricing is enabled), or "none".
    pricing_source: str = "none"
    # Context-window token limit from the bundled genai-prices dataset, when it
    # knows the model. Metadata only (independent of the default_pricing toggle);
    # ``None`` when the dataset has no value for the model.
    context_window: int | None = None
    # True when the deployment pays the upstream bill for this model, so the organization may not set its own rate.
    # An alias or policy is a name, not a model, so it is always False.
    deployment_managed: bool = False


def mark_deployment_managed(
    config: GatewayConfig, model: ModelObject, *, deployment_supplied_providers: frozenset[str]
) -> ModelObject:
    """Sets ``deployment_managed`` on ``model`` and returns it."""
    split = split_selector(model.id)
    model.deployment_managed = is_deployment_instance_key(config, model.id) or (
        split is not None and split[0] in deployment_supplied_providers
    )
    return model


def owner_from_key(model_key: str) -> str:
    """The provider a ``provider:model`` key names, or "unknown" for a bare name."""
    provider, separator, _ = model_key.partition(":")
    return provider if separator else "unknown"


def created_timestamp(model: "Model") -> int:
    """A discovered model's creation timestamp, or 0 when the provider omits one.

    ``Model.created`` is typed ``int`` but the OpenAI SDK builds response models
    without validation, so an OpenAI-compatible provider that reports ``null``
    (or a non-numeric value) reaches us as-is. Falling back to 0 matches what the
    other catalog phases publish for models with no known creation time; raising
    here would fail the whole listing over one provider's payload.
    """
    try:
        return int(model.created)
    except (TypeError, ValueError):
        return 0


def context_window_for_key(model_key: str) -> int | None:
    """genai-prices context-window for a ``provider:model`` (or bare) key.

    Metadata, so it is filled whether or not the default-pricing fallback is on;
    ``None`` when the dataset does not know the model or lists no window for it.
    """
    provider_part, separator, model_part = model_key.partition(":")
    provider = provider_part if separator else None
    model_name = model_part if separator else model_key
    return model_context_window(provider, model_name)


def model_from_pricing(pricing: ModelPricing) -> ModelObject:
    """Convert a ModelPricing row to an OpenAI-compatible ModelObject."""
    created = int(calendar.timegm(pricing.created_at.utctimetuple()))
    return ModelObject(
        id=pricing.model_key,
        created=created,
        owned_by=owner_from_key(pricing.model_key),
        pricing=pricing_info(pricing),
        pricing_source="configured",
        context_window=context_window_for_key(pricing.model_key),
    )


def alias_model(config: GatewayConfig, alias: str, target: str, pricing_lookup: dict[str, ModelPricing]) -> ModelObject:
    """Build a ModelObject for an alias, from config.yml or from storage.

    The alias id is what the caller sees; pricing is looked up from the resolved
    target's canonical key so an alias shows the real model's price without
    revealing the provider/model behind it. ``pricing_source`` describes where
    that price came from, just as for a real model; the alias itself is
    identified by ``owned_by``.
    """
    canonical_target = normalize_pricing_key(config, target)
    pricing = pricing_lookup.get(canonical_target)
    obj = ModelObject(
        id=alias,
        created=0,
        owned_by=ALIAS_OWNED_BY,
        pricing=pricing_info(pricing) if pricing else None,
        pricing_source="configured" if pricing else "none",
        # From the resolved target, like pricing: an alias's display name is not a
        # model the dataset knows. Exposing the window does not reveal the target.
        context_window=context_window_for_key(canonical_target),
    )
    # Priced from the target, never from the alias's display name: the fallback
    # keys on a real provider/model, and the display name is neither. Without
    # this an alias to an unpriced model would report no price while the gateway
    # billed it at the default rate.
    apply_default_pricing(obj, pricing_selector=canonical_target)
    return obj


def alias_target_keys(config: GatewayConfig, aliases: dict[str, str]) -> set[str]:
    """Canonical pricing keys of every alias target."""
    return {normalize_pricing_key(config, target) for target in aliases.values()}


def dynamic_policy_model(policy_name: str) -> ModelObject:
    """Build a ModelObject for a policy whose candidate depends on the request.

    A router-driven or condition-driven policy has no single target, so it has no
    single price either. Reporting the default candidate's price would be a guess
    that happens to be wrong whenever the policy does its job. ``pricing_source``
    already exists for exactly this kind of statement, so it carries ``"dynamic"``
    rather than inventing a second field; ``pricing`` stays null, and a client that
    does not know the new value sees "unpriced" instead of a fabricated rate.
    """
    return ModelObject(
        id=policy_name,
        created=0,
        owned_by=ALIAS_OWNED_BY,
        pricing=None,
        pricing_source="dynamic",
    )


def catalog_aliases(
    config: GatewayConfig,
    *,
    caller_user_id: str | None,
    caller_workspace_id: uuid.UUID | None,
    workspace_layer: bool,
) -> dict[str, str]:
    """The aliases to list, with or without the workspace-scoped rows.

    ``workspace_layer`` is false only for a session that may not see the
    workspace ``effective_aliases`` would read (see
    ``services/tenancy/organization_model_access``), where the configured aliases
    are the whole answer: an alias name is not filtered by the model allow-list,
    because the target it resolves to can be one every tenant may reach.
    """
    if not workspace_layer:
        return dict(config.aliases)
    return effective_aliases(config, caller_user_id, workspace_id=caller_workspace_id)


def policy_catalog_entries(
    config: GatewayConfig,
    caller_user_id: str | None,
    caller_workspace_id: uuid.UUID | None,
    *,
    workspace_layer: bool = True,
) -> tuple[dict[str, str], dict[str, PolicySpec]]:
    """Split the policies in force into ``{name: target}`` and dynamic ones.

    Reads through :func:`effective_policies`, so stored policies are listed
    alongside the ones from ``config.yml`` and a caller's user-scoped policy wins,
    exactly as it does at request time. Listing only the configured ones would mean
    a policy created in the dashboard worked but was invisible in the catalog. The
    exception is ``workspace_layer=False``, which is :func:`catalog_aliases`'s
    and for the same reason: a stored policy is a workspace's row too.

    A static policy is an alias in every way the catalog cares about (one name, one
    target, priced from the target), so it is folded into the alias map and listed
    that way. Its target stays in the listing though, unlike an alias's: see the
    note on the listing. A dynamic one is listed separately by
    :func:`dynamic_policy_model`.
    """
    if not workspace_layer:
        policies = dict(config.routing.policies) if config.routing.enabled else {}
    else:
        policies = effective_policies(config, caller_user_id, workspace_id=caller_workspace_id)
    static: dict[str, str] = {}
    dynamic: dict[str, PolicySpec] = {}
    for name, spec in policies.items():
        if spec.is_dynamic:
            dynamic[name] = spec
        else:
            static[name] = spec.default_target
    return static, dynamic


def pricing_key_candidates(config: GatewayConfig, target: str) -> list[str]:
    """Stored key forms that could hold pricing for ``target``.

    Keys are canonicalized on write, but rows predating that may still use the
    legacy ``provider/model`` separator, so both forms are offered.
    """
    canonical = normalize_pricing_key(config, target)
    return sorted({target, canonical, canonical.replace(":", "/", 1)})


def normalized_pricing_lookup(config: GatewayConfig, pricing_map: dict[str, ModelPricing]) -> dict[str, ModelPricing]:
    """Re-key a pricing map by canonical model key, matching how targets resolve."""
    return {normalize_pricing_key(config, key): row for key, row in pricing_map.items()}


def apply_default_pricing(obj: ModelObject, pricing_selector: str | None = None) -> None:
    """Fill the genai-prices default rate for a model that has no DB price.

    No-op when the fallback is disabled or the model already carries a price, so
    database pricing always takes precedence. Marks the source as "default".

    ``pricing_selector`` names the model to price when that differs from
    ``obj.id`` (an alias is priced from its target); it defaults to ``obj.id``.
    """
    if obj.pricing is not None or not default_pricing_enabled():
        return
    selector = pricing_selector if pricing_selector is not None else obj.id
    provider_part, separator, model_part = selector.partition(":")
    provider = provider_part if separator else None
    model_name = model_part if separator else selector
    default = default_model_pricing(provider, model_name, normalize_effective_at(None))
    if default is not None:
        obj.pricing = pricing_info(default)
        obj.pricing_source = "default"


async def get_pricing_map(
    db: AsyncSession,
    provider_filter: str | None = None,
    model_keys: Sequence[str] | None = None,
) -> dict[str, ModelPricing]:
    """Load latest pricing per model_key, optionally filtered by provider prefix or key set."""
    latest_effective = (
        select(
            ModelPricing.model_key.label("model_key"),
            func.max(ModelPricing.effective_at).label("effective_at"),
        )
        .group_by(ModelPricing.model_key)
        .subquery()
    )

    stmt = select(ModelPricing).join(
        latest_effective,
        (ModelPricing.model_key == latest_effective.c.model_key)
        & (ModelPricing.effective_at == latest_effective.c.effective_at),
    )

    if provider_filter:
        stmt = stmt.where(ModelPricing.model_key.startswith(f"{provider_filter}:"))

    if model_keys is not None:
        stmt = stmt.where(ModelPricing.model_key.in_(model_keys))

    stmt = stmt.order_by(ModelPricing.model_key)
    result = await db.execute(stmt)
    pricings = result.scalars().all()
    return {p.model_key: p for p in pricings}


@dataclass(frozen=True)
class CatalogScope:
    """What one caller may be shown of the catalog."""

    allowlist: list[str] | None
    """``None`` is unrestricted."""

    reads_workspace_layer: bool
    """Whether the workspace-scoped alias and policy rows may be read at all.

    False only for a dashboard session that may not see the workspace those rows
    would come from. Every other caller keeps the layer it always read: an API
    key names its own workspace, and a master key reads the deployment's default,
    which is where its own writes land.
    """

    deployment_supplied_providers: frozenset[str]
    """Hosted providers the deployment pays for in at least one of the organization's workspaces."""

    offered_keys: frozenset[str]
    """Selectors the caller's organization offers on its own provider keys.

    Listed by nothing else. Discovery (phase 1) dials ``config.providers``
    instances, and the pricing-only pass (phase 2) lists keys the *deployment*
    price list names, so a model an organization adopted on its own credential is
    permitted by the allow-list and published by neither. See phase 2b.
    """


async def _operator_offered_keys(db: AsyncSession, identity: TenancyUser) -> frozenset[str]:
    """The offered models of the organization a deployment operator is acting in.

    ``active_organization_id`` is the pointer the rest of the tenancy surface
    reads, and an operator who points at nothing has no organization's models to
    be shown.
    """
    if identity.active_organization_id is None:
        return frozenset()
    return await resolve_organization_offered_keys(db, identity.active_organization_id)


async def catalog_scope(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    auth: tuple[APIKey | None, bool],
    session_identity: TenancyUser | None,
    anonymous: bool = False,
    model_provider: ModelProviderPort | None,
    include_offered: bool = True,
) -> CatalogScope:
    """Returns what this caller may be shown, by how they authenticated.

    An API key gets its stored allow-list.
    A master key, and a session that operates the deployment, are unrestricted.
    Any other session gets what its organization can reach,
    and the workspace-scoped rows only for workspaces it may see.
    A visitor to the public catalog gets the configured instances alone.
    ``include_offered=False`` leaves out the organization's offered models, for
    the deployment-wide view the selector index is built from, which reads every
    organization's offerings separately.
    """
    # A visitor, while the catalog is public: the deployment's configured
    # instances and nothing that belongs to a tenant. Not a member of anything,
    # so no BYO key, no workspace's aliases or policies.
    if anonymous:
        return CatalogScope(
            allowlist=[f"{instance}:*" for instance in config.providers],
            reads_workspace_layer=False,
            deployment_supplied_providers=frozenset(),
            offered_keys=frozenset(),
        )
    if session_identity is not None:
        if await DeploymentUserService(db).has_administration_access(session_identity):
            # Unrestricted, and still carrying its *own* organization's offered
            # models. Not every organization's, which would cross the tenant line
            # the rest of this function draws, and not none: on a standalone
            # deployment the operator is also the single organization's owner, so
            # an empty set would hide the model they just adopted on Providers
            # from the catalog they were told it joined.
            return CatalogScope(
                allowlist=None,
                reads_workspace_layer=True,
                deployment_supplied_providers=frozenset(),
                offered_keys=await _operator_offered_keys(db, session_identity) if include_offered else frozenset(),
            )
        scope = await resolve_session_catalog_scope(db, config, user=session_identity, model_provider=model_provider)
        return CatalogScope(
            allowlist=scope.allowlist,
            reads_workspace_layer=scope.reads_default_workspace,
            deployment_supplied_providers=scope.deployment_supplied_providers,
            offered_keys=scope.offered_keys if include_offered else frozenset(),
        )
    api_key, is_master_key = auth
    # An API key's hosted models stay unflagged, because this does not resolve the key's organization.
    return CatalogScope(
        allowlist=None if is_master_key else await resolve_request_allowlist(db, api_key),
        reads_workspace_layer=True,
        deployment_supplied_providers=frozenset(),
        # A master key is the deployment acting on its own behalf, which lands in
        # the default workspace exactly as pricing resolution does for it; a real
        # key names its own workspace, and what that workspace's organization
        # offers is what it may be shown.
        offered_keys=(
            frozenset()
            if not include_offered
            else await resolve_default_workspace_offered_keys(db)
            if is_master_key
            else await resolve_workspace_offered_keys(db, api_key.workspace_id if api_key else None)
        ),
    )


@dataclass
class MergedCatalog:
    """Every selector one caller may be shown, before any presentation.

    Shared by the flat listing and the catalog grouped by model, so the two
    cannot disagree about which selectors exist or which of them the caller may
    see.
    """

    models: dict[str, ModelObject]
    aliases: dict[str, str]
    dynamic_policies: dict[str, PolicySpec]
    discovered_keys: set[str]
    """The selectors phase 1 heard from a provider, as opposed to only priced."""


async def build_merged_catalog(
    db: AsyncSession,
    config: GatewayConfig,
    *,
    auth: tuple[APIKey | None, bool],
    session_identity: TenancyUser | None,
    provider: str | None = None,
    anonymous: bool = False,
    cached_only: bool = False,
    model_provider: ModelProviderPort | None,
    include_offered: bool = True,
) -> MergedCatalog:
    """Merge discovery, stored prices, defaults, aliases and policies for one caller.

    ``anonymous`` is the public catalog's visitor, who is answered from the
    configured instances alone; see :func:`catalog_scope`.

    ``cached_only`` builds the view without dialing any provider, for a caller
    that runs off the request path; see :func:`discover_all_models`.

    Passing ``None`` as ``model_provider`` lists no hosted providers, and
    ``include_offered=False`` leaves out the caller's organization's offered
    models; both are what the selector index's deployment-wide build wants.
    """
    # Aliases are scoped, so the catalog is too: a caller sees their workspace's
    # aliases and the configured ones, plus their own user-scoped layer, never
    # another user's and never another workspace's. A master-key caller has no key
    # to read either off, so it sees the configured layer plus the default
    # workspace's, which is where its own writes land.
    caller_user_id = auth[0].user_id if auth[0] is not None else None
    caller_workspace_id = auth[0].workspace_id if auth[0] is not None else None
    # Resolved before the alias and policy layers are read, not only before they
    # are filtered: it decides whether the workspace-scoped rows may be read at
    # all, which no filter over targets can decide afterwards.
    scope = await catalog_scope(
        db,
        config,
        auth=auth,
        session_identity=session_identity,
        anonymous=anonymous,
        model_provider=model_provider,
        include_offered=include_offered,
    )
    pricing_map = await get_pricing_map(db, provider_filter=provider)
    # Snapshot before phase 1 mutates ``pricing_map`` (it pops matched keys), so
    # alias pricing can still be looked up by the target's canonical key. Keys are
    # canonicalized because an alias target is always canonical while a stored row
    # may use the legacy "provider/model" form; without this a legacy-form row
    # would be withheld from the listing by phase 2 yet never match its alias, so
    # its price would show nowhere.
    pricing_lookup = normalized_pricing_lookup(config, pricing_map)
    # Read once: phase 2 withholds these targets and phase 3 lists the names, and
    # the two must agree even if a write lands between them.
    # A static policy is an alias for catalog purposes (one name, one target,
    # priced from the target), so it joins the alias map and is listed the same
    # way. Startup validation refuses a policy that collides with an alias name,
    # so this merge cannot silently shadow one.
    configured_aliases = catalog_aliases(
        config,
        caller_user_id=caller_user_id,
        caller_workspace_id=caller_workspace_id,
        workspace_layer=scope.reads_workspace_layer,
    )
    static_policies, dynamic_policies = policy_catalog_entries(
        config,
        caller_user_id,
        caller_workspace_id,
        workspace_layer=scope.reads_workspace_layer,
    )
    aliases = {**configured_aliases, **static_policies}
    # Alias targets are withheld from every phase that could surface the real
    # model, discovery (phase 1) as well as pricing-only (phase 2): publishing the
    # target under either would expose the provider:model name the alias exists to
    # hide. Computed before phase 1 so discovery honors it too.
    #
    # A *policy's* candidates are not withheld, which is why only the alias map
    # feeds this. The two indirections look alike but exist for different reasons:
    # an alias is a naming device whose whole purpose is to stand in for a target,
    # while a policy decides where traffic goes among models the caller may name
    # directly anyway. Hiding its candidates cost more than it bought: one policy
    # naming a fallback chain could empty most of the catalog, a model priced by
    # the genai-prices default then vanished from the dashboard along with its
    # rate, and the single-model read served the same model with its price all
    # along, so nothing was actually kept off the wire.
    alias_targets = alias_target_keys(config, configured_aliases)

    merged: dict[str, ModelObject] = {}
    discovered_keys: set[str] = set()

    # Phase 1: auto-discovered models from upstream providers.
    if config.model_discovery:
        try:
            # Cache-only when a refresher owns the dialing. The listing is
            # reachable with any API key, so it deliberately has no ``refresh``
            # escape hatch: forcing a fanout across every configured provider is
            # an operator action, and lives on the operator-gated discovery and
            # provider-health reads instead.
            discovered = await discover_all_models(
                config,
                provider_filter=provider,
                serve_stale=background_discovery_enabled(config),
                cached_only=cached_only,
            )
        except Exception:
            logger.exception("Model discovery failed unexpectedly")
            discovered = []

        for provider_name, model in discovered:
            model_key = f"{provider_name}:{model.id}"
            if normalize_pricing_key(config, model_key) in alias_targets:
                continue
            discovered_keys.add(model_key)
            pricing = pricing_map.pop(model_key, None)
            merged[model_key] = ModelObject(
                id=model_key,
                created=created_timestamp(model),
                owned_by=provider_name,
                pricing=pricing_info(pricing) if pricing else None,
                pricing_source="configured" if pricing else "none",
                context_window=context_window_for_key(model_key),
            )

    # Phase 2: pricing-only models (not discovered but have pricing entries).
    # An alias target is skipped: billing keys on the real model, so aliasing one
    # forces a pricing entry for it, and publishing that entry here would expose
    # the very name the alias exists to hide. Whether real models are listed is
    # governed by ``model_discovery`` (phase 1), never by pricing config.
    for model_key, pricing in pricing_map.items():
        if model_key in merged or normalize_pricing_key(config, model_key) in alias_targets:
            continue
        # A gateway-run tool is priced under the reserved ``otari:`` provider (see
        # ``gateway_tool_pricing_key``). It is not a model: publishing it would put a
        # selectable entry in the OpenAI-compatible catalog whose per-request rate
        # reads as a per-million-token price, and calling it would fail.
        if model_key.startswith(f"{GATEWAY_TOOL_PRICING_PROVIDER}:"):
            continue
        merged[model_key] = model_from_pricing(pricing)

    # Phase 2b: models the caller's organization offers on its own provider keys.
    #
    # Neither phase above can list these. A BYO key is not a discovery source
    # (phase 1 dials ``config.providers`` instances only), and an organization's
    # own rate lives in ``organization_model_pricing`` rather than in the
    # deployment price list phase 2 reads. Without this phase an organization can
    # adopt a model, price it and serve it while the catalog says it does not
    # exist.
    #
    # Priced by phase 3 and then by the per-viewer pass, which reads the
    # organization's own rate, so nothing here carries a price of its own: doing
    # so would state a rate from the wrong rung.
    #
    # ``?provider=`` filters these like any real model, and the allow-list filter
    # at the end still applies: an offered model the caller's key may not use is
    # withheld exactly as a discovered one would be.
    for model_key in sorted(scope.offered_keys):
        if model_key in merged or normalize_pricing_key(config, model_key) in alias_targets:
            continue
        owner = owner_from_key(model_key)
        if provider is not None and owner != provider:
            continue
        merged[model_key] = ModelObject(
            id=model_key,
            created=0,
            owned_by=owner,
            pricing=None,
            pricing_source="none",
            context_window=context_window_for_key(model_key),
        )

    # Phase 3: fill the genai-prices default for unpriced models, so the catalog
    # shows the effective rate when the fallback is active. Database pricing
    # (phases 1-2) always wins; this only touches models still without a price.
    # Runs before aliases are added: this fills from ``id``, and an alias's id is
    # a display name the fallback must never be asked to price. Aliases fill
    # their own default from the resolved target in phase 4.
    for obj in merged.values():
        apply_default_pricing(obj)

    # Phase 4: aliases, from config.yml and from storage alike. An alias is a
    # display name, not a provider, so it is only listed for the unfiltered
    # listing; a ``?provider=`` filter asks for one provider's real models and
    # must not leak the alias mapping.
    if provider is None:
        for alias, target in aliases.items():
            merged[alias] = alias_model(config, alias, target, pricing_lookup)
        # Phase 5: policies whose candidate is decided per request. Listed last so
        # nothing else can price them, and excluded from a ``?provider=`` filter
        # for the same reason aliases are.
        for policy_name in dynamic_policies:
            merged[policy_name] = dynamic_policy_model(policy_name)

    # Model access control: hide models the calling key may not use, so the
    # catalog never advertises a model that would 403 at inference. Both surfaces
    # feed the SAME matcher the SAME canonical instance:model key; an alias id is a
    # display name, so it is matched on its resolved target. Master key sees all.
    key_allowlist = scope.allowlist
    if key_allowlist is not None:
        # A dynamic policy is listed when the key may use *any* of its candidates,
        # which is what the compiler will do at request time: it drops the ones the
        # key cannot use and serves from the rest. Hiding it unless every candidate
        # were permitted would withhold a policy the caller can in fact call.
        dynamic_reachable = {
            name: [normalize_pricing_key(config, selector) for selector in spec.static_selectors()]
            for name, spec in dynamic_policies.items()
        }

        def _permitted(model_id: str) -> bool:
            candidates = dynamic_reachable.get(model_id)
            if candidates is not None:
                return any(is_model_allowed(key_allowlist, candidate) for candidate in candidates)
            target = aliases[model_id] if model_id in aliases else model_id
            return is_model_allowed(key_allowlist, normalize_pricing_key(config, target))

        merged = {mid: obj for mid, obj in merged.items() if _permitted(mid)}

    for obj in merged.values():
        mark_deployment_managed(config, obj, deployment_supplied_providers=scope.deployment_supplied_providers)

    return MergedCatalog(
        models=merged,
        aliases=aliases,
        dynamic_policies=dynamic_policies,
        discovered_keys=discovered_keys,
    )


class ViewerPrice(NamedTuple):
    """The rate one viewer is charged for an offering, and which rung it came from."""

    pricing: ModelPricingInfo | None
    source: PriceSource | None
    reference: str | None
    """What the rate can be traced to: the selector, or the dataset entry behind a default."""


def viewer_price(
    obj: ModelObject,
    overrides: dict[str, list[OverridePeriod]],
    *,
    instance: str,
    model_id: str,
    as_of: datetime,
) -> ViewerPrice:
    """Price one offering for the organization whose overrides are loaded.

    The order is settlement's, and is stated once in
    :func:`pricing_service.find_model_pricing`: the organization's own override,
    then the deployment's stored row, then the genai-prices default. The merged
    catalog has already folded the last two into ``obj`` (phases 2 and 3), so
    what is left here is the override, resolved against the same key forms
    settlement offers, and naming which rung answered.
    """
    override = resolve_organization_override(overrides, pricing_key_forms(obj.id), as_of)
    if override is not None:
        return ViewerPrice(pricing_info(override), "organization", obj.id)
    if obj.pricing is not None and obj.pricing_source == "configured":
        return ViewerPrice(obj.pricing, "deployment", obj.id)
    if obj.pricing is not None and obj.pricing_source == "default":
        return ViewerPrice(obj.pricing, "defaults", default_pricing_reference(instance, model_id, as_of))
    return ViewerPrice(None, None, None)
