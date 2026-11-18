"""The regions a hosted provider serves from: added, changed, reordered and removed.

A provider with no regions serves from its own connection. Its first region
keeps that connection as a region of its own, ahead of the new one, so adding
a region never changes what a request already served is served on. Every field
a region leaves unset is the provider's (``HostedProviderRegion``).
"""

from collections.abc import Awaitable, Callable, Sequence

from gateway.core.config import GatewayConfig, hosted_key_optional, region_instance, region_slug_problem
from gateway.core.settings.providers import https_or_loopback_problem
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    HostedProviderApiKeyMissingError,
    HostedProviderClientArgsUnreadableError,
    HostedProviderInvalidRegionSlugError,
    HostedProviderKeyRequiredError,
    HostedProviderLastRegionError,
    HostedProviderNotFoundError,
    HostedProviderRegionAlreadyExistsError,
    HostedProviderRegionKeyRequiredError,
    HostedProviderRegionNotFoundError,
    HostedProviderUnsafeApiBaseError,
)
from gateway.log_config import logger
from gateway.models.providers import HostedProvider, HostedProviderRegion
from gateway.models.secret_fields import carries_redaction, redact_secret_like_values, restore_redacted_values
from gateway.repositories.providers import HostedProviderRegionRepository, HostedRegionConflict
from gateway.schemas.providers import (
    HostedProviderRegionCreateRequest,
    HostedProviderRegionPublic,
    HostedProviderRegionUpdateRequest,
)
from gateway.services.model_discovery_service import get_model_cache
from gateway.services.providers._hosted_credentials import (
    client_args_of,
    encrypt_client_args,
    encrypt_key,
    last4,
    normalize_api_base,
    origin_of,
    public_api_base,
    region_decryptable,
    unmask_api_base,
)
from gateway.services.providers._hosted_name_index import refresh_hosted_name_index
from gateway.services.providers._hosted_sources import HostedSource

PRIMARY_REGION_SLUG = "primary"


def region_public(provider: str, region: HostedProviderRegion) -> HostedProviderRegionPublic:
    """One region for the API: no key, and its own extras with credential-shaped values masked."""
    return HostedProviderRegionPublic(
        slug=region.slug,
        selector=region_instance(provider, region.slug),
        label=region.label,
        position=region.position,
        enabled=region.enabled,
        data_location=region.data_location,
        has_api_key=region.encrypted_api_key is not None,
        api_key_last4=region.api_key_last4,
        api_base=public_api_base(region.api_base),
        client_args=redact_secret_like_values(client_args_of(region)),
        supported_countries=region.supported_countries,
        decryptable=region_decryptable(region),
        created_at=region.created_at,
        updated_at=region.updated_at,
    )


def _slug(slug: str) -> str:
    """``slug`` as stored.

    Raises:
        HostedProviderInvalidRegionSlugError: a selector could not pin it.
    """
    if (problem := region_slug_problem(slug)) is not None:
        raise HostedProviderInvalidRegionSlugError(slug, problem)
    return slug


def _primary_slug(row: HostedProvider) -> str:
    """The slug the provider's own connection takes as a region: its data location, else ``primary``."""
    if row.data_location is not None and region_slug_problem(row.data_location.lower()) is None:
        return row.data_location.lower()
    return PRIMARY_REGION_SLUG


def _safe_origin(api_base: str | None) -> object:
    """``origin_of`` that reads a stored endpoint it cannot parse as an origin of its own, rather than failing."""
    try:
        return origin_of(api_base)
    except ValueError:
        return ("unparsed", api_base)


def _ensure_servable(row: HostedProvider, *, encrypted_api_key: str | None, api_base: str | None) -> None:
    """Refuse a region the provider's key would follow to another endpoint, or one left with no key it needs.

    Raises:
        HostedProviderKeyRequiredError: the region points the provider's key at another origin.
        HostedProviderApiKeyMissingError: neither the region nor the provider holds a key the endpoint needs.
        HostedProviderUnsafeApiBaseError: the region's own endpoint would carry a key over plain http.
    """
    has_key = encrypted_api_key is not None or row.encrypted_api_key is not None
    if api_base is not None and has_key and (problem := https_or_loopback_problem(api_base)) is not None:
        # A key travels to it, so it is held to what a configured provider's endpoint is.
        raise HostedProviderUnsafeApiBaseError(f"Region endpoint {problem}")
    if encrypted_api_key is not None:
        return
    if (
        row.encrypted_api_key is not None
        and api_base is not None
        and _safe_origin(api_base) != _safe_origin(row.api_base)
    ):
        # The provider's key would otherwise reach an endpoint chosen without
        # ever having to know the key, as an update refuses.
        raise HostedProviderKeyRequiredError
    effective_base = api_base if api_base is not None else row.api_base
    if row.encrypted_api_key is None and not hosted_key_optional(row.provider, effective_base):
        raise HostedProviderApiKeyMissingError(row.provider)


def ensure_regions_follow(row: HostedProvider, regions: Sequence[HostedProviderRegion], *, moved: bool) -> None:
    """Refuse a change to the provider that its regions cannot follow.

    ``moved`` is whether the change points the provider at another origin. A
    region with a key of its own on the provider's endpoint must not have that
    key carried there, a region using the provider's key on an endpoint of its
    own must be on the provider's origin (checked whatever changed, a key added
    included), and a region using the provider's key must still have one where
    its endpoint needs it.

    Raises:
        HostedProviderRegionKeyRequiredError: a region's key would reach an endpoint it was not entered for.
        HostedProviderApiKeyMissingError: a region is left with no key where one is needed.
    """
    for region in regions:
        if region.encrypted_api_key is not None:
            if moved and region.api_base is None:
                raise HostedProviderRegionKeyRequiredError(region.slug)
            continue
        if (
            row.encrypted_api_key is not None
            and region.api_base is not None
            and _safe_origin(region.api_base) != _safe_origin(row.api_base)
        ):
            raise HostedProviderRegionKeyRequiredError(region.slug)
        api_base = region.api_base if region.api_base is not None else row.api_base
        if row.encrypted_api_key is None and not hosted_key_optional(row.provider, api_base):
            raise HostedProviderApiKeyMissingError(row.provider)


def _region(regions: Sequence[HostedProviderRegion], provider: str, slug: str) -> HostedProviderRegion:
    """The region of that slug.

    Raises:
        HostedProviderRegionNotFoundError: the provider has no region of that slug.
    """
    for region in regions:
        if region.slug == slug:
            return region
    raise HostedProviderRegionNotFoundError(provider, slug)


class HostedProviderRegions:
    """Change the regions of a stored hosted provider. A provider defined in configuration has none.

    Every change locks the provider's row first and reads its regions after
    the lock, so two changes to one provider's regions run one after the other:
    neither can, say, switch off the last region the other left switched on.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        regions: HostedProviderRegionRepository,
        stored: Callable[[str], Awaitable[tuple[HostedProvider, HostedSource]]],
        lock: Callable[[HostedProvider], Awaitable[bool]],
        taken: Callable[[str], Awaitable[bool]],
        validate_api_base: Callable[[str | None], Awaitable[None]],
    ) -> None:
        """Bind the unit of work, the regions table, and the provider lookups the service owns.

        ``stored`` answers a provider's row and its merged view, refusing one
        defined in configuration; ``lock`` locks a provider's row for the rest
        of the transaction and reads it again under the lock, false when it
        was deleted first; ``taken`` answers whether a hosted provider is
        configured under a whole name, which a region's selector must not
        shadow; ``validate_api_base`` is the SSRF gate an endpoint passes
        before it is stored.
        """
        self.uow = uow
        self.config = config
        self.regions = regions
        self.stored = stored
        self.lock = lock
        self.taken = taken
        self.validate_api_base = validate_api_base

    async def add(self, provider: str, request: HostedProviderRegionCreateRequest) -> str:
        """Add a region after the provider's others, and return the provider's name.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
            HostedProviderInvalidRegionSlugError: a slug a selector could not pin.
            HostedProviderRegionAlreadyExistsError: the provider has a region of that slug, or a provider is
                configured under the selector it would take.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate, or carries a key over http.
            HostedProviderKeyRequiredError: the region points the provider's key at another origin.
            HostedProviderApiKeyMissingError: the region is left with no key where one is needed.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
        """
        slug = _slug(request.slug)
        api_base = normalize_api_base(request.api_base)
        await self.validate_api_base(api_base)
        encrypted_key = encrypt_key(request.api_key)
        encrypted_client_args = encrypt_client_args(request.client_args)

        async with self.uow:
            row, regions = await self._locked(provider)
            position = max((region.position for region in regions), default=-1) + 1
            new_slugs = [slug]
            if not regions:
                primary = _slug(request.primary_slug or _primary_slug(row))
                if primary == slug:
                    raise HostedProviderRegionAlreadyExistsError(row.name, slug)
                new_slugs.append(primary)
            for each in new_slugs:
                if await self.taken(region_instance(row.name, each)):
                    raise HostedProviderRegionAlreadyExistsError(row.name, each)
            _ensure_servable(row, encrypted_api_key=encrypted_key, api_base=api_base)
            if not regions:
                await self._insert(
                    row,
                    slug=new_slugs[1],
                    label=request.primary_label or row.data_location or "Primary",
                    position=0,
                    data_location=row.data_location,
                )
                position = 1
            await self._insert(
                row,
                slug=slug,
                label=request.label,
                position=position,
                enabled=request.enabled,
                # Copied, never inherited: a later change to the provider must
                # not move where a region says its data is kept.
                data_location=request.data_location
                if "data_location" in request.model_fields_set
                else row.data_location,
                encrypted_api_key=encrypted_key,
                api_key_last4=last4(request.api_key) if request.api_key is not None else None,
                api_base=api_base,
                encrypted_client_args=encrypted_client_args,
                supported_countries=request.supported_countries,
            )
            name = row.name
        await self._written(name)
        return name

    async def update(self, provider: str, slug: str, request: HostedProviderRegionUpdateRequest) -> str:
        """Change one region, and return the provider's name.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
            HostedProviderRegionNotFoundError: the provider has no region of that slug.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate, or carries a key over http.
            HostedProviderApiBaseMaskedError: ``api_base`` carries the mask but is not the stored one echoed back.
            HostedProviderKeyRequiredError: the region's endpoint moves to another origin and no key came
                with it, or it points the provider's key at another origin.
            HostedProviderApiKeyMissingError: the region is left with no key where one is needed.
            HostedProviderClientArgsUnreadableError: a masked entry has nothing stored to keep.
            HostedProviderLastRegionError: the change leaves no region switched on.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
        """
        fields = request.model_fields_set
        api_base = normalize_api_base(request.api_base)
        if "api_base" in fields:
            await self.validate_api_base(api_base)
        encrypted_key = encrypt_key(request.api_key) if request.api_key is not None else None

        async with self.uow:
            row, regions = await self._locked(provider)
            region = _region(regions, row.name, slug)
            if request.label is not None:
                region.label = request.label
            if "api_base" in fields:
                before = _safe_origin(region.api_base if region.api_base is not None else row.api_base)
                region.api_base = None if api_base is None else unmask_api_base(api_base, region.api_base)
                after = _safe_origin(region.api_base if region.api_base is not None else row.api_base)
                if region.encrypted_api_key is not None and "api_key" not in fields and before != after:
                    # The region's own key would otherwise reach an endpoint
                    # chosen without ever having to know it.
                    raise HostedProviderKeyRequiredError
            if "api_key" in fields:
                region.encrypted_api_key = encrypted_key
                region.api_key_last4 = last4(request.api_key) if request.api_key is not None else None
            if "client_args" in fields:
                stored = client_args_of(region)
                if stored is None and carries_redaction(request.client_args):
                    raise HostedProviderClientArgsUnreadableError
                region.encrypted_client_args = encrypt_client_args(restore_redacted_values(request.client_args, stored))
            if "data_location" in fields:
                region.data_location = request.data_location
            if "supported_countries" in fields:
                region.supported_countries = request.supported_countries
            if request.enabled is not None:
                region.enabled = request.enabled
            if not any(each.enabled for each in regions):
                raise HostedProviderLastRegionError
            _ensure_servable(row, encrypted_api_key=region.encrypted_api_key, api_base=region.api_base)
            ordered = [other for other in regions if other is not region]
            ordered.insert(
                min(request.position, len(ordered)) if request.position is not None else regions.index(region),
                region,
            )
            for position, each in enumerate(ordered):
                each.position = position
            await self.regions.flush()
            name = row.name
        await self._written(name)
        return name

    async def remove(self, provider: str, slug: str) -> str:
        """Remove one region, and return the provider's name.

        Removing the last leaves the provider serving from its own connection.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
            HostedProviderRegionNotFoundError: the provider has no region of that slug.
            HostedProviderLastRegionError: the regions left would have none switched on.
        """
        async with self.uow:
            row, regions = await self._locked(provider)
            region = _region(regions, row.name, slug)
            remaining = [other for other in regions if other is not region]
            if remaining and not any(other.enabled for other in remaining):
                raise HostedProviderLastRegionError
            await self.regions.delete_row(region)
            name = row.name
        await self._written(name)
        return name

    async def _locked(self, provider: str) -> tuple[HostedProvider, list[HostedProviderRegion]]:
        """The provider's row, locked, and its regions as they stand once it is."""
        row, _ = await self.stored(provider)
        if not await self.lock(row):
            raise HostedProviderNotFoundError(provider)
        return row, await self.regions.list_for_provider(row.id, fresh=True)

    async def _insert(self, row: HostedProvider, **fields: object) -> None:
        try:
            await self.regions.insert(hosted_provider_id=row.id, **fields)
        except HostedRegionConflict as conflict:
            raise HostedProviderRegionAlreadyExistsError(row.name, conflict.slug) from conflict

    async def _written(self, name: str) -> None:
        """Bring this worker's name index and model listing up to the regions just stored.

        The regions are committed by now, so a failed refresh is logged rather
        than raised: the periodic refresh brings the index up within its TTL.
        """
        get_model_cache().clear(name)
        try:
            await refresh_hosted_name_index(self.uow, self.config)
        except Exception:
            logger.warning(
                "Hosted provider name refresh after a region change failed; the next refresh will retry", exc_info=True
            )
