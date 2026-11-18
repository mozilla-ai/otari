"""The regions a hosted provider serves from: added, changed, reordered and removed.

A provider with no regions serves from its own connection. Its first region
keeps that connection as a region of its own, ahead of the new one, so adding
a region never changes what a request already served is served on. Every field
a region leaves unset is the provider's (``HostedProviderRegion``).
"""

from collections.abc import Awaitable, Callable

from gateway.core.config import GatewayConfig, hosted_key_optional, region_instance, region_slug_problem
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    HostedProviderApiKeyMissingError,
    HostedProviderClientArgsUnreadableError,
    HostedProviderInvalidRegionSlugError,
    HostedProviderKeyRequiredError,
    HostedProviderRegionAlreadyExistsError,
    HostedProviderRegionNotFoundError,
)
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


def _ensure_servable(row: HostedProvider, *, encrypted_api_key: str | None, api_base: str | None) -> None:
    """Refuse a region left with no key where its provider needs one, or one sending the provider's key elsewhere.

    Raises:
        HostedProviderKeyRequiredError: the region points the provider's key at another origin.
        HostedProviderApiKeyMissingError: neither the region nor the provider holds a key the endpoint needs.
    """
    if encrypted_api_key is not None:
        return
    if row.encrypted_api_key is not None and api_base is not None and origin_of(api_base) != origin_of(row.api_base):
        # The provider's key would otherwise reach an endpoint chosen without
        # ever having to know the key, as an update refuses.
        raise HostedProviderKeyRequiredError
    effective_base = api_base if api_base is not None else row.api_base
    if row.encrypted_api_key is None and not hosted_key_optional(row.provider, effective_base):
        raise HostedProviderApiKeyMissingError(row.provider)


class HostedProviderRegions:
    """Change the regions of a stored hosted provider. A provider defined in configuration has none."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        regions: HostedProviderRegionRepository,
        stored: Callable[[str], Awaitable[tuple[HostedProvider, HostedSource]]],
        validate_api_base: Callable[[str | None], Awaitable[None]],
    ) -> None:
        """Bind the unit of work, the regions table, and the provider lookups the service owns.

        ``stored`` answers a provider's row and its merged view, refusing one
        defined in configuration; ``validate_api_base`` is the SSRF gate an
        endpoint passes before it is stored.
        """
        self.uow = uow
        self.config = config
        self.regions = regions
        self.stored = stored
        self.validate_api_base = validate_api_base

    async def add(self, provider: str, request: HostedProviderRegionCreateRequest) -> str:
        """Add a region after the provider's others, and return the provider's name.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
            HostedProviderInvalidRegionSlugError: a slug a selector could not pin.
            HostedProviderRegionAlreadyExistsError: the provider already has a region of that slug.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
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
            row, source = await self.stored(provider)
            position = max((region.position for region in source.regions), default=-1) + 1
            if not source.regions:
                primary = _slug(request.primary_slug or _primary_slug(row))
                if primary == slug:
                    raise HostedProviderRegionAlreadyExistsError(row.name, slug)
                await self._insert(
                    row,
                    slug=primary,
                    label=request.primary_label or row.data_location or "Primary",
                    position=0,
                )
                position = 1
            _ensure_servable(row, encrypted_api_key=encrypted_key, api_base=api_base)
            await self._insert(
                row,
                slug=slug,
                label=request.label,
                position=position,
                enabled=request.enabled,
                data_location=request.data_location,
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
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            HostedProviderApiBaseMaskedError: ``api_base`` carries the mask but is not the stored one echoed back.
            HostedProviderKeyRequiredError: the region points the provider's key at another origin.
            HostedProviderApiKeyMissingError: the region is left with no key where one is needed.
            HostedProviderClientArgsUnreadableError: a masked entry has nothing stored to keep.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
        """
        fields = request.model_fields_set
        api_base = normalize_api_base(request.api_base)
        if "api_base" in fields:
            await self.validate_api_base(api_base)
        encrypted_key = encrypt_key(request.api_key) if request.api_key is not None else None

        async with self.uow:
            row, source = await self.stored(provider)
            region = source.region(slug)
            if region is None:
                raise HostedProviderRegionNotFoundError(row.name, slug)
            if request.label is not None:
                region.label = request.label
            if "api_base" in fields:
                region.api_base = None if api_base is None else unmask_api_base(api_base, region.api_base)
            if "api_key" in fields:
                region.encrypted_api_key = encrypted_key
                region.api_key_last4 = last4(request.api_key) if request.api_key is not None else None
            if "client_args" in fields:
                stored = client_args_of(region)
                if stored is None and carries_redaction(request.client_args):
                    raise HostedProviderClientArgsUnreadableError
                region.encrypted_client_args = encrypt_client_args(
                    restore_redacted_values(request.client_args, stored)
                )
            if "data_location" in fields:
                region.data_location = request.data_location
            if "supported_countries" in fields:
                region.supported_countries = request.supported_countries
            if request.enabled is not None:
                region.enabled = request.enabled
            _ensure_servable(row, encrypted_api_key=region.encrypted_api_key, api_base=region.api_base)
            ordered = [other for other in source.regions if other is not region]
            if request.position is not None:
                ordered.insert(min(request.position, len(ordered)), region)
            else:
                ordered.insert(source.regions.index(region), region)
            for position, each in enumerate(ordered):
                each.position = position
                await self.regions.save(each)
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
        """
        async with self.uow:
            row, source = await self.stored(provider)
            region = source.region(slug)
            if region is None:
                raise HostedProviderRegionNotFoundError(row.name, slug)
            await self.regions.delete_row(region)
            name = row.name
        await self._written(name)
        return name

    async def _insert(self, row: HostedProvider, **fields: object) -> None:
        try:
            await self.regions.insert(hosted_provider_id=row.id, **fields)
        except HostedRegionConflict as conflict:
            raise HostedProviderRegionAlreadyExistsError(row.name, conflict.slug) from conflict

    async def _written(self, name: str) -> None:
        """Bring this worker's name index and model listing up to the regions just stored."""
        await refresh_hosted_name_index(self.uow, self.config)
        get_model_cache().clear(name)
