"""Where a hosted provider comes from: a stored row, or an entry in configuration.

The deployment's hosted providers are the stored rows merged with the
``hosted_providers:`` block, one per name, and a row wins over an entry of the
same name. Everything that serves, lists or dials a hosted provider
reads this merged view, so a provider declared in configuration is served,
discovered and swept exactly as a stored one is. An entry is read from the
configuration on every use and never written to the database.
"""

import uuid
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from gateway.core.config import REGION_SEPARATOR, GatewayConfig
from gateway.core.settings.inference import data_location_admitted
from gateway.core.settings.providers import HostedProviderEntry
from gateway.models.providers import HostedProvider, HostedProviderRegion
from gateway.repositories.providers import HostedProviderRegionRepository, HostedProviderRepository
from gateway.services.providers._hosted_credentials import (
    ResolvedHostedProvider,
    credential_of,
    lookup_name,
    region_credential_of,
)
from gateway.services.providers._hosted_name_index import serving_data_locations, serving_region_data_locations
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

HostedSourceKind = Literal["database", "config"]


@dataclass(frozen=True)
class HostedProviderProfile:
    """What the public Providers pages say about a hosted provider: its implementation and its policies."""

    implementation: str
    terms_url: str | None
    privacy_url: str | None


@dataclass(frozen=True)
class HostedDataLocations:
    """Where hosted providers keep request data, as a viewer may use them.

    ``providers`` is by the provider's name, one entry per place it may serve
    from. ``regions`` is by a region's name (``tokenhub@sg``), for an offering
    pinned to it, built from the region rows and never by splitting a name.
    ``None`` stands for a connection that declares nothing.
    """

    providers: dict[str, tuple[str | None, ...]]
    regions: dict[str, tuple[str | None, ...]]

    def within(self, allowed: frozenset[str] | None) -> "HostedDataLocations":
        """Only the locations ``allowed`` admits, as dispatch serves a workspace; ``None`` for any location."""
        if allowed is None:
            return self

        def kept(found: dict[str, tuple[str | None, ...]]) -> dict[str, tuple[str | None, ...]]:
            return {
                name: tuple(each for each in locations if data_location_admitted(each, allowed))
                for name, locations in found.items()
            }

        return HostedDataLocations(providers=kept(self.providers), regions=kept(self.regions))


@dataclass(frozen=True)
class HostedSource:
    """One hosted provider as the deployment serves it, from whichever source won.

    Exactly one of ``row`` and ``entry`` is set.
    """

    row: HostedProvider | None = None
    entry: HostedProviderEntry | None = None
    # Whether a configuration entry of the same name is shadowed by ``row``.
    shadows_config: bool = False
    # The stored row's regions, in the order they are tried; none serves from the row itself.
    regions: tuple[HostedProviderRegion, ...] = ()

    @property
    def source(self) -> HostedSourceKind:
        return "database" if self.row is not None else "config"

    @property
    def name(self) -> str:
        """What requests, the roster, the price list and usage know this provider by."""
        return self.row.name if self.row is not None else self._entry.name

    @property
    def implementation(self) -> str:
        """The any-llm implementation serving it."""
        return self.row.provider if self.row is not None else self._entry.provider

    @property
    def identity(self) -> tuple[str, uuid.UUID | None]:
        """What serves the provider: a configuration entry, or one stored row."""
        return (self.source, None if self.row is None else self.row.id)

    @property
    def enabled(self) -> bool:
        return self.row.enabled if self.row is not None else self._entry.enabled

    @property
    def serves_unlisted_models(self) -> bool:
        """Whether a model with no roster row is served; a configuration entry serves its roster alone."""
        return self.row is not None and self.row.serves_unlisted_models

    @property
    def restricted_to_organization_id(self) -> uuid.UUID | None:
        if self.row is not None:
            return self.row.restricted_to_organization_id
        return self._entry.restricted_to_organization_id

    @property
    def supported_countries(self) -> Collection[str] | None:
        """The countries whose requests it serves, or None for every country."""
        return self.row.supported_countries if self.row is not None else self._entry.supported_countries

    @property
    def zero_data_retention(self) -> bool:
        """Whether the operator declared that the upstream keeps no prompt or completion data."""
        return self.row.zero_data_retention if self.row is not None else self._entry.zero_data_retention

    @property
    def terms_url(self) -> str | None:
        return self.row.terms_url if self.row is not None else self._entry.terms_url

    @property
    def privacy_url(self) -> str | None:
        return self.row.privacy_url if self.row is not None else self._entry.privacy_url

    @property
    def data_location(self) -> str | None:
        """Where the upstream keeps request data, as the operator declares it, or None."""
        return self.row.data_location if self.row is not None else self._entry.data_location

    def region(self, slug: str) -> HostedProviderRegion | None:
        """The region of that slug, or None."""
        return next((region for region in self.regions if region.slug == slug), None)

    @property
    def declared_models(self) -> tuple[str, ...]:
        """The model ids configuration offers when the provider cannot list them; none for a row."""
        return () if self.entry is None else self.entry.models

    def credential(self, *, quiet: bool = False) -> ResolvedHostedProvider | None:
        """The connection this provider serves on first, or None when there is none it can read.

        For a provider with regions, the first enabled region: what discovery,
        the health check and a request no country narrows are served on.
        """
        regions = self.region_credentials(quiet=quiet)
        return regions[0] if regions else None

    def region_credentials(self, *, quiet: bool = False, slug: str | None = None) -> list[ResolvedHostedProvider]:
        """The connections this provider serves on, in the order they are tried.

        One, the provider's own, for a provider with no regions. Otherwise each
        enabled region a configured key reads, or only the one ``slug`` names.
        A provider whose own ciphertext no key reads serves nothing.
        """
        own = self._own_credential(quiet=quiet)
        if own is None:
            return []
        if not self.regions:
            return [own] if slug is None else []
        credentials: list[ResolvedHostedProvider] = []
        for region in self.regions:
            if not region.enabled or (slug is not None and region.slug != slug):
                continue
            credential = region_credential_of(own, region, quiet=quiet)
            if credential is not None:
                credentials.append(credential)
        return credentials

    def _own_credential(self, *, quiet: bool) -> ResolvedHostedProvider | None:
        if self.row is not None:
            return credential_of(self.row, quiet=quiet)
        entry = self._entry
        return ResolvedHostedProvider(
            api_key=entry.api_key,
            api_base=entry.api_base,
            client_args=entry.client_args,
            name=entry.name,
            implementation=entry.provider,
            supported_countries=entry.supported_countries,
            data_location=entry.data_location,
        )

    def withheld_from(self, organization_id: uuid.UUID | None) -> bool:
        """Whether ``organization_id`` is served as if this provider did not exist.

        ``None`` is the deployment's own view, which sees every provider.
        """
        restriction = self.restricted_to_organization_id
        return organization_id is not None and restriction is not None and restriction != organization_id

    @property
    def _entry(self) -> HostedProviderEntry:
        if self.entry is None:  # pragma: no cover - the constructor sets one of the two
            msg = "a hosted source holds a row or an entry"
            raise AssertionError(msg)
        return self.entry


def _region_selector(name: str) -> tuple[str, str] | None:
    """``name`` read as ``<provider>@<slug>``, or None when it names no region."""
    base, separator, slug = name.rpartition(REGION_SEPARATOR)
    return (base, slug) if separator and slug else None


@dataclass(frozen=True)
class HostedSnapshot:
    """Every hosted provider by name, read once, on rows no session holds."""

    by_name: dict[str, HostedSource]

    def provider_named(self, name: str) -> bool:
        """Whether a hosted provider goes by exactly ``name``, rather than a region of one."""
        return name in self.by_name

    def get_serving(self, name: str) -> tuple[HostedSource, str | None] | None:
        """What :meth:`HostedSources.get_serving` answers, from the snapshot."""
        source = self.by_name.get(lookup_name(name))
        if source is not None:
            return source, None
        selector = _region_selector(name)
        if selector is None:
            return None
        base, slug = selector
        source = self.by_name.get(lookup_name(base))
        if source is None or source.region(slug) is None:
            return None
        return source, slug


class HostedSources:
    """Read the merged view. Called inside the caller's Unit of Work, for the rows."""

    def __init__(
        self, config: GatewayConfig, providers: HostedProviderRepository, regions: HostedProviderRegionRepository
    ) -> None:
        self.config = config
        self.providers = providers
        self.regions = regions

    def entries(self) -> dict[str, HostedProviderEntry]:
        """The hosted providers configuration declares, validated."""
        return self.config.hosted_provider_entries()

    async def all(self) -> list[HostedSource]:
        """Every hosted provider by name, a row winning over an entry of the same name."""
        entries = self.entries()
        rows = await self.providers.list_all()
        regions = await self.regions.list_for_providers([row.id for row in rows])
        merged = {
            row.name: HostedSource(row=row, shadows_config=row.name in entries, regions=tuple(regions.get(row.id, ())))
            for row in rows
        }
        for name, entry in entries.items():
            merged.setdefault(name, HostedSource(entry=entry))
        return [merged[name] for name in sorted(merged)]

    async def snapshot(self) -> HostedSnapshot | None:
        """Every hosted provider by name on detached rows, or None when there are more than a listing holds."""
        rows = await self.providers.list_all_detached()
        if rows is None:
            return None
        entries = self.entries()
        regions = await self.regions.list_for_providers_detached([row.id for row in rows])
        merged = {
            row.name: HostedSource(row=row, shadows_config=row.name in entries, regions=tuple(regions.get(row.id, ())))
            for row in rows
        }
        for name, entry in entries.items():
            merged.setdefault(name, HostedSource(entry=entry))
        return HostedSnapshot(merged)

    async def zero_data_retention_names(self, organization_id: uuid.UUID | None) -> frozenset[str]:
        """The enabled hosted providers ``organization_id`` may use that declare zero data retention, by name."""
        return frozenset(
            source.name
            for source in await self.all()
            if source.enabled and source.zero_data_retention and not source.withheld_from(organization_id)
        )

    async def data_locations(self, organization_id: uuid.UUID | None) -> HostedDataLocations:
        """Where each enabled hosted provider ``organization_id`` may use keeps request data, and each region of one.

        Read from the declarations themselves, as the workspace narrowing
        reads them (:func:`serving_data_locations`), with no key decrypted.
        """
        everything = await self.all()
        names = {source.name for source in everything}
        providers: dict[str, tuple[str | None, ...]] = {}
        regions: dict[str, tuple[str | None, ...]] = {}
        for source in everything:
            if not source.enabled or source.withheld_from(organization_id):
                continue
            providers[source.name] = serving_data_locations(source.name, source.data_location, source.regions, names)
            for instance, location in serving_region_data_locations(source.name, source.regions, names).items():
                regions[instance] = (location,)
        return HostedDataLocations(providers=providers, regions=regions)

    async def names_limited_to_countries(self) -> list[str]:
        """The enabled hosted providers that serve only some countries, by name."""
        return [
            source.name
            for source in await self.all()
            if source.enabled
            and (source.supported_countries or any(region.supported_countries for region in source.regions))
        ]

    async def get(self, name: str) -> HostedSource | None:
        """One hosted provider by name, with the accepted aliases folded, or None.

        Exact: what the operator API changes is the provider of that whole name.
        """
        name = lookup_name(name)
        entries = self.entries()
        row = await self.providers.get_by_name(name)
        if row is not None:
            regions = await self.regions.list_for_provider(row.id)
            return HostedSource(row=row, shadows_config=name in entries, regions=tuple(regions))
        entry = entries.get(name)
        return None if entry is None else HostedSource(entry=entry)

    async def get_serving(self, name: str) -> tuple[HostedSource, str | None] | None:
        """The provider a request for ``name`` is served by, and the region ``name`` pins, or None.

        A provider of that whole name wins, even one whose name holds the
        separator. Otherwise ``tokenhub@sg`` is the region ``sg`` of
        ``tokenhub``, and only a region the provider has.
        """
        source = await self.get(name)
        if source is not None:
            return source, None
        selector = _region_selector(name)
        if selector is None:
            return None
        base, slug = selector
        source = await self.get(base)
        if source is None or source.region(slug) is None:
            return None
        return source, slug


async def refuse_unsafe_configured_endpoints(config: GatewayConfig) -> None:
    """Refuse a ``hosted_providers:`` endpoint the SSRF gate rejects, as a stored provider's is refused at write.

    A no-op unless ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false`` turns the gate
    on. Run at boot rather than at config load, because the gate resolves the
    host.

    Raises:
        ValueError: every entry the gate rejects, by name.
    """
    problems: list[str] = []
    for name, entry in config.hosted_provider_entries().items():
        if not entry.api_base:
            continue
        try:
            await validate_provider_api_base(entry.api_base)
        except UnsafeURLError as exc:
            problems.append(f"hosted_providers.{name}.api_base: {exc}")
    if problems:
        raise ValueError("; ".join(problems))
