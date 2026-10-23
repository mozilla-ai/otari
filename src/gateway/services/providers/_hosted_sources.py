"""Where a hosted provider comes from: a stored row, or an entry in configuration.

The deployment's hosted providers are the stored rows merged with the
``hosted_providers:`` block, one per name, and a row wins over an entry of the
same name. Everything that serves, lists or dials a hosted provider
reads this merged view, so a provider declared in configuration is served,
discovered and swept exactly as a stored one is. An entry is read from the
configuration on every use and never written to the database.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from gateway.core.config import GatewayConfig
from gateway.core.settings.providers import HostedProviderEntry
from gateway.models.providers import HostedProvider
from gateway.repositories.providers import HostedProviderRepository
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider, credential_of, lookup_name
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

HostedSourceKind = Literal["database", "config"]


@dataclass(frozen=True)
class HostedSource:
    """One hosted provider as the deployment serves it, from whichever source won.

    Exactly one of ``row`` and ``entry`` is set.
    """

    row: HostedProvider | None = None
    entry: HostedProviderEntry | None = None
    # Whether a configuration entry of the same name is shadowed by ``row``.
    shadows_config: bool = False

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
    def declared_models(self) -> tuple[str, ...]:
        """The model ids configuration offers when the provider cannot list them; none for a row."""
        return () if self.entry is None else self.entry.models

    def credential(self, *, quiet: bool = False) -> ResolvedHostedProvider | None:
        """The connection this provider serves on, or None when a stored one cannot be decrypted."""
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


class HostedSources:
    """Read the merged view. Called inside the caller's Unit of Work, for the rows."""

    def __init__(self, config: GatewayConfig, providers: HostedProviderRepository) -> None:
        self.config = config
        self.providers = providers

    def entries(self) -> dict[str, HostedProviderEntry]:
        """The hosted providers configuration declares, validated."""
        return self.config.hosted_provider_entries()

    async def all(self) -> list[HostedSource]:
        """Every hosted provider by name, a row winning over an entry of the same name."""
        entries = self.entries()
        merged = {
            row.name: HostedSource(row=row, shadows_config=row.name in entries)
            for row in await self.providers.list_all()
        }
        for name, entry in entries.items():
            merged.setdefault(name, HostedSource(entry=entry))
        return [merged[name] for name in sorted(merged)]

    async def get(self, name: str) -> HostedSource | None:
        """One hosted provider by name, with the accepted aliases folded, or None."""
        name = lookup_name(name)
        entries = self.entries()
        row = await self.providers.get_by_name(name)
        if row is not None:
            return HostedSource(row=row, shadows_config=name in entries)
        entry = entries.get(name)
        return None if entry is None else HostedSource(entry=entry)


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
