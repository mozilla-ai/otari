"""The providers domain's settings.

Only the hosted providers declared in configuration live here so far; the
``providers:`` block of named instances is still on ``GatewayConfig`` in
``core/config.py``.

``hosted_providers`` is keyed by the hosted provider's name and takes the
fields the hosted-providers API body takes, ``provider`` naming the any-llm
implementation where it differs from the name. An entry is never written to the
database and never encrypted: it is read from this block on every use, and a
stored row of the same name wins over it. ``${VAR}`` interpolation reaches it as it
reaches the rest of a configuration file, and ``OTARI_HOSTED_PROVIDERS`` sets
the whole block as JSON, replacing the file's block rather than merging into it
(``load_config`` applies it).

Validation errors name the entry and the field and never the value, because a
value here may be a key.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

from any_llm import LLMProvider
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gateway.core.settings_view import OMITTED

# What a ``hosted_providers:`` entry may not carry, with the reason it is refused.
_REFUSED_KEYS = {
    "provider_type": "a hosted provider names its implementation in 'provider'",
}


class _HostedProviderEntryShape(BaseModel):
    """The fields one ``hosted_providers:`` entry takes, as the API body takes them."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(default=None, min_length=1)
    api_key: str | None = Field(default=None, min_length=1)
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
    enabled: bool = True
    restricted_to_organization_id: uuid.UUID | None = None
    models: list[str] | None = None


@dataclass(frozen=True)
class HostedProviderEntry:
    """One hosted provider declared in configuration, validated."""

    name: str
    provider: str
    api_key: str | None = field(repr=False)
    api_base: str | None
    client_args: dict[str, Any] | None = field(repr=False)
    enabled: bool
    restricted_to_organization_id: uuid.UUID | None
    # Model ids the provider offers when it cannot list them itself.
    models: tuple[str, ...]


def _entry_error(name: str, exc: ValidationError) -> ValueError:
    """The first problem with an entry, named by its field and never by its value."""
    problems = []
    for error in exc.errors(include_input=False, include_url=False):
        where = ".".join(str(part) for part in error["loc"])
        problems.append(f"hosted_providers.{name}.{where}: {error['msg']}" if where else error["msg"])
    return ValueError("; ".join(problems))


def parse_hosted_providers(raw: Mapping[str, Any]) -> dict[str, HostedProviderEntry]:
    """Validate the ``hosted_providers:`` block into one entry per name.

    Raises:
        ValueError: an entry names no any-llm implementation, has a name a
            hosted provider cannot take, carries a key it may not, or holds a
            field of the wrong shape.
    """
    entries: dict[str, HostedProviderEntry] = {}
    for name, entry in raw.items():
        # A bare ``ollama:`` is an entry with every field at its default.
        entry = {} if entry is None else entry
        if not isinstance(entry, Mapping):
            msg = f"hosted_providers.{name} must be a mapping of settings."
            raise ValueError(msg)
        for key, reason in _REFUSED_KEYS.items():
            if key in entry:
                msg = f"hosted_providers.{name}.{key} is not accepted: {reason}"
                raise ValueError(msg)
        try:
            shape = _HostedProviderEntryShape.model_validate(dict(entry))
        except ValidationError as exc:
            raise _entry_error(name, exc) from None
        provider = shape.provider.strip() if shape.provider is not None else name
        try:
            LLMProvider(provider)
        except ValueError:
            where = f"hosted_providers.{name}.provider" if shape.provider is not None else f"hosted_providers.{name}"
            msg = f"{where} is not an any-llm implementation; name one in 'provider' for a named entry."
            raise ValueError(msg) from None
        api_base = shape.api_base.strip() if shape.api_base is not None else None
        entries[name] = HostedProviderEntry(
            name=name,
            provider=provider,
            api_key=shape.api_key,
            api_base=api_base or None,
            client_args=shape.client_args,
            enabled=shape.enabled,
            restricted_to_organization_id=shape.restricted_to_organization_id,
            models=tuple(shape.models or ()),
        )
    return entries


class ProviderSettings(BaseModel):
    """The hosted providers this deployment serves on credentials it was configured with."""

    # ``Any`` values rather than a model: a validation error carries the input
    # it refused, and an input here may be a key. ``parse_hosted_providers``
    # validates the entries with errors that name a field and never a value.
    hosted_providers: Annotated[dict[str, Any], OMITTED] = Field(
        default_factory=dict,
        # Each entry may carry a key, so the block stays out of the config's repr.
        repr=False,
        description=(
            "Hosted providers declared in configuration, keyed by name, each with the fields the hosted-providers "
            "API takes: 'provider' (the any-llm implementation, defaulting to the name), 'api_key', 'api_base', "
            "'client_args', 'enabled', "
            "'restricted_to_organization_id', and an optional 'models' list offered when the provider cannot list "
            "its own. Served after every other credential, as a stored hosted provider is, and never written to "
            "the database; a stored hosted provider of the same name wins. Settable as JSON through "
            "OTARI_HOSTED_PROVIDERS, which replaces the file's block whole."
        ),
    )

    # Read once at startup by the import it gates, so it is no setting the dashboard shows or changes.
    import_configured_providers: Annotated[bool, OMITTED] = Field(
        default=True,
        description=(
            "Copy each providers: entry of config.yml onto a hosted provider of its name at startup, once, and serve "
            "the name from that hosted provider from then on. The providers: block is deprecated; an entry a hosted "
            "provider cannot hold keeps serving from config.yml. Also settable via OTARI_IMPORT_CONFIGURED_PROVIDERS."
        ),
    )

    @field_validator("hosted_providers", mode="before")
    @classmethod
    def _empty_block_is_no_providers(cls, value: Any) -> Any:
        """Read a ``hosted_providers:`` block with every entry commented out as no providers."""
        return {} if value is None else value

    def hosted_provider_entries(self) -> dict[str, HostedProviderEntry]:
        """The configured hosted providers, validated.

        Raises:
            ValueError: the block holds an entry ``parse_hosted_providers`` refuses.
        """
        return parse_hosted_providers(self.hosted_providers)
