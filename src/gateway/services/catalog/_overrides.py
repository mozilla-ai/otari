"""What an operator or an organization says about a model, kept in the database and in each worker.

A server can list a model with nothing but its name, and models.dev may not
know it, so its limits and capabilities would read as unknown. An override
fills in what nobody else says, and wins over what they do: whoever set it
knows how the model is served here.

The copy is reloaded with the served models (``refresh_served_models``), so a
write reaches its own worker at once and every other worker within
``SERVED_MODEL_REFRESH_SECONDS``.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, fields

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.catalog import ModelMetadataOverride
from gateway.repositories.catalog import ModelMetadataOverrideRepository


@dataclass(frozen=True, slots=True)
class MetadataOverride:
    """One model's overridden fields; ``None`` leaves a field to the server and models.dev."""

    context_window: int | None = None
    max_output_tokens: int | None = None
    tool_call: bool | None = None
    reasoning: bool | None = None
    structured_output: bool | None = None
    input_modalities: tuple[str, ...] | None = None


_FIELDS = tuple(field.name for field in fields(MetadataOverride))

# (organization, provider) -> model id -> override; the organization is ``None`` for the deployment's.
_overrides: dict[tuple[uuid.UUID | None, str], dict[str, MetadataOverride]] = {}


def metadata_override(
    provider: str, model_id: str, organization_id: uuid.UUID | None = None
) -> MetadataOverride | None:
    """The override of ``provider:model_id``: the deployment's, or ``organization_id``'s own."""
    return _overrides.get((organization_id, provider), {}).get(model_id)


def _override(row: ModelMetadataOverride) -> MetadataOverride:
    return MetadataOverride(
        context_window=row.context_window,
        max_output_tokens=row.max_output_tokens,
        tool_call=row.tool_call,
        reasoning=row.reasoning,
        structured_output=row.structured_output,
        input_modalities=tuple(row.input_modalities) if row.input_modalities is not None else None,
    )


def replace_overrides(rows: Sequence[ModelMetadataOverride]) -> None:
    """Swap the copy for ``rows`` in one step."""
    global _overrides  # noqa: PLW0603

    overrides: dict[tuple[uuid.UUID | None, str], dict[str, MetadataOverride]] = {}
    for row in rows:
        overrides.setdefault((row.organization_id, row.provider), {})[row.model] = _override(row)
    _overrides = overrides


def _split(selector: str) -> tuple[str, str]:
    provider, separator, model = selector.partition(":")
    if not separator or not provider or not model:
        raise ValueError(f"'{selector}' is not a provider:model selector")
    return provider, model


class ModelMetadataOverrideService:
    """Set, clear and read the overrides of one owner: the deployment, or an organization.

    Who may act for the owner is the caller's to check.
    """

    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    @staticmethod
    def current(selector: str, organization_id: uuid.UUID | None = None) -> MetadataOverride | None:
        """The override stored for ``selector`` under exactly this owner, from the worker's copy."""
        provider, _, model = selector.partition(":")
        return metadata_override(provider, model, organization_id)

    async def set(
        self, selector: str, override: MetadataOverride, organization_id: uuid.UUID | None = None
    ) -> MetadataOverride:
        """Replace the owner's override of ``selector``; an override with no field set clears it.

        Raises:
            ValueError: ``selector`` is not in ``provider:model`` form.
        """
        provider, model = _split(selector)
        if override == MetadataOverride():
            await self.clear(selector, organization_id)
            return override
        values = {name: getattr(override, name) for name in _FIELDS}
        if override.input_modalities is not None:
            values["input_modalities"] = list(override.input_modalities)
        async with self.uow:
            await ModelMetadataOverrideRepository(self.uow).put(provider, model, organization_id, values)
        _overrides.setdefault((organization_id, provider), {})[model] = override
        return override

    async def clear(self, selector: str, organization_id: uuid.UUID | None = None) -> bool:
        """Drop the owner's override of ``selector``. Whether there was one.

        Raises:
            ValueError: ``selector`` is not in ``provider:model`` form.
        """
        provider, model = _split(selector)
        async with self.uow:
            removed = await ModelMetadataOverrideRepository(self.uow).remove(provider, model, organization_id)
        _overrides.get((organization_id, provider), {}).pop(model, None)
        return removed
