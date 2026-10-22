"""A catalog model's token limits and capabilities, from the live models.dev catalog and the pricing snapshot."""

from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, Field

from gateway.core.config import GatewayConfig
from gateway.services.catalog._metadata import metadata_entry
from gateway.services.model_catalog_service import ModelCatalogEntry
from gateway.services.pricing_service import model_context_window

if TYPE_CHECKING:
    from gateway.services.merged_catalog_service import ModelObject


class ModelArchitecture(BaseModel):
    """What a model reads and writes, in OpenRouter's ``architecture`` shape."""

    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)


class ModelLimits(NamedTuple):
    """A model's token limits, ``None`` where neither dataset knows one."""

    context_window: int | None
    max_output_tokens: int | None


class ModelCapabilities(NamedTuple):
    """What models.dev says a model is called and accepts, in OpenRouter's listing fields."""

    name: str | None
    supported_parameters: list[str]
    input_modalities: list[str]
    output_modalities: list[str]


def _entry(config: GatewayConfig, models_dev: dict[str, Any] | None, model_key: str) -> ModelCatalogEntry | None:
    """models.dev's entry for a ``provider:model`` key; a bare key names no provider to look it up under."""
    instance, separator, model_id = model_key.partition(":")
    if not separator:
        return None
    return metadata_entry(models_dev, config.provider_instance_type(instance), model_id)


def model_limits(config: GatewayConfig, models_dev: dict[str, Any] | None, model_key: str) -> ModelLimits:
    """The token limits of a ``provider:model`` (or bare) key.

    The live models.dev catalog first, because it describes the model as that
    provider serves it, then the pricing snapshot for the context window. The
    snapshot keeps no output limit, so without a live entry there is none. A bare
    key names no provider to look either up under, so it has no limits.
    """
    instance, separator, model_id = model_key.partition(":")
    if not separator:
        return ModelLimits(model_context_window(None, model_key), None)
    metadata = _entry(config, models_dev, model_key)
    return ModelLimits(
        context_window=(metadata.context_window if metadata else None) or model_context_window(instance, model_id),
        max_output_tokens=metadata.max_output_tokens if metadata else None,
    )


def model_capabilities(
    config: GatewayConfig, models_dev: dict[str, Any] | None, model_key: str
) -> ModelCapabilities | None:
    """A model's capabilities, or ``None`` when models.dev does not describe it.

    ``supported_parameters`` names only the request parameters models.dev records
    a flag for, so it is not the full set a model accepts: ``tools`` and
    ``tool_choice`` for ``tool_call``, ``response_format`` for ``structured_output``,
    ``reasoning`` for ``reasoning``.
    """
    metadata = _entry(config, models_dev, model_key)
    if metadata is None:
        return None
    flags = (
        (metadata.tool_call, ("tools", "tool_choice")),
        (metadata.structured_output, ("response_format",)),
        (metadata.reasoning, ("reasoning",)),
    )
    return ModelCapabilities(
        name=metadata.name,
        supported_parameters=[parameter for flag, parameters in flags if flag for parameter in parameters],
        input_modalities=metadata.input_modalities,
        output_modalities=metadata.output_modalities,
    )


def apply_model_metadata(
    config: GatewayConfig,
    models_dev: dict[str, Any] | None,
    obj: "ModelObject",
    metadata_selector: str | None = None,
    *,
    with_name: bool = True,
) -> None:
    """Fill ``obj``'s token limits and capabilities from ``metadata_selector``, or from its own id.

    ``with_name=False`` leaves the display name unset, for an alias whose target's
    name would say what the alias hides.
    """
    selector = metadata_selector or obj.id
    limits = model_limits(config, models_dev, selector)
    obj.context_window = limits.context_window
    obj.max_output_tokens = limits.max_output_tokens
    capabilities = model_capabilities(config, models_dev, selector)
    if capabilities is None:
        obj.name = obj.supported_parameters = obj.architecture = None
        return
    obj.name = capabilities.name if with_name else None
    obj.supported_parameters = capabilities.supported_parameters
    obj.architecture = ModelArchitecture(
        input_modalities=capabilities.input_modalities, output_modalities=capabilities.output_modalities
    )
