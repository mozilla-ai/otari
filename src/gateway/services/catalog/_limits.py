"""A catalog model's token limits: the live models.dev catalog first, then the pricing snapshot's context window."""

from typing import TYPE_CHECKING, Any, NamedTuple

from gateway.core.config import GatewayConfig
from gateway.services.catalog._metadata import metadata_entry
from gateway.services.pricing_service import model_context_window

if TYPE_CHECKING:
    from gateway.services.merged_catalog_service import ModelObject


class ModelLimits(NamedTuple):
    """A model's token limits, ``None`` where neither dataset knows one."""

    context_window: int | None
    max_output_tokens: int | None


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
    metadata = metadata_entry(models_dev, config.provider_instance_type(instance), model_id)
    return ModelLimits(
        context_window=(metadata.context_window if metadata else None) or model_context_window(instance, model_id),
        max_output_tokens=metadata.max_output_tokens if metadata else None,
    )


def apply_model_limits(
    config: GatewayConfig, models_dev: dict[str, Any] | None, obj: "ModelObject", limits_selector: str | None = None
) -> None:
    """Fill ``obj``'s token limits from ``limits_selector``, or from its own id."""
    limits = model_limits(config, models_dev, limits_selector or obj.id)
    obj.context_window = limits.context_window
    obj.max_output_tokens = limits.max_output_tokens
