"""A catalog model's token limits and capabilities.

From what the engine serving it says (``_served``), then the live models.dev
catalog, then the pricing snapshot.
"""

import uuid
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, Field

from gateway.core.config import GatewayConfig
from gateway.services.catalog._metadata import metadata_entry
from gateway.services.catalog._model_type import model_type
from gateway.services.catalog._served import ServedFacts, served_facts
from gateway.services.model_catalog_service import ModelCatalogEntry
from gateway.services.pricing_service import model_context_window
from gateway.types.models_dev import ModelsDevCatalog

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


# Where models.dev lists open-weight models by their Hugging Face ids, which is
# what a vLLM or SGLang server loaded and reports as ``root``.
_OPEN_WEIGHTS_PROVIDER = "huggingface"


def _served(config: GatewayConfig, model_key: str, organization_id: uuid.UUID | None) -> ServedFacts | None:
    """What the engine serving a ``provider:model`` key said of it, as the caller reaches it.

    In the order a request resolves the prefix: a configured instance, which no
    organization key ever shadows, then the caller's organization's own key,
    then the deployment's hosted provider.
    """
    instance, separator, model_id = model_key.partition(":")
    if not separator:
        return None
    if organization_id is not None and instance not in config.providers:
        own = served_facts(instance, model_id, organization_id)
        if own is not None:
            return own
    return served_facts(instance, model_id)


def _entry(
    config: GatewayConfig, models_dev: ModelsDevCatalog | None, model_key: str, served: ServedFacts | None
) -> ModelCatalogEntry | None:
    """models.dev's entry for a ``provider:model`` key; a bare key names no provider to look it up under.

    A model an engine serves is looked up by its Hugging Face id when its
    provider has no entry for it, which a self-hosted provider never does.
    """
    instance, separator, model_id = model_key.partition(":")
    if not separator:
        return None
    entry = metadata_entry(models_dev, config.provider_instance_type(instance), model_id)
    if entry is not None or served is None:
        return entry
    for name in dict.fromkeys((model_id, served.root)):
        if name and (entry := metadata_entry(models_dev, _OPEN_WEIGHTS_PROVIDER, name)) is not None:
            return entry
    return None


def _sends_response_format(provider_type: str, model_id: str) -> bool:
    """Whether any-llm passes ``response_format`` to this provider's model rather than refusing it.

    Its Bedrock provider sends it only to Anthropic models, as a forced tool
    call, whatever models.dev says the model can do.
    """
    return provider_type != "bedrock" or "anthropic." in model_id


def model_limits(
    config: GatewayConfig,
    models_dev: ModelsDevCatalog | None,
    model_key: str,
    organization_id: uuid.UUID | None = None,
) -> ModelLimits:
    """The token limits of a ``provider:model`` (or bare) key, as a caller in ``organization_id`` reaches it.

    The engine serving the model first, because it reports the context length
    it actually runs the model at. Then the live models.dev catalog, because it
    describes the model as that provider serves it, then the pricing snapshot
    for the context window. The snapshot keeps no output limit, so without a
    live entry there is none, and no output limit exceeds what the engine
    serves. A bare key names no provider to look any up under.
    """
    instance, separator, model_id = model_key.partition(":")
    if not separator:
        return ModelLimits(model_context_window(None, model_key), None)
    served = _served(config, model_key, organization_id)
    metadata = _entry(config, models_dev, model_key, served)
    served_window = served.context_window if served else None
    max_output = metadata.max_output_tokens if metadata else None
    if max_output is not None and served_window is not None:
        max_output = min(max_output, served_window)
    return ModelLimits(
        context_window=served_window
        or (metadata.context_window if metadata else None)
        or model_context_window(instance, model_id),
        max_output_tokens=max_output,
    )


def served_context_window(provider: str | None, model: str, organization_id: uuid.UUID | None = None) -> int | None:
    """A model's context window as its engine serves it, else as the pricing snapshot gives it.

    ``organization_id``'s own key is read first. Unlike a listing this needs no
    configured instance check: a key's prefix never names one.
    """
    served = None
    if provider and organization_id is not None:
        served = served_facts(provider, model, organization_id)
    if provider and served is None:
        served = served_facts(provider, model)
    if served is not None and served.context_window is not None:
        return served.context_window
    return model_context_window(provider, model)


def model_capabilities(
    config: GatewayConfig,
    models_dev: ModelsDevCatalog | None,
    model_key: str,
    organization_id: uuid.UUID | None = None,
) -> ModelCapabilities | None:
    """A model's capabilities, or ``None`` when neither models.dev nor its engine describes it.

    ``supported_parameters`` names only the request parameters models.dev records
    a flag for, so it is not the full set a model accepts: ``tools`` and
    ``tool_choice`` for ``tool_call``, ``response_format`` for ``structured_output``,
    ``reasoning`` for ``reasoning``. ``response_format`` is left out where any-llm
    refuses it for the provider, since a request carrying it would fail.

    An engine that names its parsers overrides two of those flags: it calls
    tools only with a tool-call parser, whatever the model can do, and a
    reasoning parser means the model reasons. What it says it reads overrides
    models.dev's input modalities, whether or not it named its parsers.
    """
    served = _served(config, model_key, organization_id)
    metadata = _entry(config, models_dev, model_key, served)
    settings = served if served is not None and served.settings_reported else None
    served_modalities = served.input_modalities if served is not None else None
    if metadata is None and settings is None and served_modalities is None:
        return None
    tool_call = metadata.tool_call if metadata else None
    reasoning = metadata.reasoning if metadata else None
    instance, _, model_id = model_key.partition(":")
    structured_output = metadata.structured_output if metadata else None
    if not _sends_response_format(config.provider_instance_type(instance), model_id):
        structured_output = False
    input_modalities = (
        list(served_modalities) if served_modalities else (metadata.input_modalities if metadata else ["text"])
    )
    if settings is not None:
        tool_call = settings.tool_call_parser is not None
        reasoning = reasoning or settings.reasoning_parser is not None
    flags = (
        (tool_call, ("tools", "tool_choice")),
        (structured_output, ("response_format",)),
        (reasoning, ("reasoning",)),
    )
    return ModelCapabilities(
        name=metadata.name if metadata else None,
        supported_parameters=[parameter for flag, parameters in flags if flag for parameter in parameters],
        input_modalities=input_modalities,
        output_modalities=metadata.output_modalities if metadata else ["text"],
    )


def apply_model_metadata(
    config: GatewayConfig,
    models_dev: ModelsDevCatalog | None,
    obj: "ModelObject",
    metadata_selector: str | None = None,
    *,
    with_name: bool = True,
    organization_id: uuid.UUID | None = None,
) -> None:
    """Fill ``obj``'s token limits and capabilities from ``metadata_selector``, or from its own id.

    ``with_name=False`` leaves the display name unset, for an alias whose target's
    name would say what the alias hides. ``organization_id`` is the caller's,
    whose own provider keys describe the models it reaches on them.
    """
    selector = metadata_selector or obj.id
    limits = model_limits(config, models_dev, selector, organization_id)
    obj.context_window = limits.context_window
    obj.max_output_tokens = limits.max_output_tokens
    obj.type = model_type(config, models_dev, selector)
    capabilities = model_capabilities(config, models_dev, selector, organization_id)
    if capabilities is None:
        obj.name = obj.supported_parameters = obj.architecture = None
        return
    obj.name = capabilities.name if with_name else None
    obj.supported_parameters = capabilities.supported_parameters
    obj.architecture = ModelArchitecture(
        input_modalities=capabilities.input_modalities, output_modalities=capabilities.output_modalities
    )
