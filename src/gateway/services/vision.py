"""Caption images with a vision model so text-only models can 'see' them.

When the target model can't process images natively, the content normalizer
calls :func:`describe_image` to turn each image into a text description via a
configured vision model (``config.vision_describe_model``). That model may be a
local one (e.g. ``ollama/qwen2-vl``) to keep captioning free, or a frontier
model for best quality.
"""

from __future__ import annotations

from any_llm.types.completion import CompletionUsage

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.inference import chat_user_fields
from gateway.services.provider_kwargs import require_declared_credential, resolve_provider_selector
from gateway.services.providers import CountryGate, acompletion

_DESCRIBE_PROMPT = (
    "You are assisting a text-only language model that cannot see images. "
    "Describe the following image thoroughly and objectively so the model can "
    "reason about it: transcribe any visible text verbatim, and describe layout, "
    "charts, diagrams, tables, and notable visual details. Reply with the "
    "description only."
)


async def describe_image(
    config: GatewayConfig, image_data_url: str, *, upstream_user: str | None, client_country: str | None = None
) -> tuple[str | None, CompletionUsage | None]:
    """Caption an image, returning ``(description, usage)``.

    ``upstream_user`` is the end-user identifier of the request the image came
    from, which the call carries in the field the vision model's provider reads.
    ``client_country`` is the country that request comes from: a vision model
    whose provider ``provider_supported_countries`` closes to it is not called.

    ``description`` is ``None`` (no configured model, or a failed side-call) to
    signal the caller to fall back to its configured strategy (OCR or
    drop-with-log) rather than failing the user's request. ``usage`` is the
    describe model's token usage when the call succeeded, so the caller can
    meter and bill this side-call against the user's budget; it is ``None`` when
    no call was made or the provider returned no usage.
    """
    model = config.vision_describe_model
    if not model:
        return None, None

    resolved = resolve_provider_selector(config, model)
    if CountryGate(client_country).refuses(config, resolved.instance):
        logger.info("vision describe via %s skipped: its provider does not serve the request's country", model)
        return None, None
    provider, model_name = resolved.provider, resolved.model
    provider_kwargs = resolved.kwargs
    user_fields = chat_user_fields(provider, upstream_user) if upstream_user is not None else {}

    try:
        require_declared_credential(config, resolved.instance, provider, provider_kwargs)
        completion = await acompletion(
            model=model_name,
            provider=provider,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DESCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=config.vision_describe_max_tokens,
            **provider_kwargs,
            **user_fields,
        )
    except Exception as exc:  # noqa: BLE001 — captioning is best-effort
        logger.warning("vision describe via %s failed: %s", model, exc)
        return None, None

    usage = getattr(completion, "usage", None)
    try:
        content = completion.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError):
        return None, usage
    if isinstance(content, str) and content.strip():
        return content.strip(), usage
    return None, usage
