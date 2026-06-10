"""Caption images with a vision model so text-only models can 'see' them.

When the target model can't process images natively, the content normalizer
calls :func:`describe_image` to turn each image into a text description via a
configured vision model (``config.vision_describe_model``). That model may be a
local one (e.g. ``ollama/qwen2-vl``) to keep captioning free, or a frontier
model for best quality.
"""

from __future__ import annotations

from any_llm import AnyLLM, acompletion

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.provider_kwargs import get_provider_kwargs

_DESCRIBE_PROMPT = (
    "You are assisting a text-only language model that cannot see images. "
    "Describe the following image thoroughly and objectively so the model can "
    "reason about it: transcribe any visible text verbatim, and describe layout, "
    "charts, diagrams, tables, and notable visual details. Reply with the "
    "description only."
)


async def describe_image(config: GatewayConfig, image_data_url: str) -> str | None:
    """Return a text description of an image, or ``None`` if unavailable.

    ``None`` (no configured model, or a failed side-call) signals the caller to
    fall back to its configured strategy (OCR or drop-with-log) rather than
    failing the user's request.
    """
    model = config.vision_describe_model
    if not model:
        return None

    provider, model_name = AnyLLM.split_model_provider(model)
    provider_kwargs = get_provider_kwargs(config, provider)

    try:
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
            **provider_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — captioning is best-effort
        logger.warning("vision describe via %s failed: %s", model, exc)
        return None

    try:
        content = completion.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError):
        return None
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None
