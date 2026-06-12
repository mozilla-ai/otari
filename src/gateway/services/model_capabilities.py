"""Resolve whether a target model natively understands images / documents.

The content normalizer uses this to decide, per request, whether to forward a
file/image content block to the provider (native) or extract it to text
(text-only). Getting the *default* right matters: a wrong "native" verdict for a
text-only local model means the block is silently dropped and the model answers
as if no file were attached.

Resolution order (most → least authoritative):

1. **Operator override** — ``config.model_capabilities[provider/model]`` (or the
   bare model key). The only fully reliable signal for local models.
2. **any-llm provider metadata** — trusted only for hosted providers. Local /
   OpenAI-compatible servers (ollama, vllm, llamacpp, lmstudio) set the flags on
   a shared base class and over-report (e.g. Ollama claims PDF support but its
   adapter silently drops document blocks), so we ignore metadata there and fall
   through to the safe default.
3. **Default** — ``False`` (extract). A needless extraction still yields a
   correct answer; a wrong passthrough does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from any_llm import AnyLLM, LLMProvider

from gateway.core.config import GatewayConfig
from gateway.log_config import logger

# Providers whose any-llm capability flags over-report because they are set on a
# shared base class rather than derived from the actually-served model. For
# these, native passthrough must be opted into per-model via model_capabilities.
_LOCAL_PROVIDERS: frozenset[LLMProvider] = frozenset(
    {
        LLMProvider.OLLAMA,
        LLMProvider.VLLM,
        LLMProvider.LLAMACPP,
        LLMProvider.LMSTUDIO,
    }
)


@dataclass(frozen=True)
class Capabilities:
    """Whether the target model natively understands each modality."""

    image: bool
    pdf: bool
    source: str  # "config" | "provider_metadata" | "default"


def resolve_capabilities(
    config: GatewayConfig,
    provider: LLMProvider,
    model: str,
) -> Capabilities:
    """Resolve native image/pdf support for ``provider/model``."""
    # Accept both separators: model selectors and the pricing config use
    # ``provider:model`` (colon), while ``provider/model`` (slash) is also common.
    # The bare model key is the final fallback.
    for key in (f"{provider.value}:{model}", f"{provider.value}/{model}", model):
        override = config.model_capabilities.get(key)
        if override is not None:
            return Capabilities(override.supports_image, override.supports_pdf, "config")

    if provider not in _LOCAL_PROVIDERS:
        try:
            meta = AnyLLM.get_provider_class(provider).get_provider_metadata()
            return Capabilities(meta.image, meta.pdf, "provider_metadata")
        except Exception as exc:  # noqa: BLE001 — metadata lookup is best-effort
            logger.debug("provider metadata lookup failed for %s: %s", provider.value, exc)

    return Capabilities(image=False, pdf=False, source="default")
