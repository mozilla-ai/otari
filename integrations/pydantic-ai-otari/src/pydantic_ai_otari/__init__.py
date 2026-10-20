"""Pydantic AI models served through an Otari gateway.

```python
from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel

agent = Agent(OtariModel("anthropic:claude-sonnet-4-6"))
```

`OTARI_API_KEY` is the key, and `OTARI_BASE_URL` a self-hosted gateway's API root
(otari.ai's gateway when unset), unless passed to `OtariProvider` directly.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from httpx import AsyncClient
    from openai import AsyncOpenAI
    from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
    from pydantic_ai.settings import ModelSettings

__all__ = ["DEFAULT_BASE_URL", "OtariModel", "OtariProvider"]

DEFAULT_BASE_URL = "https://api.otari.ai/api/v1"

# Otari names a model `provider:model` (or `provider/model`); the provider part
# says which family's capabilities apply, whatever API Otari reaches it through.
_PROFILES: dict[str, Callable[[str], ModelProfile | None]] = {
    "openai": openai_model_profile,
    "azure": openai_model_profile,
    "anthropic": anthropic_model_profile,
    "bedrock": amazon_model_profile,
    "gemini": google_model_profile,
    "google": google_model_profile,
    "vertexai": google_model_profile,
    "mistral": mistral_model_profile,
    "cohere": cohere_model_profile,
    "deepseek": deepseek_model_profile,
    "groq": groq_model_profile,
    "xai": grok_model_profile,
    "meta": meta_model_profile,
    "moonshotai": moonshotai_model_profile,
    "qwen": qwen_model_profile,
}


class OtariProvider(OpenAIProvider):
    """Provider for an Otari gateway, over its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncClient | None = None,
    ) -> None:
        """Create a provider for an Otari gateway.

        Args:
            base_url: The gateway's API root, ending in `/api/v1`. Defaults to
                `OTARI_BASE_URL`, then to otari.ai's gateway.
            api_key: An Otari API key or user token. Defaults to `OTARI_API_KEY`.
            openai_client: A preconfigured client, used instead of the arguments above.
            http_client: The HTTP client to send requests with.
        """
        if openai_client is not None:
            super().__init__(openai_client=openai_client)
            return
        api_key = api_key or os.getenv("OTARI_API_KEY")
        if not api_key:
            # Refused rather than left to the OpenAI client, which would send OPENAI_API_KEY to the gateway.
            raise UserError("Set the `OTARI_API_KEY` environment variable or pass `api_key` to `OtariProvider`.")
        base_url = base_url or os.getenv("OTARI_BASE_URL") or DEFAULT_BASE_URL
        super().__init__(base_url=base_url, api_key=api_key, http_client=http_client)

    @property
    def name(self) -> str:
        """The provider name Pydantic AI reports on responses and usage."""
        return "otari"

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Describe the model by the family its name starts with, ``provider:model`` or ``provider/model``."""
        provider, separator, model = model_name.partition(":")
        if not separator:
            provider, separator, model = model_name.partition("/")
        profile_for = _PROFILES.get(provider) if separator else None
        profile = (profile_for(model) if profile_for else None) or openai_model_profile(model_name)
        # The request is always Chat Completions, so its JSON schema rules are the default, as LiteLLM's provider does.
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)


class OtariModel(OpenAIChatModel):
    """A model served through an Otari gateway.

    `model_name` is any model the gateway routes: `provider:model` such as
    `openai:gpt-4o`, or the name of a routing policy or alias.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: OtariProvider | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        """Create the model, with an ``OtariProvider`` from the environment unless one is passed."""
        super().__init__(model_name, provider=provider or OtariProvider(), profile=profile, settings=settings)
