"""LangChain chat models served through an Otari gateway.

```python
from langchain_otari import ChatOtari

llm = ChatOtari(model="anthropic:claude-sonnet-4-6")
```

`OTARI_API_KEY` is the key, and `OTARI_BASE_URL` a self-hosted gateway's API root
(otari.ai's gateway when unset), unless passed as `base_url` and `api_key`.
LangGraph takes the model like any other chat model.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import model_validator

__all__ = ["DEFAULT_BASE_URL", "ChatOtari"]

DEFAULT_BASE_URL = "https://api.otari.ai/api/v1"


class ChatOtari(ChatOpenAI):
    """A chat model served through an Otari gateway, over its OpenAI-compatible API.

    `model` is any model the gateway routes: `provider:model` such as
    `openai:gpt-4o`, or the name of a routing policy or alias. Every other
    `ChatOpenAI` argument works unchanged.
    """

    @model_validator(mode="before")
    @classmethod
    def _otari_environment(cls, values: Any) -> Any:
        """Fill ``base_url`` and ``api_key`` from the Otari variables, before ChatOpenAI looks for its own."""
        if not isinstance(values, dict):
            return values
        if not (values.get("base_url") or values.get("openai_api_base")):
            values["base_url"] = os.getenv("OTARI_BASE_URL") or DEFAULT_BASE_URL
        if not (values.get("api_key") or values.get("openai_api_key")):
            api_key = os.getenv("OTARI_API_KEY")
            if not api_key:
                # Refused rather than left to ChatOpenAI, which would send OPENAI_API_KEY to the gateway.
                raise ValueError("Set the `OTARI_API_KEY` environment variable or pass `api_key` to `ChatOtari`.")
            values["api_key"] = api_key
        # Otari reports usage on streams; ChatOpenAI only asks for it from OpenAI itself.
        values.setdefault("stream_usage", True)
        return values

    @property
    def _llm_type(self) -> str:
        """Identify the model as Otari's in callbacks and traces."""
        return "otari-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Name the variable a serialized model takes its key from."""
        return {"openai_api_key": "OTARI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the namespace this class serializes under."""
        return ["langchain_otari", "chat_models"]
