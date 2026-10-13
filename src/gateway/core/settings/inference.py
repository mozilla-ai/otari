"""Inference request settings."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown


class InferenceSettings(BaseModel):
    """How a completion request is retried upstream, deduplicated, streamed and attributed to an end user."""

    provider_max_retries: Annotated[int | None, Shown(SettingsGroup.MODELS)] = Field(
        default=None,
        ge=0,
        description=(
            "How many times a provider client retries a failed call (a 429, a 5xx or a dropped connection) "
            "before Otari sees the failure. Unset leaves each provider SDK's own default. A provider "
            "instance's max_retries overrides it. Applies to the OpenAI-compatible, Anthropic, Groq, "
            "Cerebras, Together, Gemini and Vertex AI clients; other providers keep their SDK's behavior."
        ),
    )

    provider_user_identifier_enabled: Annotated[bool, Shown(SettingsGroup.MODELS)] = Field(
        default=True,
        description=(
            "Whether each provider call carries an opaque identifier of the end user behind it. A startup "
            "setting: the dashboard shows it but cannot change it. See "
            "docs/configuration.md#end-user-identifiers."
        ),
    )
    provider_user_identifier_key: Annotated[str | None, SECRET] = Field(
        default=None,
        min_length=32,
        description=(
            "Key for the digest behind provider_user_identifier_enabled. Unset, the digest is unkeyed. "
            "Must differ from OTARI_SECRET_KEY and the master key."
        ),
    )

    streaming_first_chunk_hold_ms: Annotated[int, Shown(SettingsGroup.GENERAL)] = Field(
        default=15000,
        ge=0,
        description=(
            "How long in milliseconds a standalone gateway holds a streaming response's status line "
            "for the provider's first chunk, so a provider failure before any output is answered with "
            "its own HTTP status (and falls over in a routing policy) rather than as an error event "
            "after a 200. A provider slower than this still streams: the response is committed and "
            "keeps waiting. Streams that run gateway tools are not held. 0 commits the response as "
            "soon as the stream opens."
        ),
    )

    idempotency_retention_sec: Annotated[int, OMITTED] = Field(
        default=86400,
        ge=0,
        description=(
            "How long a non-streaming completion sent with an Idempotency-Key is kept, so that a "
            "retry with the same key returns the stored response instead of calling the provider "
            "and billing again. The stored response includes the generated content. 0 ignores the "
            "header. Standalone mode only."
        ),
    )
    idempotency_lease_sec: Annotated[int, OMITTED] = Field(
        default=60,
        ge=3,
        description=(
            "How long a claim on an Idempotency-Key stays valid without being renewed. The request "
            "holding it renews it every third of this while it runs, so this bounds how long a key "
            "stays blocked after the worker running its request dies, not how long a request may take."
        ),
    )
    idempotency_sweep_interval_sec: Annotated[int, OMITTED] = Field(
        default=3600,
        ge=1,
        description="How often expired idempotency records are deleted.",
    )
