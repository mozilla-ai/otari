"""Inference request settings."""

import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown

# An RFC 9110 field name (a token).
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


class InferenceSettings(BaseModel):
    """How a completion request is sent and retried upstream, deduplicated, streamed and attributed to an end user."""

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

    provider_request_timeout_seconds: Annotated[float, Shown(SettingsGroup.MODELS)] = Field(
        default=600.0,
        gt=0,
        description=(
            "How long a non-streaming provider call may take, in seconds, for providers that take a "
            "per-request timeout. Connecting stays capped at 5 seconds where the provider's SDK allows "
            "it. Without one the Anthropic SDK refuses a non-streaming request with a large max_tokens "
            "before sending it."
        ),
    )

    provider_user_identifier_key: Annotated[str | None, SECRET] = Field(
        default=None,
        min_length=32,
        description=(
            "Key for the digest behind the opaque end-user identifier each provider call carries. Unset, "
            "the digest is unkeyed. Must differ from OTARI_SECRET_KEY and the master key. See "
            "docs/configuration.md#end-user-identifiers."
        ),
    )

    client_country_header: Annotated[str | None, Shown(SettingsGroup.SERVER)] = Field(
        default=None,
        description=(
            "A header a trusted proxy sets to the client's ISO country code, such as Cloudflare's "
            "CF-IPCountry. Read before client_country_database. Set it only when every request reaches "
            "Otari through that proxy, which replaces any value a client sends: otherwise a client "
            "picks its own country."
        ),
    )

    client_country_database: Annotated[str | None, Shown(SettingsGroup.SERVER)] = Field(
        default=None,
        description=(
            "Path to a MaxMind DB country file (GeoLite2 Country, DB-IP Lite Country) in which the "
            "client address is looked up. A replaced file is read again without a restart. Needs the "
            "geoip extra."
        ),
    )

    streaming_first_chunk_hold_ms: Annotated[int, Shown(SettingsGroup.GENERAL)] = Field(
        default=15000,
        ge=0,
        description=(
            "How long in milliseconds the gateway holds a streaming response's status line "
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
            "header."
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

    @field_validator("client_country_header", "client_country_database")
    @classmethod
    def _blank_is_off(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()

    @field_validator("client_country_header")
    @classmethod
    def _validate_client_country_header(cls, value: str | None) -> str | None:
        if value is not None and not _HEADER_NAME.fullmatch(value):
            msg = f"client_country_header must be an HTTP header name, got {value!r}"
            raise ValueError(msg)
        return value
