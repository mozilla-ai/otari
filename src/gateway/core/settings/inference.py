"""Inference request settings."""

import re
from typing import Annotated, Self

from any_llm import LLMProvider
from pydantic import BaseModel, Field, field_validator, model_validator

from gateway.core.country_codes import COUNTRY_CODES
from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown

# An RFC 9110 field name (a token).
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_COUNTRY_CODE = re.compile(r"[A-Z]{2}")


class InferenceSettings(BaseModel):
    """How a completion request is sent and retried upstream, deduplicated, streamed and attributed to an end user.

    Also which providers the country a request comes from may reach.
    """

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

    provider_supported_countries: Annotated[dict[str, list[str]], OMITTED] = Field(
        default_factory=dict,
        description=(
            "The ISO country codes each provider type serves, such as openai: [US, DE]. A request from "
            "another country, or from one no source knows, skips that provider's candidates. A provider "
            "type with no entry serves every country. Needs client_country_header or "
            "client_country_database."
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

    @field_validator("provider_supported_countries", mode="before")
    @classmethod
    def _normalize_provider_supported_countries(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        known = {member.value for member in LLMProvider}
        normalized: dict[str, list[str]] = {}
        for raw_type, raw_codes in value.items():
            provider_type = str(raw_type).strip().lower()
            if provider_type not in known:
                msg = f"provider_supported_countries: {raw_type!r} is not a provider type"
                raise ValueError(msg)
            if not isinstance(raw_codes, list):
                msg = f"provider_supported_countries.{provider_type} must be a list of country codes"
                raise ValueError(msg)
            codes: list[str] = []
            for raw_code in raw_codes:
                code = str(raw_code).strip().upper()
                if code not in COUNTRY_CODES:
                    msg = (
                        f"provider_supported_countries.{provider_type}: {raw_code!r} is not an ISO 3166-1 "
                        "alpha-2 country code"
                    )
                    raise ValueError(msg)
                if code not in codes:
                    codes.append(code)
            normalized[provider_type] = codes
        return normalized

    @model_validator(mode="after")
    def _country_gate_has_a_source(self) -> Self:
        if (
            self.provider_supported_countries
            and self.client_country_header is None
            and self.client_country_database is None
        ):
            msg = (
                "provider_supported_countries needs client_country_header or client_country_database: "
                "without one no request has a country, and every listed provider would be refused"
            )
            raise ValueError(msg)
        return self
