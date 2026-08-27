import base64
import binascii
import os
import re
import types
import typing
from collections.abc import Container
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

import yaml
from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.core.addresses import normalized_address
from gateway.core.env import otari_env
from gateway.log_config import logger
from gateway.models.routing import RoutingConfig

API_KEY_HEADER = "Otari-Key"
# Aliases accepted for a provider instance's ``provider_type`` that map onto a
# real any-llm implementation. The "openai-compatible" spelling mirrors the
# naming opencode / pi use for self-hosted OpenAI-compatible backends.
PROVIDER_TYPE_ALIASES = {
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "anthropic-compatible": "anthropic",
    "anthropic_compatible": "anthropic",
}
X_API_KEY_HEADER = "x-api-key"  # Anthropic-native clients send credentials here (no Bearer prefix).

# The OAuth providers a deployment may configure dashboard sign-in with, in the
# spelling that appears in a config key (``oauth_google_client_id``), on the
# wire (``GET /v1/bootstrap``'s ``oauth_providers``, the route path segment),
# and in the ``user.oauth_provider`` column. One vocabulary rather than four,
# and it lives here because the config fields are what make a provider real on
# a deployment; ``services.oauth_service`` holds what each one means.
#
# Not an enum: the column stores a plain string so an overlay binding its own
# ``IdentityProviderPort`` can record a connection this tuple never named, and
# a closed enum here would make that value unrepresentable.
OAUTH_PROVIDERS: tuple[str, ...] = ("github", "google")
# Per-request opt-out for a policy's learned router: "off" serves the policy's
# default target and skips the router entirely. There is no "force on": the
# router is enabled by the policy, not by the caller.
ROUTER_HEADER = "Otari-Router"
# Stable per-conversation id for trace-sticky routing. When set, it is the trace
# identity (namespaced per user); absent, the router falls back to hashing the
# conversation's opening messages, which cannot tell apart two conversations that
# open identically. See docs/routing-scaling.md.
CONVERSATION_HEADER = "Otari-Conversation-Id"
# Routing-memory partition (use case / category) for this request. When set, the
# router votes only over records carrying the same task label and stays in
# pass-through until that partition alone is warm; records from other tasks never
# influence it. Submit the matching label via the /rank task_id.
ROUTER_TASK_HEADER = "Otari-Router-Task"
DEFAULT_PLATFORM_BASE_URL = "https://api.otari.ai/api/v1"
# Where a hybrid gateway's control plane lives for a person, as opposed to
# DEFAULT_PLATFORM_BASE_URL above, which is where it lives for the gateway. The
# deployment bootstrap hands this to the browser so a hybrid landing page can
# link to the dashboard that actually manages this gateway; an operator pointed
# at a staging platform overrides it with platform.management_url.
DEFAULT_PLATFORM_MANAGEMENT_URL = "https://otari.ai"
PLATFORM_TOKEN_ENV_VAR = "OTARI_AI_TOKEN"
# User-facing config env vars use the OTARI_ prefix (e.g. OTARI_MASTER_KEY,
# OTARI_PORT), which is also the native pydantic prefix below.
OTARI_ENV_PREFIX = "OTARI_"
# Full structured config supplied through the environment, for PaaS platforms
# (Railway, Render, Fly.io, Kubernetes) where mounting a config.yml is awkward.
# These carry the entire YAML schema (providers, pricing, etc.), not just the
# scalar fields reachable via OTARI_<FIELD>. Raw YAML wins when both are set.
OTARI_CONFIG_YAML_ENV = "OTARI_CONFIG_YAML"
OTARI_CONFIG_B64_ENV = "OTARI_CONFIG_B64"
# GatewayConfig fields promoted from ad hoc otari_env() reads in route/service
# code. The read sites still consult otari_env() (they have no config object in
# scope), so load_config bridges values set in the YAML config into the process
# environment for these fields; without the bridge a YAML-set value would
# validate at startup and then be silently ignored at request time. Each field
# name maps onto its OTARI_<FIELD> environment variable.
ENV_BRIDGED_FIELDS = (
    "sandbox_url",
    "guardrails_url",
    "tools_header",
    "sandbox_purpose_hint",
    "sandbox_session_image",
    "sandbox_allowed_session_images",
    "web_search_url",
    "web_search_purpose_hint",
    "web_search_engines",
    "web_search_max_results",
    "web_search_extract",
    "web_search_allow_private_hosts",
    "mcp_allow_loopback",
    "mcp_allow_private_hosts",
    "provider_allow_private_hosts",
)


# Allowed values for the enum config fields. Defined once so the field
# validators and the runtime-settings layer (which lets the dashboard hot-change
# these) agree on the accepted set.
STREAM_MISSING_USAGE_POLICIES = ("estimate", "fail", "allow_free")
VISION_STRATEGIES = ("describe", "ocr", "off")
ROUTER_GRANULARITIES = ("trace_sticky", "step")
# Selectable mail transports, plus the two states that are not a transport:
# "auto" derives one from whether SMTP is configured, "none" turns mail off
# even when it is. See GatewayConfig.mail_transport.
MAIL_TRANSPORT_SETTINGS = ("auto", "smtp", "console", "none")

# Search providers the standalone POST /v1/search endpoint can dispatch to.
# Declared here rather than in the adapter module so startup validation can
# reject an unknown ``search_tools.<name>.provider`` without the config layer
# importing the service layer.
SEARCH_PROVIDERS = ("exa", "searxng")
# Providers that authenticate with an API key, so a tool declaring one of them
# without a key is a misconfiguration. A SearXNG-shaped backend is normally
# keyless (the bundled container, a self-hosted adapter), which is why the key
# is per-provider rather than universally required.
SEARCH_PROVIDERS_REQUIRING_API_KEY = ("exa",)
# Providers with no endpoint of their own to default to, so the tool has to say
# where the backend is. The only one today is ``searxng``, which speaks the same
# wire contract as the in-loop otari_web_search backend and therefore inherits
# ``web_search_url`` when the tool declares no ``api_base``.
SEARCH_PROVIDERS_REQUIRING_API_BASE = ("searxng",)


def validate_search_tool_entry(name: str, entry: Any) -> None:
    """Validate one ``search_tools`` entry, raising ``ValueError`` on any problem.

    Module-level rather than a method so the runtime CRUD path
    (``/v1/search-tools``) can hold a dashboard-written tool to the same rules
    the config file is held to at startup, instead of restating them.

    A tool on a provider that authenticates with an API key is rejected here
    without one, rather than at request time as an opaque upstream 401; a keyless
    provider (a self-hosted SearXNG or an adapter fronting one) is allowed to
    declare none. The tool name doubles as a ``/v1/search/{tool}`` path segment,
    so it must not contain a slash.

    A missing backend URL is deliberately not fatal here; see
    :meth:`GatewayConfig.search_tools_without_backend_url`.
    """
    if not name:
        msg = "search tool name must not be empty."
        raise ValueError(msg)
    if "/" in name:
        msg = f"search tool name '{name}' must not contain '/' (it is used as a URL path segment)."
        raise ValueError(msg)
    if not isinstance(entry, dict):
        msg = f"search_tools.{name} must be a mapping."
        raise ValueError(msg)
    provider = entry.get("provider") or name
    if provider not in SEARCH_PROVIDERS:
        msg = (
            f"search_tools.{name}.provider '{provider}' is not a supported search provider "
            f"(one of: {', '.join(SEARCH_PROVIDERS)})."
        )
        raise ValueError(msg)
    if provider in SEARCH_PROVIDERS_REQUIRING_API_KEY and not entry.get("api_key"):
        msg = f"search_tools.{name}.api_key is required for provider '{provider}'."
        raise ValueError(msg)
    timeout = entry.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            msg = f"search_tools.{name}.timeout must be a number of seconds."
            raise ValueError(msg)
        # A negative timeout would reach httpx and fail at request time, and a
        # zero is silently swapped for the default when the tool is resolved.
        # Both are misconfigurations worth failing on here.
        if timeout <= 0:
            msg = f"search_tools.{name}.timeout must be greater than 0 seconds, got {timeout}."
            raise ValueError(msg)
    options = entry.get("options")
    if options is not None and not isinstance(options, dict):
        msg = f"search_tools.{name}.options must be a mapping."
        raise ValueError(msg)


class _NonScalarField(Exception):
    """Raised when a config field is not a simple scalar settable from a plain env string."""


def _get_platform_token_from_env() -> str | None:
    token = os.getenv(PLATFORM_TOKEN_ENV_VAR, "").strip()
    return token or None


def provider_credential_env_names(provider_type: str) -> tuple[str, ...] | None:
    """Environment variables any-llm reads for a provider's credential.

    Returns an empty tuple when the provider needs no API key: the keyless local
    backends (ollama, llamacpp, llamafile) declare the literal string ``"None"``,
    and a provider authenticating through a cloud SDK (Vertex AI) declares an
    empty name. Returns ``None`` when the provider cannot be inspected at all
    (not a known implementation, or an optional SDK dependency that is not
    installed), so callers can stay lenient about what they cannot determine.

    The declared value is a label rather than always a single variable name:
    providers with alternatives join them with ``/`` (gemini's
    ``"GEMINI_API_KEY/GOOGLE_API_KEY"``) and providers needing several parts join
    them with ``and`` (sagemaker), so it is split into candidate names.
    """
    try:
        declared = AnyLLM.get_provider_class(provider_type).ENV_API_KEY_NAME
    except (AnyLLMError, ImportError, AttributeError) as exc:
        logger.debug("no credential env var known for provider type %r: %s", provider_type, exc)
        return None
    if not declared or declared == "None":
        return ()
    candidates = (part.strip() for chunk in declared.split("/") for part in chunk.split(" and "))
    return tuple(candidate for candidate in candidates if candidate)


class PricingTierConfig(BaseModel):
    """One whole-request context threshold price rule from configuration."""

    min_input_tokens: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_1h_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTierConfig":
        rates = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.cache_read_price_per_million,
            self.cache_write_price_per_million,
            self.cache_write_1h_price_per_million,
        )
        if all(rate is None for rate in rates):
            raise ValueError("pricing tier must override at least one price field")
        return self


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    cache_read_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M cached-input tokens (OpenAI/Gemini discount rate or Anthropic cache-read rate).",
    )
    cache_write_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M cache-write (creation) tokens. Anthropic only.",
    )
    cache_write_1h_price_per_million: float | None = Field(
        default=None,
        ge=0,
        description="Price per 1M Anthropic 1-hour cache-write tokens.",
    )
    pricing_tiers: list[PricingTierConfig] = Field(
        default_factory=list,
        description="Whole-request context threshold pricing rules.",
    )
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )

    @model_validator(mode="after")
    def validate_unique_tier_thresholds(self) -> "PricingConfig":
        thresholds = [tier.min_input_tokens for tier in self.pricing_tiers]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("pricing_tiers must not repeat min_input_tokens")
        return self


class ModelCapabilityConfig(BaseModel):
    """Per-model multimodal capability override.

    any-llm exposes provider-class-level ``SUPPORTS_COMPLETION_IMAGE`` /
    ``SUPPORTS_COMPLETION_PDF`` flags, but those are set on the OpenAI-compatible
    base class and so over-report for text-only local models served behind an
    OpenAI-compatible endpoint (vLLM, llama.cpp, LM Studio). This map lets an
    operator state the truth per ``provider/model`` key so the content
    normalizer extracts files to text instead of forwarding blocks the model
    silently drops. See gateway.services.model_capabilities.
    """

    supports_image: bool = Field(
        default=False,
        description="Model can natively understand image content blocks (vision).",
    )
    supports_pdf: bool = Field(
        default=False,
        description="Model can natively understand PDF/document content blocks.",
    )


def _host_of(url: str | None) -> str:
    """The bare hostname of an absolute URL, or "" if there isn't one.

    Port and scheme are dropped: a WebAuthn relying-party ID is a domain, not an
    origin. ``urlsplit().hostname`` also lowercases and strips the brackets off
    an IPv6 literal, which is the comparison form a browser uses.
    """
    if not url:
        return ""
    return urlsplit(url.strip()).hostname or ""


class RelyingParty(NamedTuple):
    """The WebAuthn relying party a deployment presents itself as.

    Resolved once from configuration (``GatewayConfig.webauthn_relying_party``)
    rather than per request, so every ceremony on a deployment agrees on the ID
    a passkey is bound to. See that property for why it is not read off the
    request.
    """

    rp_id: str
    name: str
    origins: tuple[str, ...]

    def covers(self, origin: str) -> bool:
        """Whether ``origin``'s host is the relying-party ID or below it.

        The registrable-suffix rule from the WebAuthn spec, applied to this
        deployment's own configured origins so a typo is caught at startup
        rather than as an unexplained ceremony failure in a browser. The
        boundary check ('otari.example.com'.endswith('.example.com')) is what
        keeps 'notexample.com' from passing as a subdomain of 'example.com'.
        """
        host = _host_of(origin)
        return host == self.rp_id or host.endswith(f".{self.rp_id}")


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="OTARI_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        # Treat an empty OTARI_<FIELD> env var as unset, matching the empty-skip
        # in _apply_otari_env_overrides. Without this, a blank OTARI_MASTER_KEY
        # (common from container templating) would read as "" instead of None,
        # and an empty bearer token would then satisfy is_valid_master_key.
        env_ignore_empty=True,
    )

    database_url: str = Field(
        default="sqlite:///./otari.db",
        description="Database connection URL (SQLite default for local use; PostgreSQL recommended for production)",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Automatically run database migrations on startup",
    )
    db_pool_size: int = Field(
        default=10,
        ge=1,
        description="Number of persistent connections in the DB pool per worker.",
    )
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        description="Extra connections the pool can open above db_pool_size during bursts.",
    )
    db_pool_timeout: float = Field(
        default=30.0,
        ge=0,
        description="Seconds to wait for an available connection before raising TimeoutError.",
    )
    db_pool_recycle: int = Field(
        default=-1,
        description="Recycle connections older than this many seconds. -1 disables.",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")  # noqa: S104
    port: int = Field(default=8000, description="Port to bind the server to")
    master_key: str | None = Field(default=None, description="Master key for protecting management endpoints")
    dashboard_session_ttl_hours: int = Field(
        default=168,
        ge=1,
        description=(
            "How long a dashboard sign-in stays valid, in hours. Signing in to the admin "
            "dashboard exchanges the master key for an HttpOnly session cookie with this "
            "lifetime; the master key itself never expires."
        ),
    )
    activation_guide: bool = Field(
        default=True,
        description=(
            "Offer the dashboard's first-request setup guide in a workspace that has not served "
            "a successful request yet. False turns the flow off for the whole deployment: the "
            "endpoints stay mounted and report every workspace ineligible, so a dashboard that "
            "is already open stops offering it too."
        ),
    )
    rate_limit_rpm: int | None = Field(
        default=None, ge=1, description="Maximum requests per minute per user (None disables rate limiting)"
    )
    dashboard_login_rate_limit_per_minute: int | None = Field(
        default=10,
        ge=1,
        description=(
            "Maximum calls per client IP per minute to the app's unauthenticated "
            "surfaces (None disables this limit): failed POST /v1/auth/session "
            "attempts (a correct master key is never throttled there), every call "
            "to the two public invitation-accept routes, and every call to the "
            "signup, verification and password-reset routes, counted whether they "
            "succeed or fail. Separate from rate_limit_rpm, which is keyed to "
            "authenticated users and does not cover any of these pre-auth paths."
        ),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="Allowed CORS origins (empty list disables CORS)"
    )
    public_base_url: str | None = Field(
        default=None,
        description=(
            "This deployment's own externally-reachable URL, with no trailing slash "
            "(e.g. 'https://otari.example.com'). Used to build absolute links in outgoing "
            "email (an invitation's accept link) and to derive the WebAuthn relying-party "
            "ID a passkey is bound to; nothing else here needs to describe its own "
            "address, since every other reference is relative to the request."
        ),
    )
    docs_url: str | None = Field(
        default=None,
        description=(
            "Where this deployment's documentation lives, as an absolute http(s) URL "
            "(e.g. 'https://docs.otari.ai/en/'). Unset, the dashboard's Documentation links "
            "point at the operator guide bundled with the gateway at /#/docs, which is the "
            "right default for a self-hosted deployment. Set it to retarget those links at "
            "a product documentation site instead; the bundled guide stays reachable at "
            "/#/docs either way."
        ),
    )
    data_plane_url: str | None = Field(
        default=None,
        description=(
            "Where this deployment's inference traffic belongs, as an absolute http(s) URL "
            "with no trailing slash (e.g. 'https://gateway.otari.ai'). Only a hosted control "
            "plane needs it: a standalone gateway and a hybrid one both serve inference at "
            "their own address, so whatever reached the dashboard reaches the API, and this "
            "stays unset. A control plane serving many organizations does not, so it has to "
            "say where the data-plane gateway is or the dashboard has nothing runnable to "
            "hand somebody with a new key. Unlike platform.management_url, which is a human "
            "link target, this is the base URL a client suffixes with /v1."
        ),
    )
    webauthn_rp_id: str | None = Field(
        default=None,
        description=(
            "The WebAuthn relying-party ID passkeys registered here are bound to: a bare "
            "domain with no scheme, port or path (e.g. 'otari.example.com'). Defaults to "
            "the host of public_base_url. Set it explicitly to bind passkeys to a parent "
            "domain of the one serving the dashboard (an 'example.com' passkey works on "
            "'otari.example.com', but not the reverse). Changing it invalidates every "
            "passkey already registered, because an authenticator scopes what it stored "
            "to the ID it stored it under."
        ),
    )
    webauthn_rp_name: str = Field(
        default="otari",
        description=(
            "The human-readable relying-party name an authenticator shows while a passkey "
            "is being created, and files it under afterwards. Cosmetic: nothing verifies it."
        ),
    )
    webauthn_allowed_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Origins a passkey ceremony may be performed from, each with a scheme and no "
            "trailing slash (e.g. 'https://otari.example.com'). Defaults to public_base_url "
            "alone. Set this only when more than one origin serves the dashboard under one "
            "relying-party ID; every entry must be the relying-party ID or a subdomain of it, "
            "which is checked at startup."
        ),
    )
    oauth_google_client_id: str | None = Field(
        default=None,
        description=(
            "The Google OAuth client ID dashboard sign-in uses. Set this and "
            "oauth_google_client_secret to offer 'Sign in with Google'; with either missing, the "
            "provider is absent from the sign-in screen rather than offered and then refused. "
            "public_base_url has to be set too, because the redirect URI is derived from it."
        ),
    )
    oauth_google_client_secret: str | None = Field(
        default=None,
        description="The Google OAuth client secret paired with oauth_google_client_id.",
    )
    oauth_github_client_id: str | None = Field(
        default=None,
        description=(
            "The GitHub OAuth client ID dashboard sign-in uses. Set this and "
            "oauth_github_client_secret to offer 'Sign in with GitHub'; with either missing, the "
            "provider is absent from the sign-in screen rather than offered and then refused. "
            "public_base_url has to be set too, because the redirect URI is derived from it."
        ),
    )
    oauth_github_client_secret: str | None = Field(
        default=None,
        description="The GitHub OAuth client secret paired with oauth_github_client_id.",
    )
    mail_transport: str = Field(
        default="auto",
        description=(
            "Which transport delivers outgoing mail: 'auto' (default) uses SMTP when "
            "smtp_host and mail_from_email are both set and sends nothing otherwise, "
            "'smtp' requires them and refuses to start without them, 'console' logs "
            "each message instead of delivering it (local development only: the log "
            "line contains the message body, including the invitation or reset token "
            "in its link), and 'none' turns mail off even where SMTP is configured."
        ),
    )
    smtp_host: str | None = Field(
        default=None,
        description=(
            "SMTP server host for outgoing mail. Unset disables mail entirely under the "
            "default 'auto' transport."
        ),
    )
    smtp_port: int = Field(default=587, ge=1, le=65535, description="SMTP server port.")
    smtp_user: str | None = Field(default=None, description="SMTP username, if the server requires auth.")
    smtp_password: str | None = Field(default=None, description="SMTP password, if the server requires auth.")
    smtp_tls: bool = Field(default=True, description="Use STARTTLS when connecting to the SMTP server.")
    mail_from_email: str | None = Field(
        default=None,
        description=(
            "The 'From' address on outgoing mail. Required, alongside smtp_host, before the "
            "default 'auto' transport sends anything over SMTP."
        ),
    )
    mail_from_name: str = Field(default="Otari", description="The 'From' display name on outgoing mail.")
    invitation_expiry_hours: int = Field(
        default=168,
        ge=1,
        description="How long an organization invitation stays acceptable, in hours (default 7 days).",
    )
    email_verification_expiry_hours: int = Field(
        default=48,
        ge=1,
        description="How long an email-verification link stays acceptable, in hours.",
    )
    password_reset_expiry_hours: int = Field(
        default=2,
        ge=1,
        description="How long a password-reset link stays acceptable, in hours.",
    )
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Pre-configured provider credentials, keyed by instance name. The key is "
            "normally the any-llm implementation (e.g. 'openai'); to run multiple "
            "instances of one implementation (e.g. real OpenAI plus a self-hosted "
            "OpenAI-compatible backend), give each a distinct instance name and set "
            "'provider_type' to the underlying implementation. An optional 'models' "
            "list declares model ids for instances whose backend has no /v1/models."
        ),
    )
    aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Model name aliases (display name -> target selector). A request naming an alias "
            "is routed to its target ('instance:model' or 'provider:model'), and the alias is "
            "what users see in GET /v1/models and in response 'model' fields, so the underlying "
            "provider/model can stay hidden. Pricing, budgets, and usage logs key on the resolved "
            "target. Standalone-mode only (hybrid resolves models against the platform)."
        ),
    )
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig,
        description=(
            "Named routing policies. A policy is a model name callers use like any other, which "
            "decides which real model serves the request ('select'), what is tried after a retryable "
            "failure ('on_failure'), and which guardrails always run. A one-target policy is an alias, "
            "so 'aliases:' remains its shorthand. Standalone-mode only: in hybrid mode the platform "
            "resolves the model, so a policy name would be sent upstream and rejected there."
        ),
    )
    # Tuning for the learned (kNN) router a policy can name via `select: [{router: knn}]`.
    # There is no on/off switch here on purpose: a policy naming the router is the
    # switch, so the router cannot be enabled globally behind an operator's back,
    # and two policies can never disagree about whether routing is on.
    router_alpha: float = Field(
        default=0.3,
        ge=0.0,
        description=(
            "Learned router's cost-vs-quality dial: score(model) = predicted_quality - alpha * "
            "normalized_cost. 0 ignores cost (always the best-predicted model); higher prefers "
            "cheaper candidates more aggressively."
        ),
    )
    router_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Neighbor count for the learned router's vote. A request whose partition holds fewer "
            "than k comparable examples stays on the policy's default target."
        ),
    )
    router_embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        description=(
            "provider:model used to embed the task signal. Changing it invalidates existing "
            "routing-memory records rather than mixing incomparable vector spaces, so the router "
            "returns to pass-through until the new space is re-taught."
        ),
    )
    router_confidence_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum share of the k neighbors that must agree on the winning candidate. Below it, "
            "the policy's default target leads and the router's order becomes the failover chain."
        ),
    )
    router_seed_count: int = Field(
        default=20,
        ge=0,
        description=(
            "Routing-memory records a pool needs before the router routes at all. Under it, every "
            "request through the policy serves the default target."
        ),
    )
    router_granularity: str = Field(
        default="trace_sticky",
        description=(
            "'trace_sticky' (default) decides once per conversation and reuses that decision on "
            "later turns; 'step' re-decides on every call."
        ),
    )
    router_max_records_per_user: int = Field(
        default=5000,
        ge=0,
        description=(
            "Cap on stored routing-memory records per user; the oldest are evicted past it. The "
            "store is scanned linearly, so this also bounds per-request routing latency. 0 disables "
            "eviction rather than storing nothing, and the per-request read falls back to the default "
            "bound, so the store grows without limit while each decision stays bounded."
        ),
    )
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description=(
            "Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})"
        ),
    )
    search_tools: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Search tools served by POST /v1/search, keyed by the name callers pass as "
            "'search_tool_name' (or in the /v1/search/{tool} path). Each entry may declare a "
            "'provider' (one of: exa, searxng; defaults to the tool name), an 'api_key' "
            "(required for exa), an 'api_base' (required for searxng unless web_search_url is "
            "set, which it then inherits), a 'timeout' in seconds, and an 'options' mapping of "
            "provider-native defaults. Standalone-mode only."
        ),
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint at /metrics",
    )
    enable_docs: bool = Field(
        default=True,
        description="Enable FastAPI docs endpoints (/docs, /redoc, /openapi.json). Enabled by default.",
    )
    bootstrap_api_key: bool = Field(
        default=True,
        description="Create a first-use API key on startup when no API keys exist",
    )
    bootstrap: str | None = Field(
        default=None,
        description=(
            "Composition-root bootstrap, as a 'module:callable' selector (OTARI_BOOTSTRAP). "
            "Imported once at startup after the core adapters are bound, and called with the "
            "container so an overlay can rebind ports and contribute routers. Unset means "
            "nothing is imported. Unrelated to bootstrap_api_key."
        ),
    )
    log_writer_strategy: str = Field(
        default="single",
        description="How usage log rows are written: 'single' (inline) or 'batch' (background).",
    )
    budget_strategy: str = Field(
        default="for_update",
        description="Budget validation strategy: 'for_update' (default), 'cas' (lock-free), or 'disabled'.",
    )
    require_pricing: bool = Field(
        default=True,
        description=(
            "Reject requests for models that have no configured pricing (fail-closed, default). "
            "When False, unpriced models are served and logged without cost (legacy behavior). "
            "Audio and moderation endpoints are always exempt — they have no token-based pricing."
        ),
    )
    default_pricing: bool = Field(
        default=False,
        description=(
            "When a model has no pricing in the database, fall back to community-maintained "
            "default pricing from the bundled genai-prices dataset. Off by default: a billing "
            "gateway should price from rates you control, and these community estimates can lag "
            "or differ from real provider rates. Database pricing always takes precedence. Enable "
            "to auto-price common models without configuring each one; while off, require_pricing "
            "stays fail-closed for any model you have not priced explicitly."
        ),
    )
    reject_user_mismatch: bool = Field(
        default=True,
        description=(
            "When True (default), a non-master key whose request names a 'user' other than its own "
            "is rejected with 403. When False, the client-supplied 'user' is still forwarded to the "
            "provider (OpenAI-style end-user tag) but spend is always bound to the key's own user; "
            "use this if clients send arbitrary 'user' values for abuse tracking. This setting "
            "is the deployment-wide default: an individual key can override it in either "
            "direction with its own reject_user_mismatch (null inherits this setting). The "
            "master key may always bill an arbitrary user regardless of this setting."
        ),
    )
    capture_agent_telemetry: bool = Field(
        default=True,
        description=(
            "When True (default), content-free coding-agent telemetry is stored as agent_telemetry "
            "rows: behavioral log events (tool_result, tool_decision, user_prompt, api_error) "
            "received at POST /v1/logs, and outcome-metric data points (lines of code, commits, "
            "pull requests, active time) received at POST /v1/metrics. When False, both are "
            "discarded before storage; usage capture and billing are unaffected either way. This "
            "is the deployment-wide default: an individual key can override it in either direction "
            "with its own capture_agent_telemetry (null inherits this setting)."
        ),
    )
    budget_reservation_ttl_sec: int = Field(
        default=900,
        gt=0,
        description=(
            "How long a budget reservation may stay in flight before the sweep treats it as "
            "leaked and returns the hold. It must comfortably exceed the slowest request this "
            "deployment serves, because reclaiming a hold that is still live would let a "
            "concurrent request past a cap the in-flight one is already spending against."
        ),
    )
    budget_reservation_sweep_interval_sec: int = Field(
        default=300,
        ge=0,
        description=(
            "How often to sweep for leaked budget reservations across all users. 0 disables the "
            "sweep, leaving the opportunistic per-user reclaim that runs when a user next "
            "reserves. Standalone mode only."
        ),
    )
    budget_reservation_sweep_batch: int = Field(
        default=500,
        gt=0,
        description="Maximum leaked budget reservations one sweep pass reclaims before yielding.",
    )
    budget_reservation_retention_sec: int = Field(
        default=604800,
        ge=0,
        description=(
            "How long a settled, released or reclaimed budget reservation is kept before the "
            "sweep deletes it. The row exists to make an in-flight hold reclaimable; what a "
            "request cost is recorded durably in usage_logs, so this is an audit window rather "
            "than an accounting record. 0 keeps every row forever. Standalone mode only."
        ),
    )
    budget_estimate_default_output_tokens: int = Field(
        default=1024,
        ge=0,
        description=(
            "Output-token count assumed when reserving budget for a request whose max output is "
            "unbounded. Used by the pre-debit estimate; reconciled to actual usage on completion."
        ),
    )
    stream_missing_usage_policy: str = Field(
        default="estimate",
        description=(
            "How to bill a streamed response that completes without provider usage data: "
            "'estimate' (charge the pre-debit estimate, default), 'fail' (charge estimate and mark "
            "the request errored), or 'allow_free' (release the reservation, legacy behavior)."
        ),
    )
    streaming_keepalive_interval_ms: int = Field(
        default=15000,
        ge=0,
        description=(
            "Idle interval in milliseconds after which a streaming response emits a transport "
            "keepalive while it waits on the provider: a 'ping' event on /v1/messages, an SSE "
            "comment line on /v1/chat/completions and /v1/responses. Keeps an intermediary with a "
            "read timeout (Cloudflare's default Proxy Read Timeout is 125s) from severing a connection "
            "during a long time-to-first-token. Does not extend any first-chunk or failover deadline. "
            "0 disables."
        ),
    )
    model_discovery: bool = Field(
        default=True,
        description="Enable auto-discovery of models from configured providers via GET /v1/models",
    )
    model_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL in seconds for the in-memory model discovery cache (0 disables caching)",
    )
    model_discovery_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Per-provider timeout in seconds for a live model-discovery (list_models) "
            "call. Bounds how long an unreachable or slow provider can stall discovery "
            "before it is treated as failed and the declared models: fallback is used."
        ),
    )
    model_discovery_negative_ttl_seconds: float = Field(
        default=30.0,
        ge=0,
        description=(
            "How long a failed model-discovery result is remembered before that provider "
            "is dialed again, in seconds. Stops an unreachable provider from being re-tried "
            "on every request (0 disables negative caching, restoring retry-every-time). "
            "Applies to a read that dials: a provider never dialed before, or any read "
            "while model_cache_ttl_seconds is 0. Otherwise the background refresher owns "
            "the dialing and model_cache_ttl_seconds bounds how soon a recovered provider "
            "is seen again."
        ),
    )
    models_dev_metadata: bool = Field(
        default=True,
        description=(
            "Enrich the dashboard's model detail with metadata (modalities, "
            "capabilities, knowledge cutoff) fetched from the public models.dev "
            "catalog. Set false to disable the outbound call; the gateway then "
            "falls back to the bundled genai-prices data."
        ),
    )
    models_dev_cache_ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description=(
            "TTL in seconds for the cached models.dev catalog, and the interval at which a "
            "background task refetches it (floored at 5 minutes). Above 0, GET /v1/models/metadata "
            "answers from the cache instead of waiting on the fetch; a failed fetch is held for one "
            "minute rather than the refresh interval. 0 disables caching, so every read fetches."
        ),
    )
    files_enabled: bool = Field(
        default=True,
        description="Enable the /v1/files upload/storage endpoints (standalone mode).",
    )
    files_backend: str = Field(
        default="local",
        description="Blob backend for uploaded file bytes: 'local' (filesystem) or 's3'. Future: 'gcs'.",
    )
    files_local_dir: str = Field(
        default="./otari-files",
        description="Directory for the 'local' files backend to store uploaded bytes.",
    )
    files_s3_bucket: str | None = Field(
        default=None,
        description="Bucket name for the 's3' files backend. Required when files_backend is 's3'.",
    )
    files_s3_endpoint_url: str | None = Field(
        default=None,
        description=(
            "S3-compatible endpoint URL for the 's3' files backend, e.g. a self-hosted MinIO "
            "instance. None uses AWS S3's default endpoint resolution (region-based)."
        ),
    )
    files_s3_region: str | None = Field(
        default=None,
        description=(
            "AWS region for the 's3' files backend. Most self-hosted S3-compatible stores "
            "(e.g. MinIO) ignore this; it still must resolve to some value, so it defaults to "
            "'us-east-1' when unset."
        ),
    )
    files_max_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1,
        description="Maximum size in bytes for a single uploaded file.",
    )
    files_retention_hours: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Stop serving files older than this many hours: expired files become inaccessible "
            "(404) and can no longer be referenced. Their stored bytes are not yet reclaimed "
            "automatically, so periodic cleanup is an operator task. None keeps files indefinitely."
        ),
    )
    file_understanding_enabled: bool = Field(
        default=True,
        description=(
            "Normalize file/image content blocks before the provider call: pass through for "
            "natively-capable models, extract to text for text-only models. When False, content "
            "blocks are forwarded unchanged (legacy pass-through)."
        ),
    )
    vision_strategy: str = Field(
        default="describe",
        description=(
            "How image blocks are handled for text-only models: 'describe' (side-call a vision "
            "model, falling back to a logged drop if none is configured), 'ocr' (extract text only), "
            "or 'off' (drop with a log line)."
        ),
    )
    vision_describe_model: str | None = Field(
        default=None,
        description=(
            "provider/model used to caption images for text-only target models when "
            "vision_strategy='describe'. May point at a local vision model (e.g. ollama/qwen2-vl) "
            "to keep captioning free. When unset, 'describe' falls back to a logged drop."
        ),
    )
    vision_describe_max_tokens: int = Field(
        default=1024,
        gt=0,
        description=(
            "Cap on the describe model's output tokens per image. Bounds the cost and latency "
            "of the vision side-call, which is billed to the user and runs once per image (and "
            "once per page for scanned PDFs)."
        ),
    )
    model_capabilities: dict[str, ModelCapabilityConfig] = Field(
        default_factory=dict,
        description=(
            "Per-model multimodal capability overrides (provider/model -> {supports_image, "
            "supports_pdf}). Authoritative over any-llm's provider-level flags; needed for text-only "
            "local models behind OpenAI-compatible servers."
        ),
    )
    sandbox_url: str | None = Field(
        default=None,
        description=(
            "Base URL of the code-execution sandbox backend for otari_code_execution tools. "
            "When unset, otari_code_execution requests are rejected with 400."
        ),
    )
    guardrails_url: str | None = Field(
        default=None,
        description=(
            "Default URL of the input-guardrails service used when a request does not pass its "
            "own guardrail `url`. docker-compose sets this to the bundled guardrails container."
        ),
    )
    tools_header: str | None = Field(
        default=None,
        description=(
            "Per-deployment override for the purpose-hint preamble header injected ahead of "
            "gateway-managed tool hints. When unset, a built-in default header is used."
        ),
    )
    sandbox_purpose_hint: str | None = Field(
        default=None,
        description=(
            "Default purpose hint forwarded to the sandbox backend when an otari_code_execution "
            "tool entry does not supply its own."
        ),
    )
    # "session_image" rather than the more obvious "image": ``OTARI_SANDBOX_IMAGE``
    # is already taken. ``docker-compose.yml`` documents it as the Docker tag of the
    # sandbox *container* to boot, and both names are read from the operator's own
    # environment, so a field spelled ``sandbox_image`` would silently make one
    # variable mean two things. The near-miss is what makes it dangerous: an
    # operator overriding the container tag would also start pinning that tag onto
    # every leased session, and (via ``pinnable_sandbox_images``) offering it to
    # workspaces. This names the narrower thing it actually is: the image a leased
    # session runs, not the image the backend process is.
    sandbox_session_image: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Sandbox image this deployment asks the code-execution backend to run "
            "(e.g. 'mzdotai/otari-sandbox-container:latest'). When unset, nothing is asked for and "
            "the backend runs whatever it runs by default. A workspace policy may name a different "
            "image only if sandbox_allowed_session_images lists it."
        ),
    )
    sandbox_allowed_session_images: str | None = Field(
        default=None,
        description=(
            "Comma-separated sandbox images a workspace's code-execution policy may pin "
            "(e.g. 'mzdotai/otari-sandbox-container:latest,ghcr.io/acme/sandbox:2'). Deliberately "
            "not editable from the dashboard: it is the operator's supply-chain allow-list, and "
            "sandbox_session_image is always pinnable whether or not it appears here. When unset, a "
            "workspace may not pin an image at all."
        ),
    )
    web_search_url: str | None = Field(
        default=None,
        description=(
            "Base URL of the web-search backend (SearXNG instance or a search adapter) for "
            "otari_web_search tools. When unset, otari_web_search requests are rejected with 400. "
            "docker-compose sets this to the bundled SearXNG container."
        ),
    )
    web_search_purpose_hint: str | None = Field(
        default=None,
        description=(
            "Default purpose hint for the web-search backend when an otari_web_search tool entry "
            "does not supply its own."
        ),
    )
    web_search_engines: str | None = Field(
        default=None,
        description=(
            "Comma-separated SearXNG engine list for the web-search backend (e.g. 'google,bing'). "
            "When unset, the backend default engines are used."
        ),
    )
    web_search_max_results: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Default cap on the number of hits returned by the web-search backend (a per-tool "
            "max_results still overrides it)."
        ),
    )
    web_search_extract: bool | None = Field(
        default=None,
        description=(
            "Whether the web-search backend extracts page content in-process (True) or returns "
            "snippet-only results (False). When unset, the backend default (extraction on) applies."
        ),
    )
    web_search_intercept: bool | None = Field(
        default=None,
        description=(
            "Whether a provider-named web-search declaration (bare 'web_search', Anthropic-native "
            "'web_search_<date>') is run against the gateway's own backend instead of being forwarded "
            "to the provider. Off when unset: the explicit otari_web_search type is always run by the "
            "gateway, and every other keyword reaches the provider untouched. Requires web_search_url."
        ),
    )
    web_search_allow_private_hosts: bool = Field(
        default=False,
        description=(
            "SSRF gate: allow the web-search backend to fetch private/loopback/reserved hosts. "
            "Off by default. Only enable for unusual setups such as a private search index."
        ),
    )
    mcp_allow_loopback: bool = Field(
        default=True,
        description=(
            "SSRF gate: allow MCP server URLs that resolve to loopback (useful for same-host "
            "sidecars). On by default."
        ),
    )
    mcp_allow_private_hosts: bool = Field(
        default=False,
        description=(
            "SSRF gate: allow MCP server URLs that resolve to private/reserved hosts, and accept "
            "hostnames that fail to resolve at validation time. Off by default."
        ),
    )
    provider_allow_private_hosts: bool = Field(
        default=True,
        description=(
            "SSRF gate: allow a provider api_base that resolves to private/loopback/reserved hosts. "
            "On by default (the opposite of the other SSRF gates) because operator-supplied api_base "
            "values are master-key gated and the home-lab / self-hosted use case depends on private "
            "endpoints. Set to false to make provider connection tests, model discovery, and the "
            "credential write path (POST /v1/provider-credentials and PATCH /v1/provider-credentials/{instance}) "
            "refuse an internal api_base. "
            "Chat dispatch (which dials the endpoint on every request) is not gated, so this is not a "
            "general egress control. Also settable via OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS."
        ),
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Otari operating mode: 'standalone', 'hosted' or 'hybrid'. When unset (the default), the "
            "mode is derived from the platform token: hybrid if a token is present (OTARI_AI_TOKEN), "
            "else standalone. Set explicitly to assert the intended mode: 'hybrid' requires a token, "
            "and 'standalone' or 'hosted' with a token present is rejected at startup as conflicting "
            "configuration. 'hosted' is standalone's multi-tenant sibling: it owns its own database "
            "and serves the whole management API, and it reports the per-organization provider-key "
            "surface rather than the process-global one. "
            "Legacy value 'platform' is accepted as an alias for 'hybrid'."
        ),
    )
    platform: dict[str, Any] = Field(default_factory=dict, description="otari.ai connection settings")

    # Resolved once from the environment (primed by load_config, or lazily on
    # first access for a directly-constructed config) so the runtime mode stays
    # stable for the process: the token cannot flip mid-request from an env
    # mutation, and hot paths no longer re-read os.getenv on every access.
    _platform_token: str | None = PrivateAttr(default=None)
    _platform_token_resolved: bool = PrivateAttr(default=False)

    # The config-file providers as loaded, before any dashboard-stored providers
    # are overlaid onto ``providers`` at runtime. Captured once by
    # ``provider_store_service`` so every refresh rebuilds the merged view from a
    # stable base (and a deleted stored row restores the config entry). Kept on
    # the config, not a module global, so it is per-config and cannot leak
    # between processes or tests.
    _provider_baseline: dict[str, dict[str, Any]] | None = PrivateAttr(default=None)

    # The same idea for ``search_tools``: the config-file tools as loaded, before
    # any dashboard-stored tool is overlaid by ``search_tool_store_service``.
    _search_tool_baseline: dict[str, dict[str, Any]] | None = PrivateAttr(default=None)

    # SHA-256 hash of a master key generated on first run (see
    # ``master_key_service``). Set at startup when no ``master_key`` is
    # configured, so ``verify_master_key`` can authenticate the generated key
    # without the plaintext ever living in config.
    _master_key_hash: str | None = PrivateAttr(default=None)

    def _resolve_platform_token(self) -> str | None:
        if not self._platform_token_resolved:
            self._platform_token = _get_platform_token_from_env()
            self._platform_token_resolved = True
        return self._platform_token

    @property
    def platform_token(self) -> str | None:
        return self._resolve_platform_token()

    @property
    def platform_management_url(self) -> str:
        """The otari.ai dashboard URL a hybrid gateway points its operator at.

        Not a credential and not the resolve/report base URL: it is the human
        destination the deployment bootstrap publishes so a hybrid landing page
        can link to the control plane that owns this gateway.
        """
        configured = self.platform.get("management_url")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return DEFAULT_PLATFORM_MANAGEMENT_URL

    @property
    def configured_mode(self) -> str | None:
        """The explicitly set mode (normalized), or None when unset/blank."""
        normalized = (self.mode or "").strip().lower()
        return normalized or None

    @property
    def effective_mode(self) -> str:
        configured = self.configured_mode
        if configured in {"hybrid", "platform"}:
            return "hybrid"
        if configured == "hosted":
            return "hosted"
        if configured == "standalone":
            return "standalone"
        # Mode unset: derive from the platform token.
        return "hybrid" if self.platform_token else "standalone"

    @property
    def is_hybrid_mode(self) -> bool:
        return self.effective_mode == "hybrid"

    @property
    def is_hosted_mode(self) -> bool:
        """Whether this deployment is the multi-tenant control plane, not a single tenant's.

        A data-plane sibling of ``is_hybrid_mode`` it is not: hosted mode owns
        its own database and mounts the whole management API, exactly as
        standalone does, and every request path that asks ``is_hybrid_mode``
        gets the same answer here as it would for a standalone gateway. What it
        changes is who the deployment serves, and therefore which management
        surfaces make sense on it: the per-organization credential set rather
        than the process-global one (see ``bootstrap.HOSTED_SURFACES``).
        """
        return self.effective_mode == "hosted"

    @property
    def effective_mail_transport(self) -> str:
        """Which transport a send would actually use: ``smtp``, ``console`` or ``none``.

        Answers what a send *would use*, never what was asked for, and it stays
        the same answer whether or not :meth:`validate_mail_transport` was ever
        called. That is deliberate: a readiness answer that is only truthful
        because a check ran somewhere else is the shape of bug this whole design
        exists to rule out. An explicitly-configured ``smtp`` missing the
        settings SMTP needs is still refused at startup, but it also reports
        ``none`` here, so a caller that somehow reached this config cannot be
        told mail works when :func:`~gateway.services.mail.select_transport`
        would hand it nothing.

        ``auto`` is the state a deployment that never heard of mail is in: SMTP
        when both ``smtp_host`` and ``mail_from_email`` are set, and no
        transport at all otherwise.
        """
        configured = self.mail_transport.strip().lower()
        if configured in {"console", "none"}:
            return configured
        # 'auto' and an explicit 'smtp' resolve identically, because what SMTP
        # needs to exist does not depend on how it was asked for. They differ
        # only in whether the absence is an error, which is validation's job.
        return "smtp" if self.smtp_host and self.mail_from_email else "none"

    @property
    def mail_enabled(self) -> bool:
        """Whether a transport is configured, so a send is worth attempting.

        Mail-dependent surfaces read this (or :attr:`mail_ready`, below) to
        report themselves unavailable rather than failing at send time, per the
        no-mail-configured requirement.
        """
        return self.effective_mail_transport != "none"

    @property
    def mail_ready(self) -> bool:
        """Whether this deployment can send a message that links back to itself.

        ``mail_enabled`` alone is not enough: every message the control plane
        sends carries a link into this deployment (an invitation's accept link,
        and the verification and password-reset links to come), and a deployment
        is the only one that knows its own address, so ``public_base_url`` has to
        be set too. A surface with a fallback degrades to it (an invitation is
        still created and its accept link returned, just not emailed); a surface
        with none is absent while this is false.
        """
        return self.mail_enabled and bool(self.public_base_url)

    @property
    def missing_mail_settings(self) -> tuple[str, ...]:
        """Which settings stand between this deployment and a delivered link, in config order.

        Empty exactly when :attr:`mail_ready` is true. Reported to the operator
        (``GET /v1/settings/mail``) so "mail is unavailable" names what to set
        rather than leaving them to guess, which is the whole difference between
        an honest no-transport mode and an opaque one.
        """
        configured = self.mail_transport.strip().lower()
        # An explicit 'none' turned mail off deliberately, so nothing else is
        # "missing": mail_transport itself is the one setting standing in the
        # way, and naming the SMTP fields beside it would describe a deployment
        # the operator did not ask for.
        if configured == "none":
            return ("mail_transport",)

        missing: list[str] = []
        if configured in {"auto", "smtp"}:
            if not self.smtp_host:
                missing.append("smtp_host")
            if not self.mail_from_email:
                missing.append("mail_from_email")
        if not self.public_base_url:
            missing.append("public_base_url")
        return tuple(missing)

    def oauth_client_credentials(self, provider: str) -> tuple[str, str] | None:
        """The client ID and secret configured for ``provider``, or None.

        None is a deployment that did not configure this provider, which is a
        setting and not a failure: the sign-in screen simply does not offer it
        (``GET /v1/bootstrap``'s ``oauth_providers``).

        ``public_base_url`` is part of being configured rather than a separate
        check, because the redirect URI is derived from it
        (``services.oauth_service.redirect_uri``) and a provider whose
        authorization URL cannot be built is not on offer. Half a pair (an ID
        with no secret) reads as unconfigured for the same reason: the exchange
        would fail at the provider, and offering the button would be a promise
        this deployment cannot keep.
        """
        if not self.public_base_url:
            return None
        client_id = getattr(self, f"oauth_{provider}_client_id", None)
        client_secret = getattr(self, f"oauth_{provider}_client_secret", None)
        if not client_id or not client_secret:
            return None
        return client_id, client_secret

    @property
    def oauth_providers(self) -> tuple[str, ...]:
        """Which OAuth providers this deployment can sign somebody in with, sorted.

        Empty on a deployment that configured none, which is the default, and
        which is why the sign-in screen carries no OAuth affordance out of the
        box rather than a pair of dead buttons.
        """
        return tuple(sorted(name for name in OAUTH_PROVIDERS if self.oauth_client_credentials(name) is not None))

    @property
    def webauthn_relying_party(self) -> RelyingParty | None:
        """The relying party passkeys on this deployment are bound to, or None.

        None means this deployment cannot run a passkey ceremony, which is a
        configuration state and not a failure: WebAuthn requires a relying-party
        ID, and a gateway that has not been told its own address cannot invent
        one. Deriving it from the request's ``Host`` header instead is the shape
        this deliberately does not take: ``Host`` is attacker-controlled on any
        deployment that does not pin it, and the ID is the *only* thing scoping a
        passkey to this site, so a derived-per-request ID would let a request
        choose which site's passkeys it is talking about. It is also unstable by
        nature, and an ID that moves silently orphans every passkey registered
        under the previous one. So it is configured (``webauthn_rp_id``) or
        derived once from a configured address (``public_base_url``), and
        `docs/access-control.md` documents which.

        The derivation drops the port along with the scheme: a relying-party ID
        is a bare domain, so 'https://localhost:8000' yields 'localhost', which
        is what makes passkeys work in local development over plain HTTP (the
        one origin browsers treat as secure without TLS).

        Origins default to ``public_base_url`` alone. That is the same address
        the ID came from, so the common deployment configures one setting and
        gets a consistent pair rather than two settings it can put out of step.
        """
        # Lowercased to match ``_host_of``, which returns what ``urlsplit``
        # already normalized. Without it a configured "Otari.Example.com" passes
        # validation (which compares against its own lowercased form) and then
        # fails ``covers`` against the lowercased origin host, so startup blames
        # webauthn_allowed_origins for a casing mistake in webauthn_rp_id. A
        # relying-party ID is a domain, and domains are case-insensitive.
        rp_id = (self.webauthn_rp_id or "").strip().lower() or _host_of(self.public_base_url)
        if not rp_id:
            return None
        origins = tuple(origin.strip().rstrip("/") for origin in self.webauthn_allowed_origins if origin.strip())
        if not origins:
            base = (self.public_base_url or "").strip().rstrip("/")
            if not base:
                # An explicit rp_id with no address to serve it from. Refusing
                # here rather than accepting the ID alone: `expected_origin` has
                # no safe default, and defaulting it to "https://{rp_id}" would
                # quietly guess the scheme and port of a deployment that never
                # said.
                return None
            origins = (base,)
        return RelyingParty(rp_id=rp_id, name=self.webauthn_rp_name, origins=origins)

    @property
    def webauthn_enabled(self) -> bool:
        """Whether this deployment can register and verify passkeys."""
        return self.webauthn_relying_party is not None

    def provider_instance_type(self, instance: str) -> str:
        """Return the any-llm implementation backing a provider instance.

        When the instance declares a ``provider_type`` (optionally an alias like
        ``openai-compatible``) that is returned, normalized to the real
        implementation name; otherwise the instance name itself is the
        implementation (the fully backward-compatible default). Unknown instance
        names are returned unchanged so the caller's own resolution surfaces the
        error.
        """
        entry = self.providers.get(instance)
        if isinstance(entry, dict):
            declared = entry.get("provider_type")
            if isinstance(declared, str) and declared:
                return PROVIDER_TYPE_ALIASES.get(declared, declared)
        return instance

    def provider_pricing_implementation(self, instance: str) -> str | None:
        """The vendor backing an instance for pricing purposes, or ``None``.

        Like :meth:`provider_instance_type`, except a ``*-compatible`` declaration
        yields ``None`` instead of the implementation it normalizes to, and an
        instance that declares no ``provider_type`` yields ``None`` rather than its
        own name. Those aliases name a *wire protocol*, not who serves the request:
        ``openai-compatible`` is how a self-hosted vLLM, Ollama or LiteLLM endpoint
        is declared, and such servers commonly expose OpenAI's own model names
        verbatim (``text-embedding-3-small``), so pricing them as OpenAI would
        charge OpenAI's list rate for a model the operator hosts themselves.
        Configure an explicit price for those instead.
        """
        entry = self.providers.get(instance)
        if not isinstance(entry, dict):
            return None
        declared = entry.get("provider_type")
        if not isinstance(declared, str) or not declared or declared in PROVIDER_TYPE_ALIASES:
            return None
        return declared

    def resolve_alias(self, name: str) -> str | None:
        """Return the target selector for a configured alias, or None.

        The alias is a display name (e.g. ``myopusmodel``) that maps to a real
        selector (``instance:model`` / ``provider:model``). Returns ``None`` when
        ``name`` is not a configured alias, so callers fall through to ordinary
        selector resolution.
        """
        target = self.aliases.get(name)
        return target if isinstance(target, str) and target else None

    def validate_alias(self, name: str, target: str, *, alias_names: Container[str] | None = None) -> None:
        """Validate a single alias, raising ``ValueError`` with the reason.

        An alias must name a non-empty target selector with a usable
        ``instance``/``provider`` prefix; the prefix must resolve to a configured
        provider instance or a known any-llm implementation. An alias name must
        not contain a selector delimiter (``:`` or ``/``): alias lookup runs
        before selector resolution, so such a name would silently reroute
        requests for a real ``provider:model``. It must also not collide with a
        configured provider instance (that would be ambiguous), and its target
        must not itself be an alias (no chaining for now).

        ``alias_names`` is the set of names that count as aliases for the
        chaining check, defaulting to the configured ones. Callers that also
        store aliases elsewhere (the runtime ``model_aliases`` table) pass the
        union, so chaining is rejected no matter which side each alias came from.
        """
        known_aliases: Container[str] = self.aliases if alias_names is None else alias_names

        if not name:
            msg = "alias name must not be empty."
            raise ValueError(msg)
        if ":" in name or "/" in name:
            msg = f"alias name '{name}' must not contain ':' or '/' (it would shadow a real model selector)."
            raise ValueError(msg)
        if name in self.providers:
            msg = f"alias '{name}' collides with a configured provider instance name."
            raise ValueError(msg)
        if not isinstance(target, str) or not target:
            msg = f"aliases.{name} must be a non-empty target selector string."
            raise ValueError(msg)
        colon, slash = target.find(":"), target.find("/")
        cut = colon if colon != -1 and (slash == -1 or colon < slash) else slash
        if cut <= 0 or cut == len(target) - 1:
            msg = f"aliases.{name} target '{target}' must be of the form 'instance:model' or 'provider:model'."
            raise ValueError(msg)
        prefix = target[:cut]
        if prefix in known_aliases:
            msg = f"aliases.{name} target '{target}' points at another alias; alias chaining is not supported."
            raise ValueError(msg)
        if prefix in self.providers:
            return
        # No PROVIDER_TYPE_ALIASES mapping here: that normalizes an instance's
        # declared provider_type, not a selector prefix. Request-time routing
        # splits the selector through any-llm, which knows no such mapping, so
        # accepting "openai-compatible:model" would pass startup and then fail
        # on the first request.
        try:
            LLMProvider(prefix)
        except ValueError as exc:
            msg = (
                f"aliases.{name} target '{target}' prefix '{prefix}' is neither a configured "
                "provider instance nor a known provider implementation."
            )
            raise ValueError(msg) from exc

    def validate_aliases(self) -> None:
        """Validate the ``aliases`` map at startup so misconfig fails fast."""
        for name, target in self.aliases.items():
            self.validate_alias(name, target)

    def policy_names(self) -> set[str]:
        """Every configured routing-policy name. Empty when routing is disabled."""
        if not self.routing.enabled:
            return set()
        return set(self.routing.policies)

    def validate_routing_policies(self) -> None:
        """Validate the ``routing:`` block at startup so misconfig fails fast.

        A policy name reaches requests the same way an alias name does, so it
        answers to the same rules: no selector delimiter, no collision with a
        provider instance, and no chaining (a target may not name another policy
        or an alias). It additionally may not collide with an ``aliases:`` entry:
        both would claim the same caller-facing name, and silently preferring one
        would make the other dead config.

        Validation runs even when ``routing.enabled`` is false. A disabled block
        is still config someone will re-enable, and finding out it was malformed
        at that point defeats the purpose of an off-switch.
        """
        alias_names = set(self.aliases)
        policy_names = set(self.routing.policies)

        for name, spec in self.routing.policies.items():
            where = f"routing.policies.{name}"
            if not name:
                msg = "routing policy name must not be empty."
                raise ValueError(msg)
            if ":" in name or "/" in name:
                msg = (
                    f"routing policy name '{name}' must not contain ':' or '/' "
                    "(it would shadow a real model selector)."
                )
                raise ValueError(msg)
            if name in self.providers:
                msg = f"routing policy '{name}' collides with a configured provider instance name."
                raise ValueError(msg)
            if name in alias_names:
                msg = (
                    f"routing policy '{name}' collides with the alias of the same name "
                    f"(aliases.{name} -> '{self.aliases[name]}'). Both claim the same model name for "
                    "callers, so one would be dead config. Rename the policy, or delete the alias and "
                    "express it as the policy's default target."
                )
                raise ValueError(msg)

            # Reuse the alias target rules for every selector the policy names:
            # the prefix must resolve to a configured instance or a known
            # implementation, and it must not point at another indirection.
            # `validate_alias` phrases its errors as `aliases.<name>`, so the
            # message is re-raised under the policy's own path.
            chainable = alias_names | policy_names
            for selector in spec.static_selectors():
                try:
                    self.validate_alias(name, selector, alias_names=chainable)
                except ValueError as exc:
                    detail = str(exc).replace(f"aliases.{name}", f"{where}", 1)
                    raise ValueError(detail) from exc

            if spec.default_target in spec.on_failure:
                msg = (
                    f"{where}: '{spec.default_target}' is both the default target and an on_failure "
                    "entry. Retrying the candidate that just failed cannot help; remove it from on_failure."
                )
                raise ValueError(msg)

            self._warn_on_policy_name_shadowing_a_model(name, where)

    def _warn_on_policy_name_shadowing_a_model(self, name: str, where: str) -> None:
        """Warn when a policy name matches a model id declared by an instance.

        A warning rather than a refusal, deliberately. ``aliases:`` has always
        allowed a name that collides with a real model id, so refusing outright
        would stop gateways booting on a config file that was valid before the
        upgrade. This warns for now; the refusal lands a release later, and the
        message says so. Only declared ``models`` lists are checked: the full set
        a provider serves is discovered asynchronously and is not knowable here.
        """
        for instance, entry in self.providers.items():
            declared = entry.get("models") if isinstance(entry, dict) else None
            if isinstance(declared, list) and name in declared:
                logger.warning(
                    "%s shadows model '%s' declared by provider instance '%s'. Requests naming '%s' will "
                    "route through the policy, not the model, including for pricing and budget "
                    "attribution. This will be refused in a future release; rename the policy now.",
                    where,
                    name,
                    instance,
                    name,
                )
                return

    def validate_provider_instances(self) -> None:
        """Validate per-instance ``provider_type`` / ``models`` declarations.

        Fails fast at startup so a typo in ``provider_type`` (or a non-list
        ``models``) surfaces immediately rather than as a per-request error.
        Instances without a ``provider_type`` are left unvalidated to preserve
        the existing lenient behavior (the key is the implementation). A
        settings-less entry for a provider that does need a credential only
        warns, see :meth:`_warn_on_uncredentialed_bare_entry`.
        """
        for instance, entry in self.providers.items():
            # The selector splits on the first ``:`` / ``/``, so an instance name
            # containing either could never be matched and would be silently
            # unreachable. Reject it here rather than fail confusingly at request
            # time. (No real any-llm provider name contains these characters.)
            if ":" in instance or "/" in instance:
                msg = f"provider instance name '{instance}' must not contain ':' or '/'."
                raise ValueError(msg)
            if not isinstance(entry, dict):
                continue
            declared = entry.get("provider_type")
            if isinstance(declared, str) and declared:
                impl = PROVIDER_TYPE_ALIASES.get(declared, declared)
                try:
                    LLMProvider(impl)
                except ValueError as exc:
                    msg = (
                        f"providers.{instance}.provider_type '{declared}' is not a known provider "
                        "implementation."
                    )
                    raise ValueError(msg) from exc
            models = entry.get("models")
            if models is not None and not (isinstance(models, list) and all(isinstance(m, str) for m in models)):
                msg = f"providers.{instance}.models must be a list of model id strings."
                raise ValueError(msg)
            if not entry:
                self._warn_on_uncredentialed_bare_entry(instance)

    def _warn_on_uncredentialed_bare_entry(self, instance: str) -> None:
        """Warn when a settings-less entry names a provider that needs a credential.

        A bare ``ollama:`` is the documented way to opt a keyless local backend
        into discovery, but the same shape under a keyed provider (``openai:``
        with nothing beneath it) is far more likely a truncated YAML edit than
        intent, and it used to fail the load outright as a type error. Warn rather
        than raise: the entry is legitimate whenever the credential reaches
        any-llm another way (its own environment variable, an instance role), so a
        gateway that works today must keep booting.

        A settings-less entry has no ``provider_type`` to declare either, so the
        instance name is the implementation the credential question is asked of.
        """
        env_names = provider_credential_env_names(instance)
        # Empty: a keyless backend, nothing to warn about. None: a provider we
        # cannot inspect, so we do not know that a credential is needed.
        if not env_names:
            return
        if any(os.getenv(name) for name in env_names):
            return
        logger.warning(
            "providers.%s has no settings and no credential: that provider needs an API key and none of %s is set. "
            "A keyless local backend (ollama, llamacpp, llamafile) is configured this way on purpose; otherwise "
            "this looks like a truncated config entry, and requests to it will fail.",
            instance,
            ", ".join(env_names),
        )

    @field_validator("providers", mode="before")
    @classmethod
    def _coerce_valueless_provider_entries(cls, providers: Any) -> Any:
        """Treat a provider entry with no body as an empty config block.

        A keyless local backend (ollama, llamacpp, llamafile) has no credential to
        declare, so the natural way to configure one is a bare key::

            providers:
              ollama:

        YAML parses that as ``None``, which the ``dict[str, dict[str, Any]]``
        annotation would otherwise reject with a type error. Normalize it to
        ``{}``, which means "this instance is configured, with no settings": the
        instance is then routable and, since discovery is scoped to the configured
        instances, also discoverable in ``GET /v1/models`` (issue #389).

        A ``providers:`` block with no entries at all gets the same treatment, so
        commenting out every entry reads as "no providers" rather than the same
        confusing type error.

        The coercion is deliberately not limited to the keyless backends, since an
        entry may name any instance; a bare entry under a provider that does need
        a credential is caught at startup by
        :meth:`GatewayConfig._warn_on_uncredentialed_bare_entry`, which keeps the
        truncated-YAML case visible without failing the load.
        """
        if providers is None:
            return {}
        if not isinstance(providers, dict):
            return providers
        return {instance: ({} if entry is None else entry) for instance, entry in providers.items()}

    def search_tool_providers(self) -> set[str]:
        """The distinct providers backing the configured search tools.

        These are prefixes of the ``<provider>:<tool>`` keys that search pricing,
        usage, and per-key access lists are written against, so the allow-list
        writer has to accept them alongside real provider instances.
        """
        return {str(entry.get("provider") or name) for name, entry in self.search_tools.items()}

    def search_tools_without_backend_url(self) -> list[str]:
        """Search tools whose provider needs an ``api_base`` and has none to inherit.

        Reported as a startup warning rather than raised by
        :meth:`validate_search_tools`, because ``web_search_url`` (which a
        ``searxng`` tool inherits) can also come from a dashboard-stored
        override, and those are applied to the config after it loads. Failing at
        load time would refuse to boot a gateway the operator has in fact
        configured. Enforcement is per request instead: ``resolve_search_tool``
        refuses such a tool with a 400, and the rest of the gateway serves.
        """
        if self.web_search_url:
            return []
        return [
            name
            for name, entry in self.search_tools.items()
            if isinstance(entry, dict)
            and str(entry.get("provider") or name) in SEARCH_PROVIDERS_REQUIRING_API_BASE
            and not entry.get("api_base")
        ]

    def effective_sandbox_image(self) -> str | None:
        """The image this deployment asks a sandbox session for, or ``None``.

        Resolved the way every other tool field is: the config value (which a
        dashboard override has already been written onto) and then the env var,
        because clearing an override sets the attribute to ``None`` and the
        deployment should fall back to what it was configured with rather than
        to nothing (``services/tool_settings_service``).
        """
        return (self.sandbox_session_image or "").strip() or otari_env("SANDBOX_SESSION_IMAGE") or None

    def pinnable_sandbox_images(self) -> tuple[str, ...]:
        """The sandbox images a workspace's code-execution policy may name.

        The operator's curated list, plus this deployment's own
        ``sandbox_session_image``: a workspace naming the image every request already
        gets is asking for nothing it did not already have, so refusing it would
        only be confusing. Order is the operator's, with the deployment image
        first, and duplicates collapse.

        Empty is the meaningful default. An operator who has curated nothing has
        not vetted anything for a workspace to pin, and a workspace-settable
        image is a supply-chain surface rather than a string, so the answer to
        "which images may they choose from" is *none* until one is named.
        """
        curated = self.sandbox_allowed_session_images or otari_env("SANDBOX_ALLOWED_SESSION_IMAGES") or ""
        images: list[str] = []
        for candidate in (self.effective_sandbox_image(), *curated.split(",")):
            image = (candidate or "").strip()
            if image and image not in images:
                images.append(image)
        return tuple(images)

    def validate_search_tools(self) -> None:
        """Validate the ``search_tools`` map at startup so misconfig fails fast.

        Per-entry rules live in :func:`validate_search_tool_entry`, which the
        runtime CRUD path applies to a dashboard-written tool as well.
        """
        for name, entry in self.search_tools.items():
            validate_search_tool_entry(name, entry)

    @field_validator("stream_missing_usage_policy")
    @classmethod
    def _validate_stream_missing_usage_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in STREAM_MISSING_USAGE_POLICIES:
            msg = f"stream_missing_usage_policy must be one of {sorted(STREAM_MISSING_USAGE_POLICIES)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("docs_url")
    @classmethod
    def _validate_docs_url(cls, value: str | None) -> str | None:
        """Reject a documentation link that is not an absolute http(s) URL.

        The deployment bootstrap publishes this to the browser as a link target,
        so a scheme that is not http(s) would be a script URL an operator put in
        their own config. Rejected at load rather than dropped per request, for
        the same reason ``platform.management_url`` is: a typo should be a
        startup error, not a Documentation link that silently goes nowhere.
        """
        normalized = (value or "").strip()
        if not normalized:
            return None
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = f"docs_url must be an absolute http(s) URL, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("data_plane_url")
    @classmethod
    def _validate_data_plane_url(cls, value: str | None) -> str | None:
        """Reject a data-plane address that is not an absolute http(s) URL.

        Held to the same bar as ``docs_url`` and for the same reason: the
        deployment bootstrap publishes it to the browser, which builds a runnable
        snippet from it. A typo should be a startup error rather than a curl
        command an operator copies and cannot explain.

        The trailing slash is normalized away here so no consumer has to: the
        dashboard suffixes this with ``/v1``, and ``https://host//v1`` is a
        different path on a strict router.

        A query string or a fragment is refused for the same reason, and it is
        the case a scheme check alone would miss: this is a base URL a client
        appends to, so ``https://host?trace=1`` would put ``/v1/chat/completions``
        inside the query value rather than in the path, and the snippet would
        reach the deployment's root with a very strange parameter. Unlike
        ``docs_url``, which is a link a person follows, nothing downstream can
        recover from that.
        """
        normalized = (value or "").strip().rstrip("/")
        if not normalized:
            return None
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = f"data_plane_url must be an absolute http(s) URL, got '{value}'"
            raise ValueError(msg)
        if parsed.query or parsed.fragment:
            msg = f"data_plane_url must carry no query string or fragment, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("mail_transport")
    @classmethod
    def _validate_mail_transport(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in MAIL_TRANSPORT_SETTINGS:
            msg = f"mail_transport must be one of {sorted(MAIL_TRANSPORT_SETTINGS)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("vision_strategy")
    @classmethod
    def _validate_vision_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VISION_STRATEGIES:
            msg = f"vision_strategy must be one of {sorted(VISION_STRATEGIES)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("router_granularity")
    @classmethod
    def _validate_router_granularity(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ROUTER_GRANULARITIES:
            msg = f"router_granularity must be one of {sorted(ROUTER_GRANULARITIES)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("platform")
    @classmethod
    def _validate_platform_streaming_timeouts(cls, platform: dict[str, Any]) -> dict[str, Any]:
        """Reject non-sensical platform timeout settings at load time.

        The per-attempt first-chunk and inline-settlement budgets must be
        positive. The terminal-attempt extra grace must be non-negative, since
        it is added on top of the first-chunk budget.
        """
        inline_key = "usage_inline_timeout_ms"
        if inline_key in platform:
            raw_inline_timeout = platform[inline_key]
            try:
                inline_timeout = int(raw_inline_timeout)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{inline_key} must be a positive integer, got {raw_inline_timeout!r}"
                ) from None
            if (
                isinstance(raw_inline_timeout, bool)
                or (isinstance(raw_inline_timeout, float) and not raw_inline_timeout.is_integer())
                or inline_timeout <= 0
            ):
                raise ValueError(f"{inline_key} must be a positive integer, got {raw_inline_timeout!r}")
            platform[inline_key] = inline_timeout

        positive_ms_keys = (
            "streaming_first_chunk_timeout_ms",
            "streaming_first_chunk_timeout_ms_tool_loop",
        )
        for key in positive_ms_keys:
            if key in platform and float(platform[key]) <= 0:
                raise ValueError(f"{key} must be > 0, got {platform[key]}")
        extra_key = "streaming_final_attempt_extra_first_chunk_timeout_ms"
        if extra_key in platform and float(platform[extra_key]) < 0:
            raise ValueError(f"{extra_key} must be >= 0, got {platform[extra_key]}")
        return platform

    def validate_mail_transport(self) -> None:
        """Refuse a mail transport the deployment cannot actually run.

        An operator who wrote ``mail_transport: smtp`` asked for delivery, so a
        missing host or from-address is a misconfiguration and startup says so.
        That is the difference from the ``auto`` default, where the same two
        fields being unset is the ordinary state of a deployment that wants no
        mail: there, nothing is wrong and nothing is refused. Either way the
        failure never waits until someone presses Invite.
        """
        configured = self.mail_transport.strip().lower()
        if configured == "console":
            # Once, at load, rather than per send: an operator who selected this
            # deliberately does not need it repeated, but one who inherited it
            # from a copied config needs to see it at least once, because the
            # log it writes carries the token in every link it "delivers".
            logger.warning(
                "mail_transport is 'console': outgoing mail is written to the log, token-bearing "
                "links included, and delivered to nobody. Use 'smtp' to actually send."
            )
            return
        # The *configured* value, not the effective one: an explicit 'smtp' that
        # cannot be built now resolves to 'none', which is precisely the state
        # this refuses. Keying on the effective value would make the check
        # silently vacuous.
        if configured != "smtp":
            return
        required = (("smtp_host", self.smtp_host), ("mail_from_email", self.mail_from_email))
        missing = [name for name, value in required if not value]
        if missing:
            msg = (
                f"mail_transport 'smtp' requires {' and '.join(missing)} to be set. "
                "Set them, or leave mail_transport at its 'auto' default to run without mail."
            )
            raise ValueError(msg)
        # Shape-checked here for the same reason the transport is: a typo in the
        # from-address is a misconfiguration, and letting it through means the
        # first anyone hears of it is a recipient's server rejecting the
        # envelope, which is the send-time failure this design exists to avoid.
        # Every recipient address already goes through the same check.
        #
        # Only under an explicit 'smtp', never under 'auto'. That asymmetry is
        # deliberate and costs nothing: mail_transport is new here, so no
        # existing deployment can be holding the value this refuses, while a
        # deployment that has had a working odd-looking address under the
        # implicit path keeps booting.
        if self.mail_from_email and normalized_address(self.mail_from_email) is None:
            msg = f"mail_from_email is not a valid email address: {self.mail_from_email!r}"
            raise ValueError(msg)

    def warn_about_half_configured_oauth(self) -> None:
        """Say so when OAuth client credentials were set but cannot be used.

        A warning rather than a refusal, unlike
        :meth:`validate_webauthn_relying_party`: nothing here is *wrong*, and
        refusing to boot would take a gateway offline over a sign-in method
        that is optional. But the failure is otherwise completely silent. The
        provider is absent from ``GET /v1/bootstrap``, the sign-in screen simply
        does not draw its button, and an operator who set two of the three
        settings has nothing anywhere telling them why the button they
        configured never appeared.

        A deployment that configured nothing says nothing, for the reason
        ``validate_webauthn_relying_party`` gives about its own absent case:
        that is the ordinary state, not a mistake.
        """
        for provider in OAUTH_PROVIDERS:
            client_id = getattr(self, f"oauth_{provider}_client_id", None)
            client_secret = getattr(self, f"oauth_{provider}_client_secret", None)
            if not client_id and not client_secret:
                continue
            missing = [
                name
                for name, value in (
                    (f"oauth_{provider}_client_id", client_id),
                    (f"oauth_{provider}_client_secret", client_secret),
                    ("public_base_url", self.public_base_url),
                )
                if not value
            ]
            if missing:
                logger.warning(
                    "%s sign-in is configured but will not be offered: %s %s not set. "
                    "The sign-in screen shows no %s button until it is.",
                    provider,
                    ", ".join(missing),
                    "is" if len(missing) == 1 else "are",
                    provider,
                )

    def validate_webauthn_relying_party(self) -> None:
        """Refuse a passkey configuration a browser would reject anyway.

        Only what was written explicitly is refused. A deployment that
        configured nothing has no relying party, offers no passkeys, and is not
        misconfigured, which is why the absence of one is not an error here.

        Two mistakes are worth catching at load rather than in a browser
        console, because both fail as an opaque ``SecurityError`` on the page
        with nothing on the server to correlate it with: a relying-party ID
        written as a URL rather than a domain, and an allowed origin that is
        neither the relying-party ID nor a subdomain of it. The second is the
        one that looks fine: 'example.com' with an origin of
        'https://otari.example.net' is a plausible pair of settings that can
        never complete a ceremony.
        """
        configured_id = (self.webauthn_rp_id or "").strip()
        if configured_id and _host_of(f"//{configured_id}") != configured_id.lower():
            # Two spellings reach here and neither parses under one form. A value
            # carrying a scheme ('https://example.com') needs to be read as the
            # URL it is; one carrying only a port ('localhost:8000') parses as a
            # scheme and a path unless '//' is prepended first. So both are
            # tried, in that order, and the placeholder is the last resort
            # rather than the answer to the commonest mistake.
            suggestion = _host_of(configured_id) or _host_of(f"//{configured_id}") or "otari.example.com"
            msg = (
                f"webauthn_rp_id must be a bare domain with no scheme, port or path, got {configured_id!r}. "
                f"Use {suggestion!r}."
            )
            raise ValueError(msg)
        relying_party = self.webauthn_relying_party
        if relying_party is None:
            if self.webauthn_allowed_origins and not self.public_base_url and not configured_id:
                msg = (
                    "webauthn_allowed_origins is set but no relying-party ID can be derived. "
                    "Set public_base_url, or webauthn_rp_id."
                )
                raise ValueError(msg)
            return
        # Checked before coverage, because a scheme-less entry fails coverage
        # too and the coverage message would name the wrong setting:
        # 'otari.example.com' really is a subdomain of 'example.com', so telling
        # the operator to widen webauthn_rp_id would send them to fix something
        # that is not broken. An origin is a scheme and a host, and this is the
        # one that is missing.
        schemeless = [origin for origin in relying_party.origins if "://" not in origin]
        if schemeless:
            msg = (
                f"webauthn_allowed_origins entries {schemeless} are missing a scheme. "
                "An origin is a scheme and a host, so write 'https://otari.example.com' rather "
                "than 'otari.example.com'."
            )
            raise ValueError(msg)
        stray = [origin for origin in relying_party.origins if not relying_party.covers(origin)]
        if stray:
            msg = (
                f"webauthn_allowed_origins entries {stray} are not the relying-party ID "
                f"{relying_party.rp_id!r} or a subdomain of it, so a passkey ceremony from them "
                "would be refused by the browser. Set webauthn_rp_id to a domain that covers them."
            )
            raise ValueError(msg)

    def validate_mode_selection(self) -> None:
        configured_mode = self.configured_mode
        # Mode unset: the runtime mode is derived from the token, so there is
        # nothing to assert here.
        if configured_mode is None:
            return
        # "platform" is the legacy alias for "hybrid" (the otari.ai-connected
        # runtime mode); accept it so pre-rename configs keep working.
        if configured_mode not in {"standalone", "hosted", "hybrid", "platform"}:
            msg = (
                "Invalid mode (set via OTARI_MODE or the config 'mode' field). "
                "Expected 'standalone', 'hosted' or 'hybrid'."
            )
            raise ValueError(msg)

        token_present = self.platform_token is not None
        if configured_mode in {"hybrid", "platform"} and not token_present:
            msg = "Hybrid mode (legacy value 'platform') requires OTARI_AI_TOKEN to be set."
            raise ValueError(msg)
        # Both local-control-plane modes conflict with the token for the same
        # reason: a deployment that holds its own management API is not also a
        # data plane reporting to somebody else's.
        if configured_mode in {"standalone", "hosted"} and token_present:
            msg = (
                f"{configured_mode.capitalize()} mode conflicts with OTARI_AI_TOKEN being set: the token "
                "selects hybrid mode. Unset the token to run a control plane of your own, or clear the "
                "mode setting to let the token select hybrid mode."
            )
            raise ValueError(msg)


def _load_structured_env_config() -> dict[str, Any] | None:
    """Parse a full YAML config supplied through the environment.

    Reads ``OTARI_CONFIG_YAML`` (raw YAML) or ``OTARI_CONFIG_B64`` (base64-encoded
    YAML). This lets PaaS deployments reach the entire config schema, including
    the non-scalar ``providers`` and ``pricing`` fields, without mounting a
    ``config.yml``. Raw YAML wins when both are set. ``${VAR}`` references are
    resolved exactly as in a config file. Returns the parsed mapping, or ``None``
    when neither variable is set or the content is empty. Raises ``ValueError``
    with a clear message on invalid base64, invalid YAML, or a non-mapping top
    level, so startup fails fast.
    """
    raw = os.getenv(OTARI_CONFIG_YAML_ENV)
    source = OTARI_CONFIG_YAML_ENV
    if not (raw and raw.strip()):
        encoded = os.getenv(OTARI_CONFIG_B64_ENV)
        if not (encoded and encoded.strip()):
            return None
        source = OTARI_CONFIG_B64_ENV
        # Strip whitespace first: the standard `base64` CLI and many env-var UIs
        # wrap output at 76 columns, and validate=True would reject those newlines
        # while still catching genuinely invalid characters.
        try:
            raw = base64.b64decode("".join(encoded.split()), validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
            msg = f"{OTARI_CONFIG_B64_ENV} is not valid base64-encoded UTF-8: {exc}"
            raise ValueError(msg) from exc

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"{source} is not valid YAML: {exc}"
        raise ValueError(msg) from exc

    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        msg = f"{source} must contain a YAML mapping at the top level, got {type(parsed).__name__}."
        raise ValueError(msg)

    return _resolve_env_vars(parsed)


def load_config(config_path: str | None = None) -> GatewayConfig:
    """Load configuration from file and environment variables.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        GatewayConfig instance with merged configuration

    """
    _load_dotenv(config_path)

    config_dict: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = _resolve_env_vars(yaml_config)

    structured_env_config = _load_structured_env_config()
    if structured_env_config:
        config_dict.update(structured_env_config)

    # Snapshot which bridged fields the YAML config set, before the env
    # overrides below inject OTARI_ values into the same dict: only YAML-set
    # values need bridging into the environment (env-set values are already
    # visible to the otari_env() read sites, with unchanged semantics).
    yaml_bridged_fields = {name for name in ENV_BRIDGED_FIELDS if config_dict.get(name) is not None}

    _apply_otari_env_overrides(config_dict)
    _apply_platform_env_overrides(config_dict)

    config = GatewayConfig(**config_dict)
    # Resolve and cache the platform token once, at load time, so the runtime
    # mode is fixed for the process instead of re-derived from os.getenv on
    # every property read.
    config._resolve_platform_token()
    config.validate_mode_selection()
    config.validate_provider_instances()
    config.validate_aliases()
    config.validate_routing_policies()
    config.validate_search_tools()
    config.validate_mail_transport()
    config.validate_webauthn_relying_party()
    config.warn_about_half_configured_oauth()
    _bridge_yaml_fields_to_env(config, yaml_bridged_fields)
    return config


def parse_bool_env(value: str) -> bool:
    """Parse a boolean environment-variable string, raising on an unknown spelling.

    Shared so the config layer and the ``url_safety`` SSRF gates agree on the
    accepted truthy/falsey spellings (notably ``on``/``off``): a gate that parsed
    booleans differently could silently fall open on a spelling one side rejects.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    msg = f"Invalid boolean value for environment variable: {value!r}"
    raise ValueError(msg)


def _coerce_scalar_env(value: str, annotation: Any) -> Any:
    """Coerce an env-var string to a scalar field type, or raise _NonScalarField."""
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(non_none) != 1:
            raise _NonScalarField
        annotation = non_none[0]
        origin = typing.get_origin(annotation)
    if origin is not None:
        raise _NonScalarField  # parameterized generics (list[...], dict[...]) are not scalars
    if annotation is bool:
        return parse_bool_env(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return value
    raise _NonScalarField


def _bridge_yaml_fields_to_env(config: GatewayConfig, yaml_set_fields: set[str]) -> None:
    """Bridge YAML-set promoted fields into the process env for otari_env() readers.

    The runtime read sites for ENV_BRIDGED_FIELDS call ``otari_env()``, which
    only sees environment variables. ``os.environ.setdefault`` keeps env
    precedence intact: an ``OTARI_<FIELD>`` variable that is already set always
    wins, and fields not set in YAML are left untouched, so pure-env
    deployments behave byte-for-byte as before. Values are serialized from the
    validated field, with booleans lowercased to ``true``/``false``, the
    spellings every otari_env() consumer parses.
    """
    for field_name in yaml_set_fields:
        value = getattr(config, field_name)
        if value is None:
            continue
        serialized = ("true" if value else "false") if isinstance(value, bool) else str(value)
        os.environ.setdefault(f"{OTARI_ENV_PREFIX}{field_name.upper()}", serialized)


def _apply_otari_env_overrides(config: dict[str, Any]) -> None:
    """Layer OTARI_<FIELD> env vars over the config dict for every scalar field.

    Written into the init dict so they take precedence over YAML (which pydantic's
    native OTARI_ env prefix, applied after init, cannot override). Complex fields
    (lists/dicts) are left to YAML and pydantic's native env handling.
    """
    for field_name in GatewayConfig.model_fields:
        value = os.getenv(f"{OTARI_ENV_PREFIX}{field_name.upper()}")
        if value is None or value == "":
            continue
        try:
            config[field_name] = _coerce_scalar_env(value, GatewayConfig.model_fields[field_name].annotation)
        except _NonScalarField:
            continue


def _apply_platform_env_overrides(config: dict[str, Any]) -> None:
    platform = config.get("platform")
    if not isinstance(platform, dict):
        platform = {}

    env_mappings: dict[str, tuple[str, type[Any]]] = {
        "PLATFORM_BASE_URL": ("base_url", str),
        "PLATFORM_MANAGEMENT_URL": ("management_url", str),
        "PLATFORM_RESOLVE_TIMEOUT_MS": ("resolve_timeout_ms", int),
        "PLATFORM_USAGE_TIMEOUT_MS": ("usage_timeout_ms", int),
        # Budget for the one usage report the response path waits on. Expiry
        # detaches the wait without cancelling the accounting report.
        "PLATFORM_USAGE_INLINE_TIMEOUT_MS": ("usage_inline_timeout_ms", int),
        "PLATFORM_USAGE_MAX_RETRIES": ("usage_max_retries", int),
        # Per-attempt budget for streaming fallback: how long to wait for the
        # first chunk from each attempt before treating it as hung and moving
        # to the next entry in the routing policy. Tunable per deployment;
        # v1.2 will move this onto the routing_policy schema for per-policy
        # control.
        "STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS": (
            "streaming_first_chunk_timeout_ms",
            int,
        ),
        # Extra first-chunk grace for the sole/final streaming attempt, added on
        # top of the per-attempt budget above. The failover budget exists to move
        # to the next routing-policy entry when an attempt is slow; the final
        # attempt has no next entry, so this keeps its wait bounded without cutting
        # off a slow-but-valid first token.
        "STREAMING_FALLBACK_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS": (
            "streaming_final_attempt_extra_first_chunk_timeout_ms",
            int,
        ),
    }

    for env_name, (field_name, caster) in env_mappings.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        platform[field_name] = caster(value)

    configured_mode = str(config.get("mode") or "").strip().lower()
    platform_requested = configured_mode in {"hybrid", "platform"} or _get_platform_token_from_env() is not None
    if platform_requested and not platform.get("base_url"):
        platform["base_url"] = DEFAULT_PLATFORM_BASE_URL

    if platform:
        config["platform"] = platform


def _load_dotenv(config_path: str | None = None) -> None:
    """Load .env files into process environment without overriding existing vars."""
    candidate_paths: list[Path] = [Path.cwd() / ".env"]
    if config_path:
        candidate_paths.insert(0, Path(config_path).resolve().parent / ".env")

    seen: set[Path] = set()
    for dotenv_path in candidate_paths:
        if dotenv_path in seen or not dotenv_path.exists():
            continue
        seen.add(dotenv_path)
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve environment variable references in config.

    Supports ${VAR_NAME} syntax in string values.

    Raises:
        ValueError: If an environment variable reference cannot be resolved

    """
    if isinstance(config, dict):
        return {key: _resolve_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:

        def _replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                msg = f"Environment variable '{env_var}' is not set (referenced in config as '${{{env_var}}}')"
                raise ValueError(msg)
            return value

        return re.sub(r"\$\{([^}]+)}", _replace, config)
    return config
