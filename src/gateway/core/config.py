import base64
import binascii
import os
import re
import types
import typing
from collections.abc import Container
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

# Search providers the standalone POST /v1/search endpoint can dispatch to.
# Declared here rather than in the adapter module so startup validation can
# reject an unknown ``search_tools.<name>.provider`` without the config layer
# importing the service layer.
SEARCH_PROVIDERS = ("exa",)


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
    rate_limit_rpm: int | None = Field(
        default=None, ge=1, description="Maximum requests per minute per user (None disables rate limiting)"
    )
    dashboard_login_rate_limit_per_minute: int | None = Field(
        default=10,
        ge=1,
        description=(
            "Maximum failed POST /v1/auth/session attempts per client IP per minute "
            "(None disables this limit). Only failed attempts count, so a correct "
            "master key is never throttled. Separate from rate_limit_rpm, which is "
            "keyed to authenticated users and does not cover this pre-auth path."
        ),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="Allowed CORS origins (empty list disables CORS)"
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
            "'search_tool_name' (or in the /v1/search/{tool} path). Each entry needs an "
            "'api_key' and may declare a 'provider' (one of: exa; defaults to the tool "
            "name), an 'api_base', a 'timeout' in seconds, and an 'options' mapping of "
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
            "Otari operating mode: 'standalone' or 'hybrid'. When unset (the default), the mode is "
            "derived from the platform token: hybrid if a token is present (OTARI_AI_TOKEN), else "
            "standalone. Set explicitly to assert the intended mode: 'hybrid' requires a token, and "
            "'standalone' with a token present is rejected at startup as conflicting configuration. "
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
    def configured_mode(self) -> str | None:
        """The explicitly set mode (normalized), or None when unset/blank."""
        normalized = (self.mode or "").strip().lower()
        return normalized or None

    @property
    def effective_mode(self) -> str:
        configured = self.configured_mode
        if configured in {"hybrid", "platform"}:
            return "hybrid"
        if configured == "standalone":
            return "standalone"
        # Mode unset: derive from the platform token.
        return "hybrid" if self.platform_token else "standalone"

    @property
    def is_hybrid_mode(self) -> bool:
        return self.effective_mode == "hybrid"

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

    def validate_search_tools(self) -> None:
        """Validate the ``search_tools`` map at startup so misconfig fails fast.

        Every supported provider authenticates with an API key, so a tool
        without one is rejected here rather than at request time as an opaque
        upstream 401. The tool name doubles as a ``/v1/search/{tool}`` path
        segment, so it must not contain a slash.
        """
        for name, entry in self.search_tools.items():
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
            if not entry.get("api_key"):
                msg = f"search_tools.{name}.api_key is required."
                raise ValueError(msg)
            timeout = entry.get("timeout")
            if timeout is not None:
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                    msg = f"search_tools.{name}.timeout must be a number of seconds."
                    raise ValueError(msg)
                # A negative timeout would reach httpx and fail at request time,
                # and a zero is silently swapped for the default when the tool is
                # resolved. Both are misconfigurations worth failing on here.
                if timeout <= 0:
                    msg = f"search_tools.{name}.timeout must be greater than 0 seconds, got {timeout}."
                    raise ValueError(msg)
            options = entry.get("options")
            if options is not None and not isinstance(options, dict):
                msg = f"search_tools.{name}.options must be a mapping."
                raise ValueError(msg)

    @field_validator("stream_missing_usage_policy")
    @classmethod
    def _validate_stream_missing_usage_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in STREAM_MISSING_USAGE_POLICIES:
            msg = f"stream_missing_usage_policy must be one of {sorted(STREAM_MISSING_USAGE_POLICIES)}, got '{value}'"
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
        """Reject non-sensical streaming first-chunk timeout settings at load time.

        The per-attempt first-chunk budgets must be positive: a zero or negative
        wait would treat every attempt as instantly hung. The terminal-attempt
        extra grace must be non-negative, since it is added on top of the budget.
        """
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

    def validate_mode_selection(self) -> None:
        configured_mode = self.configured_mode
        # Mode unset: the runtime mode is derived from the token, so there is
        # nothing to assert here.
        if configured_mode is None:
            return
        # "platform" is the legacy alias for "hybrid" (the otari.ai-connected
        # runtime mode); accept it so pre-rename configs keep working.
        if configured_mode not in {"standalone", "hybrid", "platform"}:
            msg = (
                "Invalid mode (set via OTARI_MODE or the config 'mode' field). "
                "Expected 'standalone' or 'hybrid'."
            )
            raise ValueError(msg)

        token_present = self.platform_token is not None
        if configured_mode in {"hybrid", "platform"} and not token_present:
            msg = "Hybrid mode (legacy value 'platform') requires OTARI_AI_TOKEN to be set."
            raise ValueError(msg)
        if configured_mode == "standalone" and token_present:
            msg = (
                "Standalone mode conflicts with OTARI_AI_TOKEN being set: the token selects hybrid "
                "mode. Unset the token to run standalone, or clear the mode setting to let the token "
                "select hybrid mode."
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
        "PLATFORM_RESOLVE_TIMEOUT_MS": ("resolve_timeout_ms", int),
        "PLATFORM_USAGE_TIMEOUT_MS": ("usage_timeout_ms", int),
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
