import base64
import binascii
import os
import re
import types
import typing
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from any_llm import LLMProvider
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "Otari-Key"
# Aliases accepted for a provider instance's ``provider_type`` that map onto a
# real any-llm implementation. The "openai-compatible" spelling mirrors the
# naming opencode / pi use for self-hosted OpenAI-compatible backends.
PROVIDER_TYPE_ALIASES = {
    "openai-compatible": "openai",
    "openai_compatible": "openai",
}
LEGACY_API_KEY_HEADERS = ("AnyLLM-Key", "X-AnyLLM-Key")  # Back-compat aliases; prefer API_KEY_HEADER.
DEFAULT_PLATFORM_BASE_URL = "https://api.otari.ai/api/v1"
PLATFORM_TOKEN_ENV_VARS = (
    "OTARI_AI_TOKEN",
    "OTARI_PLATFORM_TOKEN",
    "ANY_LLM_PLATFORM_TOKEN",
)
# User-facing config env vars use the OTARI_ prefix (e.g. OTARI_MASTER_KEY,
# OTARI_PORT). The GATEWAY_ prefix below is the legacy native pydantic prefix,
# still honored for backward compatibility; OTARI_ wins when both are set.
OTARI_ENV_PREFIX = "OTARI_"
# Full structured config supplied through the environment, for PaaS platforms
# (Railway, Render, Fly.io, Kubernetes) where mounting a config.yml is awkward.
# These carry the entire YAML schema (providers, pricing, etc.), not just the
# scalar fields reachable via OTARI_<FIELD>. Raw YAML wins when both are set.
OTARI_CONFIG_YAML_ENV = "OTARI_CONFIG_YAML"
OTARI_CONFIG_B64_ENV = "OTARI_CONFIG_B64"


class _NonScalarField(Exception):
    """Raised when a config field is not a simple scalar settable from a plain env string."""


def _get_platform_token_from_env() -> str | None:
    for env_var in PLATFORM_TOKEN_ENV_VARS:
        token = os.getenv(env_var, "").strip()
        if token:
            return token
    return None


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )


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
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
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
    rate_limit_rpm: int | None = Field(
        default=None, ge=1, description="Maximum requests per minute per user (None disables rate limiting)"
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
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description=(
            "Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})"
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
            "provider (OpenAI-style end-user tag) but spend is always bound to the key's own user — "
            "use this if clients send arbitrary 'user' values for abuse tracking. The master key may "
            "always bill an arbitrary user regardless of this setting."
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
    model_discovery: bool = Field(
        default=True,
        description="Enable auto-discovery of models from configured providers via GET /v1/models",
    )
    model_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL in seconds for the in-memory model discovery cache (0 disables caching)",
    )
    files_enabled: bool = Field(
        default=True,
        description="Enable the /v1/files upload/storage endpoints (standalone mode).",
    )
    files_backend: str = Field(
        default="local",
        description="Blob backend for uploaded file bytes: 'local' (filesystem). Future: 's3', 'gcs'.",
    )
    files_local_dir: str = Field(
        default="./otari-files",
        description="Directory for the 'local' files backend to store uploaded bytes.",
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
    mode: str = Field(default="standalone", description="Otari operating mode: standalone or hybrid")
    platform: dict[str, Any] = Field(default_factory=dict, description="otari.ai connection settings")

    @property
    def platform_token(self) -> str | None:
        return _get_platform_token_from_env()

    @property
    def effective_mode(self) -> str:
        if self.platform_token:
            return "hybrid"
        return "standalone"

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

    def validate_provider_instances(self) -> None:
        """Validate per-instance ``provider_type`` / ``models`` declarations.

        Fails fast at startup so a typo in ``provider_type`` (or a non-list
        ``models``) surfaces immediately rather than as a per-request error.
        Instances without a ``provider_type`` are left unvalidated to preserve
        the existing lenient behavior (the key is the implementation).
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

    @field_validator("stream_missing_usage_policy")
    @classmethod
    def _validate_stream_missing_usage_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"estimate", "fail", "allow_free"}
        if normalized not in allowed:
            msg = f"stream_missing_usage_policy must be one of {sorted(allowed)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    @field_validator("vision_strategy")
    @classmethod
    def _validate_vision_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"describe", "ocr", "off"}
        if normalized not in allowed:
            msg = f"vision_strategy must be one of {sorted(allowed)}, got '{value}'"
            raise ValueError(msg)
        return normalized

    def validate_mode_selection(self) -> None:
        configured_mode = self.mode.strip().lower()
        # "platform" is the legacy alias for "hybrid" (the otari.ai-connected
        # runtime mode); accept it so pre-rename configs keep working.
        if configured_mode not in {"standalone", "hybrid", "platform"}:
            msg = "Invalid OTARI_MODE value (legacy: GATEWAY_MODE). Expected 'standalone' or 'hybrid'."
            raise ValueError(msg)

        token_present = self.platform_token is not None
        if configured_mode in {"hybrid", "platform"} and not token_present:
            msg = (
                "OTARI_MODE=hybrid (legacy value: platform) requires OTARI_AI_TOKEN to be set "
                "(legacy token aliases: OTARI_PLATFORM_TOKEN, ANY_LLM_PLATFORM_TOKEN)."
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

    _apply_otari_env_overrides(config_dict)
    _apply_platform_env_overrides(config_dict)

    config = GatewayConfig(**config_dict)
    config.validate_mode_selection()
    config.validate_provider_instances()
    return config


def _parse_bool_env(value: str) -> bool:
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
        return _parse_bool_env(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return value
    raise _NonScalarField


def _apply_otari_env_overrides(config: dict[str, Any]) -> None:
    """Layer OTARI_<FIELD> env vars over the config dict for every scalar field.

    Written into the init dict so they take precedence over both YAML and the
    legacy GATEWAY_ pydantic prefix. Complex fields (lists/dicts) are left to
    YAML and pydantic's native env handling.
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
    }

    for env_name, (field_name, caster) in env_mappings.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        platform[field_name] = caster(value)

    configured_mode = str(config.get("mode", "")).strip().lower()
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
