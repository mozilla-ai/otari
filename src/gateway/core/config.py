import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "Otari-Key"
LEGACY_API_KEY_HEADERS = ("AnyLLM-Key", "X-AnyLLM-Key")  # Back-compat aliases; prefer API_KEY_HEADER.
DEFAULT_PLATFORM_BASE_URL = "https://api.otari.ai/api/v1"
PLATFORM_TOKEN_ENV_VARS = (
    "OTARI_AI_TOKEN",
    "OTARI_PLATFORM_TOKEN",
    "ANY_LLM_PLATFORM_TOKEN",
)
OTARI_ENV_ALIASES_TO_GATEWAY = {
    "OTARI_MASTER_KEY": ("master_key", str),
    "OTARI_DATABASE_URL": ("database_url", str),
    "OTARI_HOST": ("host", str),
    "OTARI_PORT": ("port", int),
    "OTARI_AUTO_MIGRATE": ("auto_migrate", bool),
    "OTARI_BOOTSTRAP_API_KEY": ("bootstrap_api_key", bool),
}


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


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite:///./otari-gateway.db",
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
        default_factory=dict, description="Pre-configured provider credentials"
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
    budget_alert_webhook_retry_interval_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Seconds between background retries for failed/pending budget alert webhooks. 0 disables.",
    )
    budget_alert_webhook_retry_max_attempts: int = Field(
        default=5,
        ge=1,
        description="Maximum delivery attempts for background budget alert webhook retries.",
    )
    budget_alert_webhook_retry_backoff_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Base seconds for budget alert webhook retry backoff after a failed retry.",
    )
    budget_alert_webhook_retry_max_backoff_seconds: float = Field(
        default=3600.0,
        ge=0,
        description="Maximum seconds between budget alert webhook retry attempts. 0 disables the cap.",
    )
    budget_alert_webhook_retry_batch_size: int = Field(
        default=50,
        ge=1,
        description="Maximum budget alert webhooks to retry per background worker pass.",
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
    mode: str = Field(default="standalone", description="Gateway operating mode: standalone or platform")
    platform: dict[str, Any] = Field(default_factory=dict, description="Platform integration settings")

    @property
    def platform_token(self) -> str | None:
        return _get_platform_token_from_env()

    @property
    def effective_mode(self) -> str:
        if self.platform_token:
            return "platform"
        return "standalone"

    @property
    def is_platform_mode(self) -> bool:
        return self.effective_mode == "platform"

    def validate_mode_selection(self) -> None:
        configured_mode = self.mode.strip().lower()
        if configured_mode not in {"standalone", "platform"}:
            msg = "Invalid GATEWAY_MODE value. Expected 'standalone' or 'platform'."
            raise ValueError(msg)

        token_present = self.platform_token is not None
        if configured_mode == "platform" and not token_present:
            msg = (
                "GATEWAY_MODE=platform requires OTARI_AI_TOKEN to be set "
                "(legacy aliases: OTARI_PLATFORM_TOKEN, ANY_LLM_PLATFORM_TOKEN)."
            )
            raise ValueError(msg)


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

    _apply_otari_env_overrides(config_dict)
    _apply_platform_env_overrides(config_dict)

    config = GatewayConfig(**config_dict)
    config.validate_mode_selection()
    return config


def _parse_bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    msg = f"Invalid boolean value for environment variable: {value!r}"
    raise ValueError(msg)


def _apply_otari_env_overrides(config: dict[str, Any]) -> None:
    for env_name, (field_name, caster) in OTARI_ENV_ALIASES_TO_GATEWAY.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        if caster is bool:
            config[field_name] = _parse_bool_env(value)
        else:
            config[field_name] = caster(value)


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
    platform_requested = configured_mode == "platform" or _get_platform_token_from_env() is not None
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
