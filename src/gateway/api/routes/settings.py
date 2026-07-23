"""Operator settings for the admin dashboard.

Exposes a non-secret slice of gateway configuration so the dashboard can explain
and, where safe, control runtime behavior. Two layers:

* A read-only "full config" view (``config``): every effective setting the
  operator should see, grouped, each marked ``settable`` (hot-changeable) or not
  (startup-only, display only). Secrets (master key, provider credentials) and
  the complex catalog fields managed on their own pages (providers, pricing,
  aliases, model capabilities, the platform block) are deliberately omitted.
* The writable subset (see ``runtime_settings_service.SETTABLE_KEYS``), persisted
  so a change survives a restart and applied to the running worker immediately.
"""

import types
import typing
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.services.dashboard_session_service import (
    SESSION_COOKIE_NAME,
    apply_session_cookie,
    create_dashboard_session,
    revoke_all_dashboard_sessions,
)
from gateway.services.master_key_service import (
    MasterKeyRotationConflictError,
    hash_master_key,
    load_master_key_hash,
    stage_generated_master_key_rotation,
)
from gateway.services.runtime_settings_service import (
    SETTABLE_KEYS,
    SettingValue,
    apply_override,
    settable_bounds,
    settable_options,
    stage_override,
)
from gateway.version import __version__

router = APIRouter(prefix="/v1/settings", tags=["settings"])

# The effective-config view, in display order. Each entry is a group label and
# the config field names shown under it. Only non-secret scalar fields appear;
# the complex catalog fields (providers, pricing, aliases, model_capabilities,
# platform) live on their own dashboard pages, and master_key is a secret.
# A field's ``settable`` flag is derived from SETTABLE_KEYS, so a field not in
# that tuple renders read-only ("startup-only") automatically.
_CONFIG_VIEW: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Server & database",
        (
            "host",
            "port",
            "database_url",
            "mode",
            "auto_migrate",
            "db_pool_size",
            "db_max_overflow",
            "db_pool_timeout",
            "db_pool_recycle",
        ),
    ),
    (
        "Metering & budgets",
        (
            "require_pricing",
            "default_pricing",
            "reject_user_mismatch",
            "stream_missing_usage_policy",
            "budget_strategy",
            "budget_estimate_default_output_tokens",
        ),
    ),
    (
        "Models & discovery",
        (
            "model_discovery",
            "model_cache_ttl_seconds",
            "model_discovery_timeout_seconds",
            "model_discovery_negative_ttl_seconds",
            "models_dev_metadata",
            "models_dev_cache_ttl_seconds",
        ),
    ),
    (
        "Rate limiting & CORS",
        ("rate_limit_rpm", "cors_allow_origins"),
    ),
    (
        "Files",
        ("files_enabled", "files_backend", "files_local_dir", "files_max_bytes", "files_retention_hours"),
    ),
    (
        "Vision & file understanding",
        ("file_understanding_enabled", "vision_strategy", "vision_describe_model", "vision_describe_max_tokens"),
    ),
    (
        "Tools & network access",
        (
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
        ),
    ),
    (
        "General",
        ("enable_metrics", "enable_docs", "bootstrap_api_key", "log_writer_strategy"),
    ),
)


class ConfigField(BaseModel):
    """One effective config value surfaced to the dashboard's config viewer."""

    key: str
    value: bool | int | float | str | list[str] | None
    type: Literal["bool", "int", "float", "str", "list"]
    settable: bool
    group: str
    description: str | None = None
    options: list[str] | None = None
    # Numeric lower bounds (settable numeric fields only), so the dashboard can
    # gate a number input the same way the backend validator does.
    minimum: float | None = None  # inclusive (ge)
    exclusive_minimum: float | None = None  # gt


class GatewaySettings(BaseModel):
    """Operator-facing runtime settings surfaced to the dashboard.

    The top-level flags are kept for the callers that read them directly
    (pricing warnings, the models/providers pages); ``config`` is the full
    read-only view with the settable/startup-only marking.
    """

    mode: str
    version: str
    model_discovery: bool
    default_pricing: bool
    require_pricing: bool
    master_key_source: Literal["configured", "generated"] = Field(
        description="Whether the dashboard master key is configured at startup or generated and stored by Otari."
    )
    config: list[ConfigField]


class UpdateSettingsRequest(BaseModel):
    """Change one or more runtime settings. Omitted fields are left unchanged.

    Only the hot-changeable subset is accepted; startup-only fields are not
    settable over the API (they are display-only in the dashboard).
    """

    model_discovery: bool | None = None
    default_pricing: bool | None = None
    require_pricing: bool | None = None
    reject_user_mismatch: bool | None = None
    models_dev_metadata: bool | None = None
    file_understanding_enabled: bool | None = None
    model_cache_ttl_seconds: int | None = Field(default=None, ge=0)
    models_dev_cache_ttl_seconds: int | None = Field(default=None, ge=0)
    vision_describe_max_tokens: int | None = Field(default=None, gt=0)
    budget_estimate_default_output_tokens: int | None = Field(default=None, ge=0)
    model_discovery_timeout_seconds: float | None = Field(default=None, gt=0)
    model_discovery_negative_ttl_seconds: float | None = Field(default=None, ge=0)
    stream_missing_usage_policy: Literal["estimate", "fail", "allow_free"] | None = None
    vision_strategy: Literal["describe", "ocr", "off"] | None = None
    # Nullable: an explicit ``null`` clears the describe model. "Provided" is
    # detected via ``model_fields_set``, so an omitted field is left unchanged
    # while an explicit null unsets it.
    vision_describe_model: str | None = None


class RotateMasterKeyResponse(BaseModel):
    """A newly generated dashboard master key, returned once."""

    master_key: str = Field(description="The new plaintext master key. Store it now; it is never returned again.")


def _scalar_type_name(annotation: Any) -> Literal["bool", "int", "float", "str", "list"]:
    """Map a config field's annotation to a display type, unwrapping Optional."""
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _scalar_type_name(args[0])
    if origin is list:
        return "list"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


# URL-valued fields whose value may embed credentials (a DB password, a bearer
# token in the userinfo or a query param). They are shown so an operator can
# confirm the host/db, but any secret is masked so it is never echoed into the
# API response or the browser's query cache, even for the master-key holder.
_REDACTED_URL_FIELDS = frozenset({"database_url", "sandbox_url", "guardrails_url"})


def _redact_url_secrets(value: str) -> str:
    """Mask credentials in a URL while keeping its shape recognizable.

    A ``user:pass@`` password is masked while the username stays visible. A
    single-component userinfo (``token@host``) is masked whole, since a lone
    userinfo component is more likely a bearer token than a benign username and
    the two cannot be told apart. Every query-parameter *value* is masked too
    (a denylist of "credential-looking" keys cannot be complete, and a config
    viewer cannot know which value is a secret), while the keys stay visible so
    the operator can still see what is set. Scheme, host, and path are preserved.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return value

    netloc = parts.netloc
    if parts.username is not None or parts.password is not None:
        host = parts.hostname or ""
        # An IPv6 literal must keep its brackets or the rebuilt URL is malformed.
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        if parts.password is not None:
            netloc = f"{parts.username or ''}:***@{host}"
        else:
            netloc = f"***@{host}"

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        # quote_via with '*' left safe keeps the mask readable ('***', not '%2A%2A%2A').
        query = urlencode([(key, "***") for key, _ in pairs], quote_via=quote, safe="*")

    if netloc == parts.netloc and query == parts.query:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _field_value(config: GatewayConfig, name: str) -> bool | int | float | str | list[str] | None:
    # ``mode`` is often unset (None) with the real mode derived from the platform
    # token; show the effective mode so the viewer is not misleading.
    if name == "mode":
        return config.effective_mode
    value: bool | int | float | str | list[str] | None = getattr(config, name)
    if name in _REDACTED_URL_FIELDS and isinstance(value, str):
        return _redact_url_secrets(value)
    return value


def _config_fields(config: GatewayConfig) -> list[ConfigField]:
    fields: list[ConfigField] = []
    for group, names in _CONFIG_VIEW:
        for name in names:
            field = GatewayConfig.model_fields[name]
            options = settable_options(name)
            minimum, exclusive_minimum = settable_bounds(name)
            fields.append(
                ConfigField(
                    key=name,
                    value=_field_value(config, name),
                    type=_scalar_type_name(field.annotation),
                    settable=name in SETTABLE_KEYS,
                    group=group,
                    description=field.description,
                    options=list(options) if options is not None else None,
                    minimum=minimum,
                    exclusive_minimum=exclusive_minimum,
                )
            )
    return fields


def _current_settings(config: GatewayConfig) -> GatewaySettings:
    return GatewaySettings(
        mode=config.effective_mode,
        version=__version__,
        model_discovery=config.model_discovery,
        default_pricing=config.default_pricing,
        require_pricing=config.require_pricing,
        master_key_source="configured" if config.master_key is not None else "generated",
        config=_config_fields(config),
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def get_settings(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Return non-secret runtime settings for the admin dashboard."""
    return _current_settings(config)


@router.patch("", dependencies=[Depends(verify_master_key)])
async def update_settings(
    request: UpdateSettingsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Persist and apply runtime setting changes.

    Each provided field is stored as an override (winning over config/env) and
    applied to the running gateway immediately. Master-key gated: these change
    how the gateway meters and lists models.
    """
    # Every field on the request maps 1:1 onto a settable key. Use the set of
    # fields actually present in the body (not a None check) so an explicit
    # ``null`` can clear a nullable field (vision_describe_model) while an omitted
    # field is left unchanged.
    updates: dict[str, SettingValue] = {
        key: getattr(request, key) for key in request.model_fields_set if key in SETTABLE_KEYS
    }

    if updates:
        try:
            normalized = {key: await stage_override(db, key, value) for key, value in updates.items()}
            await db.commit()
        except ValueError as exc:
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
        except SQLAlchemyError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error",
            ) from None
        # Apply only after the write has committed, so a failed commit never
        # leaves this worker metering or listing against an unpersisted value.
        for key, value in normalized.items():
            apply_override(config, key, value)

    return _current_settings(config)


@router.post("/master-key/rotate", dependencies=[Depends(verify_master_key)])
async def rotate_master_key(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    authenticated_key: Annotated[str | None, Depends(verify_master_key)],
) -> RotateMasterKeyResponse:
    """Regenerate the database-backed master key and invalidate the old one.

    Only the first-run generated master key can be rotated here. When a master
    key is supplied through config or ``OTARI_MASTER_KEY``, the dashboard cannot
    invalidate it; the operator must change that value and restart instead.

    Every dashboard session is revoked with the rotation (a session only proves
    possession of the now-dead key); the caller's own session is re-minted under
    the new key so the tab that performed the rotation stays signed in.
    """
    if config.master_key is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This gateway uses a configured master key; change OTARI_MASTER_KEY or config.yml and restart.",
        )
    # A cookie-authenticated caller has no raw key to hash; the stored hash is
    # the same value, so the rotation CAS still rejects a concurrent rotation.
    if authenticated_key is not None:
        current_hash = hash_master_key(authenticated_key)
    else:
        stored_hash = await load_master_key_hash(db)
        if stored_hash is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The master key was already rotated. Reload and try again.",
            )
        current_hash = stored_hash
    session_token: str | None = None
    session_expires_at = None
    try:
        token, hashed = await stage_generated_master_key_rotation(db, current_hash)
        await revoke_all_dashboard_sessions(db)
        if SESSION_COOKIE_NAME in request.cookies:
            session_token, session_expires_at = await create_dashboard_session(
                db, config.dashboard_session_ttl_hours
            )
        await db.commit()
    except MasterKeyRotationConflictError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    config._master_key_hash = hashed
    if session_token is not None and session_expires_at is not None:
        apply_session_cookie(response, session_token, session_expires_at, secure=request.url.scheme == "https")
    return RotateMasterKeyResponse(master_key=token)
