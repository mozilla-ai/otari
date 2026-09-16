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

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_session_identity, require_deployment_operator, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.core.settings_view import derive_view
from gateway.core.surface import Surface
from gateway.models.tenancy import User as TenancyUser
from gateway.services.dashboard_session_service import (
    apply_session_cookie,
    create_dashboard_session,
    record_session_key_marker,
    request_is_https,
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
from gateway.services.secret_box import secret_box_configured
from gateway.services.url_safety import redact_url_secrets
from gateway.version import __version__

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    dependencies=[Depends(require_deployment_operator)],
)

SURFACE = Surface("settings")

# The view each field declares on itself (``core/settings_view.py``), laid out
# in display order. A field's ``settable`` flag is derived from SETTABLE_KEYS,
# so a field not in that tuple renders read-only ("startup-only") automatically.
_LAYOUT = derive_view(GatewayConfig)
_CONFIG_VIEW: tuple[tuple[str, tuple[str, ...]], ...] = _LAYOUT.shown
_DELIBERATELY_OMITTED: tuple[str, ...] = _LAYOUT.hidden


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
    secret_key_configured: bool = Field(
        description=(
            "Whether OTARI_SECRET_KEY is set on the server. Provider credentials are encrypted at rest with it, "
            "so a deployment without it can store none. The dashboard reads the same fact from the membership "
            "context's provider_key_encryption_available, because this endpoint is operator-only and the "
            "provider-key pages are read by tenants."
        )
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
    pricing_refresh: Literal["manual", "review", "auto"] | None = None
    public_catalog: bool | None = None
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


def _field_value(config: GatewayConfig, name: str) -> bool | int | float | str | list[str] | None:
    # Both are often unset with the real value derived elsewhere; show what is
    # in use so the viewer is not misleading.
    if name == "mode":
        return config.effective_mode
    if name == "ui_base_url":
        return config.effective_ui_base_url or None
    value: bool | int | float | str | list[str] | None = getattr(config, name)
    if name in _REDACTED_URL_FIELDS and isinstance(value, str):
        return redact_url_secrets(value)
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
        secret_key_configured=secret_box_configured(),
        config=_config_fields(config),
    )


@router.get("")
async def get_settings(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Return non-secret runtime settings for the admin dashboard."""
    return _current_settings(config)


@router.patch("")
async def update_settings(
    request: UpdateSettingsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewaySettings:
    """Persist and apply runtime setting changes.

    Each provided field is stored as an override (winning over config/env) and
    applied to the running gateway immediately. Operator-gated: these change
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


@router.post("/master-key/rotate")
async def rotate_master_key(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    authenticated_key: Annotated[str | None, Depends(verify_master_key)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> RotateMasterKeyResponse:
    """Regenerate the database-backed master key and invalidate the old one.

    Only the first-run generated master key can be rotated here. When a master
    key is supplied through config or ``OTARI_MASTER_KEY``, the dashboard cannot
    invalidate it; the operator must change that value and restart instead.

    Every dashboard session is revoked with the rotation (a session only proves
    possession of the now-dead key); the caller's own session is re-minted under
    the new key, for the same identity it named, so the tab that performed the
    rotation stays signed in as who it was. A caller that authenticated with a
    header key has no session identity to re-mint for, so it is not handed one:
    it was not signed in to the dashboard to begin with.
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
        # Keep the startup key-change check in step, so a restart after this
        # rotation does not revoke the session re-minted below.
        await record_session_key_marker(db, hashed)
        if session_identity is not None:
            session_token, session_expires_at = await create_dashboard_session(
                db, config.dashboard_session_ttl_hours, user_id=session_identity.id
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
        apply_session_cookie(response, session_token, session_expires_at, secure=request_is_https(request))
    return RotateMasterKeyResponse(master_key=token)
