"""Built-in tool & guardrail configuration for the admin dashboard.

The service-endpoint and web-search fields that ``/v1/settings`` keeps
display-only (the ``*_url`` fields and the web-search knobs are excluded there on
SSRF grounds) are made editable here, on their own page, with structural URL
validation and per-service reachability tests. Standalone-only and master-key
gated, mirroring the other management routers.

* ``GET /v1/tool-settings`` returns each field's effective value (the value a
  request would actually use), grouped by service, with URL passwords masked.
* ``PATCH /v1/tool-settings`` persists overrides (an explicit ``null`` clears a
  field back to the configured env/YAML default; an omitted field is unchanged)
  and applies them to the running worker.
* ``POST /v1/tool-settings/{service}/test`` structurally validates a (typically
  unsaved) URL and probes it for reachability, returning ``{ok, reason}``.
"""

from typing import Annotated, Literal, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.api.routes.settings import _redact_url_secrets
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.runtime_settings_service import SettingValue
from gateway.services.tool_settings_service import (
    SERVICE_URL_FIELD,
    TOOL_SETTABLE_KEYS,
    apply_override,
    effective_values,
    field_service,
    field_type,
    stage_override,
    validate_url,
)

router = APIRouter(prefix="/v1/tool-settings", tags=["tool-settings"])

_URL_FIELDS = frozenset(SERVICE_URL_FIELD.values())

# Reachability probe timeout. Short so a mistyped or dead host fails fast in the
# dashboard rather than hanging the operator's Test click.
_PROBE_TIMEOUT_S = 5.0


class ToolSettingField(BaseModel):
    """One editable tool/guardrail field surfaced to the dashboard."""

    key: str
    service: Literal["web_search", "sandbox", "guardrails"]
    type: Literal["url", "str", "int", "bool"]
    # The tool fields are only url/str/int/bool, so no float ever appears here;
    # keeping float out of the type narrows the OpenAPI contract accordingly.
    value: bool | int | str | None
    description: str | None = None


class ToolSettingsResponse(BaseModel):
    """The effective value of every editable tool/guardrail field."""

    fields: list[ToolSettingField]


class UpdateToolSettingsRequest(BaseModel):
    """Change one or more tool settings. Omitted fields are left unchanged; an
    explicit ``null`` clears a field back to the configured env/YAML default."""

    # A valid partial update, so the generated OpenAPI/Postman example is runnable
    # (omitted fields stay unchanged; the URL is well-formed and max_results >= 1).
    model_config = ConfigDict(
        json_schema_extra={"example": {"web_search_url": "http://searxng:8080", "web_search_max_results": 5}}
    )

    web_search_url: str | None = None
    web_search_engines: str | None = None
    web_search_max_results: int | None = Field(default=None, ge=1)
    web_search_extract: bool | None = None
    web_search_purpose_hint: str | None = None
    sandbox_url: str | None = None
    sandbox_purpose_hint: str | None = None
    guardrails_url: str | None = None


class TestServiceRequest(BaseModel):
    """A (typically unsaved) URL to probe for reachability."""

    url: str


class TestServiceResponse(BaseModel):
    ok: bool
    reason: str


def _display_value(config: GatewayConfig, key: str) -> bool | int | str | None:
    value = effective_values(config)[key]
    if key in _URL_FIELDS and isinstance(value, str):
        return _redact_url_secrets(value)
    # No tool field is float-typed, so the SettingValue here is bool/int/str/None.
    return cast("bool | int | str | None", value)


def _current_fields(config: GatewayConfig) -> ToolSettingsResponse:
    fields = [
        ToolSettingField(
            key=key,
            service=field_service(key),  # type: ignore[arg-type]
            type=field_type(key),  # type: ignore[arg-type]
            value=_display_value(config, key),
            description=GatewayConfig.model_fields[key].description,
        )
        for key in TOOL_SETTABLE_KEYS
    ]
    return ToolSettingsResponse(fields=fields)


@router.get("", dependencies=[Depends(verify_master_key)])
async def get_tool_settings(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ToolSettingsResponse:
    """Return the effective tool/guardrail settings for the dashboard."""
    return _current_fields(config)


@router.patch("", dependencies=[Depends(verify_master_key)])
async def update_tool_settings(
    request: UpdateToolSettingsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ToolSettingsResponse:
    """Persist and apply tool/guardrail setting changes.

    Uses ``model_fields_set`` so an explicit ``null`` clears a field while an
    omitted field is left unchanged. Master-key gated and standalone-only.
    """
    updates: dict[str, SettingValue] = {
        key: getattr(request, key) for key in request.model_fields_set if key in TOOL_SETTABLE_KEYS
    }

    if updates:
        try:
            normalized = {key: await stage_override(db, key, value) for key, value in updates.items()}
            await db.commit()
        except ValueError as exc:
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from None
        except SQLAlchemyError:
            await db.rollback()
            # Log for the operator debugging from logs; the client sees only a
            # generic message so no internals leak. Keys, not values, are logged.
            logger.exception("Failed to persist tool settings: %s", sorted(updates))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error",
            ) from None
        # Apply only after the commit succeeds, so a failed write never leaves this
        # worker running against an unpersisted value.
        for key, value in normalized.items():
            apply_override(config, key, value)
            logger.info("Tool setting changed via dashboard: %s", key)

    return _current_fields(config)


@router.post("/{service}/test", dependencies=[Depends(verify_master_key)])
async def test_service(
    service: str,
    request: TestServiceRequest,
) -> TestServiceResponse:
    """Structurally validate a URL and probe it for reachability.

    Tests the URL in the request body (typically unsaved), so an operator can
    verify before saving. The probe is a plain HTTP GET with a short timeout: any
    HTTP response means the host is reachable; a connection/timeout/DNS error means
    it is not. The operator is trusted (master key), so no SSRF deny-list applies;
    only the structural check (http/https + host) runs first.
    """
    if service not in SERVICE_URL_FIELD:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown service: {service!r}",
        )
    try:
        url = validate_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from None

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, follow_redirects=False) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return TestServiceResponse(ok=False, reason=f"unreachable: {exc.__class__.__name__}: {exc}")
    return TestServiceResponse(ok=True, reason=f"reachable (HTTP {resp.status_code})")
