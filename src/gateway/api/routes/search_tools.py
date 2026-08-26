"""Runtime search-tool management for the dashboard (``/v1/search-tools``).

``POST /v1/search`` dispatches against the ``search_tools`` map. That map used to
come only from a config file, so a deployment configured entirely through the
dashboard and environment variables could not use the endpoint at all (issue
#601). These endpoints are the missing route in, and they are deliberately the
same shape as ``/v1/provider-credentials``: rows in ``search_tool_credentials``,
the API key encrypted at rest and never returned, merged over the config-file
tools with the stored row winning on a name collision.

Config-file tools stay honored and stay read-only here; they are reported by the
list endpoint so the dashboard can show every tool a request could name, not just
the editable ones.

Master-key gated and standalone-only (the router is not mounted in hybrid). URL
validation is structural, matching ``/v1/tool-settings`` rather than the provider
SSRF gate: the backend this most often points at is a SearXNG sidecar on a
private address, which a deny-private gate would refuse.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, require_deployment_operator
from gateway.core.config import (
    SEARCH_PROVIDERS,
    SEARCH_PROVIDERS_REQUIRING_API_BASE,
    SEARCH_PROVIDERS_REQUIRING_API_KEY,
    GatewayConfig,
    validate_search_tool_entry,
)
from gateway.log_config import logger
from gateway.models.entities import SearchToolCredential
from gateway.services.search_backend import default_api_base
from gateway.services.search_tool_store_service import (
    UNSET,
    config_file_search_tools,
    delete_search_tool,
    get_search_tool,
    get_search_tool_for_update,
    list_search_tools,
    reencrypt_search_tools,
    refresh_search_tool_cache,
    save_search_tool,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
)
from gateway.services.tool_settings_service import validate_url

router = APIRouter(prefix="/v1/search-tools", tags=["search-tools"])


class SearchProviderSchema(BaseModel):
    """One search provider this build can dispatch to, for the add-tool picker."""

    id: str = Field(description="Value to send as 'provider'.")
    requires_api_key: bool = Field(description="True when a tool on this provider must carry an API key.")
    requires_api_base: bool = Field(
        description="True when this provider has no endpoint of its own, so the tool must say where the backend is."
    )
    default_api_base: str | None = Field(
        default=None,
        description=(
            "The endpoint a tool on this provider uses when it declares no api_base. "
            "Null means nothing supplies one, so an api_base is required."
        ),
    )


class StoredSearchToolSchema(BaseModel):
    """A runtime-stored search tool. The API key is never returned, only ``last4``."""

    name: str
    provider: str
    api_base: str | None = None
    last4: str | None = None
    timeout: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    # False when the stored key cannot be decrypted with the current
    # OTARI_SECRET_KEY. Such a tool is skipped at runtime, so the dashboard flags
    # it for the operator to fix.
    decryptable: bool = True
    shadows_config: bool = Field(
        default=False,
        description="True when a config-file search tool of the same name exists; the stored one is in effect.",
    )

    @classmethod
    def from_model(
        cls,
        row: SearchToolCredential,
        *,
        decryptable: bool = True,
        shadows_config: bool = False,
    ) -> "StoredSearchToolSchema":
        return cls(**row.to_public_dict(), decryptable=decryptable, shadows_config=shadows_config)


class ConfigSearchToolSchema(BaseModel):
    """A search tool declared in the config file. Read-only: it cannot be edited here."""

    name: str
    provider: str
    api_base: str | None = None
    has_api_key: bool = Field(description="Whether the config entry carries an API key. The key itself is not shown.")
    shadowed: bool = Field(
        default=False,
        description="True when a stored search tool of the same name overrides this entry.",
    )


class SearchToolsResponse(BaseModel):
    """Every search tool ``POST /v1/search`` can name, by where it came from."""

    stored: list[StoredSearchToolSchema]
    config: list[ConfigSearchToolSchema]


class CreateSearchToolRequest(BaseModel):
    """Create a stored search tool. ``api_key`` is write-only and requires OTARI_SECRET_KEY."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"name": "local", "provider": "searxng", "api_base": "http://searxng:8080"}}
    )

    name: str = Field(min_length=1, description="Name callers pass as 'search_tool_name' or in /v1/search/{tool}.")
    provider: str = Field(description=f"Search provider, one of: {', '.join(SEARCH_PROVIDERS)}.")
    api_base: str | None = Field(
        default=None,
        description="Backend endpoint. Omit to inherit the provider's default (searxng inherits web_search_url).",
    )
    api_key: str | None = Field(default=None, description="Provider API key. Stored encrypted; never returned.")
    timeout: float | None = Field(default=None, gt=0, description="Per-request timeout in seconds.")
    options: dict[str, Any] | None = Field(
        default=None,
        description="Provider-native request fields used as defaults (e.g. exa's 'type', searxng's 'engines').",
    )


class UpdateSearchToolRequest(BaseModel):
    """Update a stored search tool. Omitted fields are unchanged; ``api_key`` rotates in place."""

    provider: str | None = None
    api_base: str | None = None
    api_key: str | None = Field(default=None, description="New API key. Omit to keep the existing one. Never returned.")
    timeout: float | None = Field(default=None, gt=0)
    options: dict[str, Any] | None = None
    expected_updated_at: str | None = Field(
        default=None,
        description="Optimistic concurrency: if set, the update 412s unless it matches the stored updated_at.",
    )


class ReencryptSearchToolsResponse(BaseModel):
    """Result of re-encrypting stored search-tool keys with the primary secret key."""

    reencrypted: int = Field(description="Number of stored search-tool keys re-encrypted.")
    unreadable: int = Field(description="Number of encrypted keys left untouched because they could not be decrypted.")


def _is_decryptable(row: SearchToolCredential) -> bool:
    """Whether the row's stored key can be read with the current OTARI_SECRET_KEY."""
    if not row.encrypted_api_key:
        return True
    try:
        decrypt_secret(row.encrypted_api_key)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return False
    return True


def _validate_entry(name: str, entry: dict[str, Any]) -> None:
    """Hold a dashboard-written tool to the rules the config file is held to, as a 422.

    The same :func:`validate_search_tool_entry` startup validation runs on, so a
    tool saved here can never be one that would refuse to boot from a config file.
    """
    try:
        validate_search_tool_entry(name, entry)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from None
    api_base = entry.get("api_base")
    if api_base:
        try:
            validate_url(str(api_base))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from None


async def _commit(db: AsyncSession, *, conflict_detail: str | None = None) -> None:
    try:
        await db.commit()
    except IntegrityError:
        # A concurrent create can slip past the pre-check and collide on the
        # primary key here; surface that as the intended 409, not a 500.
        await db.rollback()
        if conflict_detail is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict_detail) from None
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None


async def _apply_write(db: AsyncSession, config: GatewayConfig, name: str) -> None:
    """Make a committed search-tool change take effect on this worker.

    The write is already committed, so a refresh failure is logged, not surfaced
    as a 500; other workers converge within the TTL.
    """
    try:
        await refresh_search_tool_cache(db, config)
    except SQLAlchemyError:
        logger.warning("Search tool overlay refresh failed after writing '%s'; converges within TTL", name)


@router.get("/providers", dependencies=[Depends(require_deployment_operator)])
async def list_search_providers(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> list[SearchProviderSchema]:
    """List the search providers this build can dispatch to, for the add-tool form.

    Reports per provider whether an API key is required and what endpoint a tool
    inherits when it declares none, so the form can ask for exactly what the
    chosen provider needs instead of taking a free-text provider name.
    """
    return [
        SearchProviderSchema(
            id=provider,
            requires_api_key=provider in SEARCH_PROVIDERS_REQUIRING_API_KEY,
            requires_api_base=provider in SEARCH_PROVIDERS_REQUIRING_API_BASE,
            default_api_base=default_api_base(config, provider),
        )
        for provider in SEARCH_PROVIDERS
    ]


@router.get("", dependencies=[Depends(require_deployment_operator)])
async def list_all_search_tools(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SearchToolsResponse:
    """List every search tool ``POST /v1/search`` can name.

    ``stored`` are the editable rows written through this API; ``config`` are the
    config-file entries, which are still honored and are reported so the operator
    can see the whole set. Keys are never returned, only ``last4``.
    """
    from_config = config_file_search_tools(config)
    stored = await list_search_tools(db)
    stored_names = {row.name for row in stored}
    return SearchToolsResponse(
        stored=[
            StoredSearchToolSchema.from_model(
                row,
                decryptable=_is_decryptable(row),
                shadows_config=row.name in from_config,
            )
            for row in stored
        ],
        config=[
            ConfigSearchToolSchema(
                name=name,
                provider=str(entry.get("provider") or name),
                api_base=entry.get("api_base"),
                has_api_key=bool(entry.get("api_key")),
                shadowed=name in stored_names,
            )
            for name, entry in sorted(from_config.items())
        ],
    )


@router.post("/reencrypt", dependencies=[Depends(require_deployment_operator)])
async def reencrypt_stored_search_tool_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ReencryptSearchToolsResponse:
    """Re-encrypt stored search-tool keys with the primary OTARI_SECRET_KEY.

    The search-tool half of the ``OTARI_SECRET_KEY`` rotation procedure; run it
    alongside ``POST /v1/provider-credentials/reencrypt``. Rows that cannot be
    decrypted are left untouched and must be recovered by replacing the affected
    tool's key.
    """
    try:
        reencrypted, unreadable = await reencrypt_search_tools(db)
        await db.commit()
    except SecretBoxUnavailableError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from None
    try:
        await refresh_search_tool_cache(db, config)
    except SQLAlchemyError:
        logger.warning("Search tool overlay refresh failed after re-encrypting keys; converges within TTL")
    return ReencryptSearchToolsResponse(reencrypted=reencrypted, unreadable=unreadable)


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_deployment_operator)])
async def create_search_tool(
    request: CreateSearchToolRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> StoredSearchToolSchema:
    """Add a search tool at runtime. Storing an API key requires OTARI_SECRET_KEY."""
    name = request.name.strip()
    _validate_entry(
        name,
        {
            "provider": request.provider,
            "api_base": request.api_base,
            "api_key": request.api_key,
            "timeout": request.timeout,
            "options": request.options,
        },
    )
    conflict = f"A stored search tool '{name}' already exists; use PATCH to update it."
    if await get_search_tool(db, name) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict)
    try:
        row = await save_search_tool(
            db,
            name=name,
            provider=request.provider,
            api_base=request.api_base,
            api_key=request.api_key,
            timeout=request.timeout,
            options=request.options,
        )
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    await _commit(db, conflict_detail=conflict)
    shadows_config = name in config_file_search_tools(config)
    if shadows_config:
        logger.warning(
            "Stored search tool '%s' shadows the config.yml search tool of the same name; "
            "the stored entry now wins.",
            name,
        )
    await _apply_write(db, config, name)
    await db.refresh(row)
    return StoredSearchToolSchema.from_model(row, shadows_config=shadows_config)


@router.patch("/{name}", dependencies=[Depends(require_deployment_operator)])
async def update_search_tool(
    name: str,
    request: UpdateSearchToolRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> StoredSearchToolSchema:
    """Update a stored search tool. Omitted fields are left as-is; an explicit ``null`` clears them.

    ``api_key`` follows the same rule: omit it to keep the stored key, send a new
    one to rotate, or send ``null`` to clear it (a keyless SearXNG backend). The
    row is locked ``FOR UPDATE`` so the ``expected_updated_at`` check and the
    write it guards are atomic. The tool as it will be after the update is
    validated, so a change that would leave it unusable (clearing the key of a
    provider that needs one) is refused rather than stored.
    """
    existing = await get_search_tool_for_update(db, name)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No stored search tool '{name}'.")
    if request.expected_updated_at is not None:
        current = existing.updated_at.isoformat() if existing.updated_at else None
        if current != request.expected_updated_at:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="This search tool was modified since you loaded it; reload and retry.",
            )

    # Distinguish "field omitted" (keep) from "field set to null" (clear), then
    # validate the resulting tool rather than the patch in isolation.
    sent = request.model_fields_set
    merged: dict[str, Any] = {
        "provider": request.provider if "provider" in sent and request.provider else existing.provider,
        "api_base": request.api_base if "api_base" in sent else existing.api_base,
        "timeout": request.timeout if "timeout" in sent else existing.timeout_seconds,
        "options": request.options if "options" in sent else existing.options,
        # Only presence matters to the validator, and the stored key is never
        # decrypted here just to re-validate it.
        "api_key": request.api_key if "api_key" in sent else existing.encrypted_api_key,
    }
    _validate_entry(name, merged)

    try:
        row = await save_search_tool(
            db,
            name=name,
            # An explicit null provider is meaningless (the column is non-null),
            # so it is treated as "unchanged" rather than rejected.
            provider=request.provider if "provider" in sent and request.provider else UNSET,
            api_base=request.api_base if "api_base" in sent else UNSET,
            api_key=request.api_key if "api_key" in sent else UNSET,
            timeout=request.timeout if "timeout" in sent else UNSET,
            options=request.options if "options" in sent else UNSET,
        )
    except SecretBoxUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    await _commit(db)
    await _apply_write(db, config, name)
    await db.refresh(row)
    return StoredSearchToolSchema.from_model(row, shadows_config=name in config_file_search_tools(config))


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_deployment_operator)])
async def delete_stored_search_tool(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Delete a stored search tool. A config-file search tool cannot be deleted here."""
    if not await delete_search_tool(db, name):
        detail = f"No stored search tool '{name}'."
        if name in config_file_search_tools(config):
            detail = f"Search tool '{name}' is defined in the config file and cannot be deleted through the API."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    await _commit(db)
    await _apply_write(db, config, name)
