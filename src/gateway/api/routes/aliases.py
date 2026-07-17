"""Runtime model alias management.

An alias is a display name that resolves to a real ``provider:model`` selector.
``config.yml`` aliases are read-only here (they are validated at startup and
live in a file this process does not own); these routes manage the
``model_aliases`` table, which means the same thing to a request but can change
without a restart.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelAlias
from gateway.services.alias_service import refresh_alias_cache

router = APIRouter(prefix="/v1/aliases", tags=["aliases"])


class AliasRequest(BaseModel):
    """Request to create or update an alias."""

    name: str = Field(description="Display name callers use as the model, e.g. 'fast-model'.")
    target: str = Field(description="Selector the alias resolves to, as 'provider:model' or 'instance:model'.")


class AliasResponse(BaseModel):
    """A model alias and where it is defined."""

    name: str
    target: str
    # "config" for a config.yml alias (read-only here) or "stored" for a row in
    # model_aliases. Only stored aliases can be edited or deleted.
    source: str
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_model(cls, alias: ModelAlias) -> "AliasResponse":
        return cls(
            name=alias.name,
            target=alias.target,
            source="stored",
            created_at=alias.created_at.isoformat() if alias.created_at else None,
            updated_at=alias.updated_at.isoformat() if alias.updated_at else None,
        )


def _validate(config: GatewayConfig, name: str, target: str, stored: dict[str, str]) -> None:
    """Apply the startup alias rules to a runtime write, as a 400.

    A configured alias wins over a stored one during resolution, so storing a
    name that shadows one would be accepted and then never take effect. Refusing
    is the only answer that does not lie about what the gateway will do.
    """
    if name in config.aliases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already an alias in config.yml, pointing at '{config.aliases[name]}'. "
                "Config aliases take precedence, so this one would never be used. Rename it, or edit config.yml."
            ),
        )
    # The chaining check has to see both sides: an alias pointing at a stored
    # alias is just as broken as one pointing at a configured alias.
    alias_names = set(stored) | set(config.aliases) | {name}
    try:
        config.validate_alias(name, target, alias_names=alias_names)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_aliases(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> list[AliasResponse]:
    """List every alias in force, from config.yml and from storage."""
    rows = (await db.execute(select(ModelAlias).order_by(ModelAlias.name))).scalars().all()
    merged = {row.name: AliasResponse.from_model(row) for row in rows}
    # Config last, matching effective_aliases: if a name somehow exists on both
    # sides, list the one that would actually resolve rather than both.
    merged.update(
        {name: AliasResponse(name=name, target=target, source="config") for name, target in config.aliases.items()}
    )
    return sorted(merged.values(), key=lambda alias: alias.name)


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_alias(
    request: AliasRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AliasResponse:
    """Create or update a stored alias."""
    stored = await refresh_alias_cache(db)
    _validate(config, request.name, request.target, stored)

    alias = (await db.execute(select(ModelAlias).where(ModelAlias.name == request.name))).scalar_one_or_none()
    if alias:
        alias.target = request.target
    else:
        alias = ModelAlias(name=request.name, target=request.target)
        db.add(alias)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(alias)
    # The write is committed; a cache-refresh failure must not turn it into a 500.
    # This worker serves the new alias on its next background refresh, others
    # within the TTL.
    try:
        await refresh_alias_cache(db)
    except SQLAlchemyError:
        logger.warning("Alias cache refresh failed after storing '%s'; converges within TTL", alias.name)
    return AliasResponse.from_model(alias)


@router.delete("/{name:path}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_alias(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Delete a stored alias."""
    alias = (await db.execute(select(ModelAlias).where(ModelAlias.name == name))).scalar_one_or_none()
    if alias is None:
        detail = f"Alias '{name}' not found"
        if name in config.aliases:
            detail = f"Alias '{name}' is defined in config.yml and cannot be deleted through the API."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    await db.delete(alias)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    # The delete is committed; a cache-refresh failure must not turn it into a 500.
    try:
        await refresh_alias_cache(db)
    except SQLAlchemyError:
        logger.warning("Alias cache refresh failed after deleting '%s'; converges within TTL", name)
