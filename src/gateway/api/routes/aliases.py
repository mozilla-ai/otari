"""Runtime model alias management.

An alias is a display name that resolves to a real ``provider:model`` selector.
``config.yml`` aliases are read-only here (they are validated at startup and
live in a file this process does not own); these routes manage the
``model_aliases`` table, which means the same thing to a request but can change
without a restart.

A stored alias is either global (``user_id`` omitted) or scoped to one user, who
is then the only caller that resolves it. See ``services/alias_service`` for the
precedence between the layers.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelAlias
from gateway.repositories.users_repository import get_active_user
from gateway.services.alias_service import all_alias_names, refresh_alias_cache

router = APIRouter(prefix="/v1/aliases", tags=["aliases"])


class AliasRequest(BaseModel):
    """Request to create or update an alias."""

    name: str = Field(description="Display name callers use as the model, e.g. 'fast-model'.")
    target: str = Field(description="Selector the alias resolves to, as 'provider:model' or 'instance:model'.")
    user_id: str | None = Field(
        default=None,
        description=(
            "User this alias belongs to. Omit for a global alias every caller sees. "
            "A user-scoped alias resolves only for that user and shadows a global one of the same name."
        ),
    )


class AliasResponse(BaseModel):
    """A model alias and where it is defined."""

    name: str
    target: str
    # "config" for a config.yml alias (read-only here) or "stored" for a row in
    # model_aliases. Only stored aliases can be edited or deleted.
    source: str
    # The user this alias is scoped to, or null when it applies to every caller.
    # config.yml aliases are always global.
    user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_model(cls, alias: ModelAlias) -> "AliasResponse":
        return cls(
            name=alias.name,
            target=alias.target,
            source="stored",
            user_id=alias.user_id,
            created_at=alias.created_at.isoformat() if alias.created_at else None,
            updated_at=alias.updated_at.isoformat() if alias.updated_at else None,
        )


def _validate(config: GatewayConfig, name: str, target: str, user_id: str | None) -> None:
    """Apply the startup alias rules to a runtime write, as a 400.

    A configured alias wins over a *global* stored one during resolution, so
    storing a global name that shadows one would be accepted and then never take
    effect. Refusing is the only answer that does not lie about what the gateway
    will do. A user-scoped alias is exempt: it outranks both other layers, so
    shadowing a configured name is a working override rather than dead data, and
    is the reason to scope an alias in the first place.
    """
    if user_id is None and name in config.aliases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already an alias in config.yml, pointing at '{config.aliases[name]}'. "
                "Config aliases take precedence over global stored ones, so this one would never be used. "
                "Rename it, scope it to a user, or edit config.yml."
            ),
        )
    # The chaining check spans every scope, matching the pricing and allow-list
    # checks: an alias pointing at a stored alias is just as broken as one
    # pointing at a configured alias. Another user's names cannot actually be
    # reached from here (resolution is single-pass, and validate_alias inspects
    # the target's prefix, so such a target fails as an unknown provider anyway),
    # so this is for consistency rather than to close a hole.
    alias_names = all_alias_names(config) | {name}
    try:
        config.validate_alias(name, target, alias_names=alias_names)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


async def _require_user(db: AsyncSession, user_id: str) -> None:
    """404 unless ``user_id`` names a live user.

    Unknown ids have to be caught here: the column is a foreign key, so one would
    otherwise surface as an opaque 500 from the commit (or, on SQLite without
    enforcement, as an alias nobody can ever resolve). Soft-deleted users are
    rejected for the same reason every other user-scoped route uses
    ``get_active_user``: they cannot authenticate, so the alias would be dead on
    arrival.
    """
    if await get_active_user(db, user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found")


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_aliases(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> list[AliasResponse]:
    """List every alias in force, from config.yml and from storage.

    Every scope at once, global and user-scoped alike: this is the master-key
    management view, not what any one caller resolves.
    """
    rows = (await db.execute(select(ModelAlias).order_by(ModelAlias.name))).scalars().all()
    # Keyed on (name, scope) rather than name: the same display name can exist
    # globally and per user, and both are real rows to manage.
    merged = {(row.name, row.user_id): AliasResponse.from_model(row) for row in rows}
    # Config last, matching effective_aliases: if a global name somehow exists on
    # both sides, list the one that would actually resolve rather than both.
    merged.update(
        {
            (name, None): AliasResponse(name=name, target=target, source="config")
            for name, target in config.aliases.items()
        }
    )
    return sorted(merged.values(), key=lambda alias: (alias.name, alias.user_id or ""))


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_alias(
    request: AliasRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AliasResponse:
    """Create or update a stored alias, global or scoped to one user."""
    if request.user_id is not None:
        await _require_user(db, request.user_id)
    await refresh_alias_cache(db)
    _validate(config, request.name, request.target, request.user_id)

    # Scope is part of the identity: the upsert must not turn a global alias into
    # a user-scoped one (or vice versa) just because the names match.
    alias = (
        await db.execute(
            select(ModelAlias).where(ModelAlias.name == request.name, ModelAlias.user_id == request.user_id)
        )
    ).scalar_one_or_none()
    if alias:
        alias.target = request.target
    else:
        alias = ModelAlias(name=request.name, target=request.target, user_id=request.user_id)
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
    user_id: Annotated[
        str | None,
        Query(description="Delete the alias scoped to this user. Omit to delete the global alias of that name."),
    ] = None,
) -> None:
    """Delete a stored alias in one scope.

    Scoped by ``user_id`` for the same reason the upsert is: deleting the global
    alias must not take a user's override with it, and deleting an override must
    leave the global one serving everyone else.
    """
    alias = (
        await db.execute(select(ModelAlias).where(ModelAlias.name == name, ModelAlias.user_id == user_id))
    ).scalar_one_or_none()
    if alias is None:
        scope = "global" if user_id is None else f"scoped to user '{user_id}'"
        detail = f"Alias '{name}' ({scope}) not found"
        if user_id is None and name in config.aliases:
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
