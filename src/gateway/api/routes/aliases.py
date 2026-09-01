"""Runtime model alias management.

An alias is a display name that resolves to a real ``provider:model`` selector.
``config.yml`` aliases are read-only here (they are validated at startup and
live in a file this process does not own); these routes manage the
``model_aliases`` table, which means the same thing to a request but can change
without a restart.

A stored alias belongs to one workspace, and within it is either workspace-wide
(``user_id`` omitted) or scoped to one user, who is then the only caller in that
workspace that resolves it. A ``config.yml`` alias has no workspace and is in
force in all of them. See ``services/alias_service`` for the precedence between
the layers.

Every verb here takes an optional ``workspace_id``; omitting it means the
deployment's default workspace, which is where an operator acting deployment-wide
writes and where every row predating workspace scoping was backfilled. A
single-workspace deployment therefore never has to name one.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, require_deployment_operator
from gateway.api.routes._helpers import resolve_managed_workspace_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelAlias
from gateway.repositories.users_repository import get_active_user
from gateway.services.alias_service import all_alias_names, refresh_alias_cache
from gateway.services.policy_store import all_policy_names

router = APIRouter(
    prefix="/v1/aliases",
    tags=["aliases"],
    dependencies=[Depends(require_deployment_operator)],
)


class AliasRequest(BaseModel):
    """Request to create or update an alias."""

    name: str = Field(description="Display name callers use as the model, e.g. 'fast-model'.")
    target: str = Field(description="Selector the alias resolves to, as 'provider:model' or 'instance:model'.")
    user_id: str | None = Field(
        default=None,
        description=(
            "User this alias belongs to. Omit for an alias every caller in the workspace sees. "
            "A user-scoped alias resolves only for that user and shadows the workspace-wide one "
            "of the same name."
        ),
    )
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Workspace this alias belongs to. Omit for the deployment's default workspace. "
            "The alias resolves only for requests in that workspace, so two workspaces can each "
            "point the same name at a different model."
        ),
    )


class AliasResponse(BaseModel):
    """A model alias and where it is defined."""

    name: str
    target: str
    # "config" for a config.yml alias (read-only here) or "stored" for a row in
    # model_aliases. Only stored aliases can be edited or deleted.
    source: str
    # The user this alias is scoped to, or null when it applies to every caller
    # in its workspace. config.yml aliases are never user-scoped.
    user_id: str | None = None
    # The workspace the stored row lives in. Null for a config.yml alias, which
    # is deployment-wide and in force in every workspace.
    workspace_id: uuid.UUID | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_model(cls, alias: ModelAlias) -> "AliasResponse":
        return cls(
            name=alias.name,
            target=alias.target,
            source="stored",
            user_id=alias.user_id,
            workspace_id=alias.workspace_id,
            created_at=alias.created_at.isoformat() if alias.created_at else None,
            updated_at=alias.updated_at.isoformat() if alias.updated_at else None,
        )


def _validate(config: GatewayConfig, name: str, target: str, user_id: str | None) -> None:
    """Apply the startup alias rules to a runtime write, as a 400.

    A configured alias wins over a *workspace-wide* stored one during resolution,
    so storing a workspace-wide name that shadows one would be accepted and then
    never take effect. Refusing is the only answer that does not lie about what the gateway
    will do. A user-scoped alias is exempt: it outranks both other layers, so
    shadowing a configured name is a working override rather than dead data, and
    is the reason to scope an alias in the first place.
    """
    if user_id is None and name in config.aliases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already an alias in config.yml, pointing at '{config.aliases[name]}'. "
                "Config aliases take precedence over workspace-wide stored ones, so this one would never be used. "
                "Rename it, scope it to a user, or edit config.yml."
            ),
        )
    # The chaining check spans every scope, matching the pricing and allow-list
    # checks: an alias pointing at a stored alias is just as broken as one
    # pointing at a configured alias. Another user's names cannot actually be
    # reached from here (resolution is single-pass, and validate_alias inspects
    # the target's prefix, so such a target fails as an unknown provider anyway),
    # so this is for consistency rather than to close a hole.
    # Symmetry with the policy write path, which already refuses a policy named
    # after an alias. Without this, creating an alias named after an existing
    # policy would silently shadow it: alias resolution runs first, so the policy
    # would stop taking effect and nothing would say so.
    if name in all_policy_names(config):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' is already a routing policy. An alias resolves before a policy, so this would "
                "silently stop that policy taking effect. Rename the alias, or edit the policy instead."
            ),
        )
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


@router.get("")
async def list_aliases(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description=(
            "Only stored entries in this workspace. Config-file entries are always included, "
            "being deployment-wide. Omit to list the stored entries of every workspace."
        )),
    ] = None,
) -> list[AliasResponse]:
    """List every alias in force, from config.yml and from storage.

    Every scope at once, workspace-wide and user-scoped alike: this is the
    master-key management view, not what any one caller resolves.
    """
    statement = select(ModelAlias)
    if workspace_id is not None:
        # Stored rows only. A config.yml entry has no workspace: it is
        # deployment-wide and in force in every one of them, so filtering
        # it out would misreport what actually resolves.
        statement = statement.where(ModelAlias.workspace_id == workspace_id)
    rows = (await db.execute(statement.order_by(ModelAlias.name))).scalars().all()
    # Keyed on (workspace, name, user) rather than name: the same display name can
    # exist in several workspaces, and within one both workspace-wide and per
    # user, and every one of those is a real row to manage.
    merged: dict[tuple[uuid.UUID | None, str, str | None], AliasResponse] = {
        (row.workspace_id, row.name, row.user_id): AliasResponse.from_model(row) for row in rows
    }
    # Config last, matching effective_aliases: a configured name beats the stored
    # workspace-wide row in every workspace, so it is listed once, unscoped,
    # instead of shadowing each workspace's row in place.
    merged.update(
        {
            (None, name, None): AliasResponse(name=name, target=target, source="config")
            for name, target in config.aliases.items()
        }
    )
    return sorted(
        merged.values(),
        key=lambda alias: (alias.name, str(alias.workspace_id or ""), alias.user_id or ""),
    )


@router.post("")
async def set_alias(
    request: AliasRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AliasResponse:
    """Create or update a stored alias in one workspace, optionally for one user."""
    if request.user_id is not None:
        await _require_user(db, request.user_id)
    workspace_id = await resolve_managed_workspace_id(db, request.workspace_id)
    await refresh_alias_cache(db)
    _validate(config, request.name, request.target, request.user_id)

    # Both scopes are part of the identity: the upsert must not turn one
    # workspace's alias into another's, nor a workspace-wide alias into a
    # user-scoped one (or vice versa), just because the names match.
    alias = (
        await db.execute(
            select(ModelAlias).where(
                ModelAlias.workspace_id == workspace_id,
                ModelAlias.name == request.name,
                ModelAlias.user_id == request.user_id,
            )
        )
    ).scalar_one_or_none()
    if alias:
        alias.target = request.target
    else:
        alias = ModelAlias(
            name=request.name,
            target=request.target,
            user_id=request.user_id,
            workspace_id=workspace_id,
        )
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


@router.delete("/{name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alias(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user_id: Annotated[
        str | None,
        Query(
            description=(
                "Delete the alias scoped to this user. Omit to delete the workspace-wide alias "
                "of that name."
            )
        ),
    ] = None,
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Delete the alias in this workspace. Omit for the deployment's default workspace."),
    ] = None,
) -> None:
    """Delete a stored alias in one scope.

    Scoped by ``workspace_id`` and ``user_id`` for the same reason the upsert is:
    deleting one workspace's alias must not touch another's, deleting the
    workspace-wide alias must not take a user's override with it, and deleting an
    override must leave the workspace-wide one serving everyone else.
    """
    target_workspace_id = await resolve_managed_workspace_id(db, workspace_id)
    alias = (
        await db.execute(
            select(ModelAlias).where(
                ModelAlias.workspace_id == target_workspace_id,
                ModelAlias.name == name,
                ModelAlias.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if alias is None:
        scope = "workspace-wide" if user_id is None else f"scoped to user '{user_id}'"
        detail = f"Alias '{name}' ({scope}) not found in workspace '{target_workspace_id}'"
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
