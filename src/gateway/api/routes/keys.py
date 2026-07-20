import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.auth.models import generate_api_key, hash_key, key_prefix
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, User
from gateway.services.model_access import validate_allowed_models

router = APIRouter(prefix="/v1/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    key_name: str | None = Field(default=None, description="Optional name for the key")
    user_id: str | None = Field(default=None, description="Optional user ID to associate with this key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Model allow-list: null = any model, [] = deny all, or canonical "
        "instance:model entries (with instance:* / instance:prefix* wildcards).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class CreateKeyResponse(BaseModel):
    """Response model for creating a new API key."""

    id: str
    key: str
    # Leading characters of the key, echoed so the client can key its show-once
    # reveal to the same fingerprint the list will display afterward.
    key_prefix: str | None
    key_name: str | None
    user_id: str | None
    created_at: str
    expires_at: str | None
    is_active: bool
    allowed_models: list[str] | None
    metadata: dict[str, Any]


class KeyInfo(BaseModel):
    """Response model for key information."""

    id: str
    # Display-only fingerprint (leading characters of the plaintext key). Null for
    # keys minted before the prefix was recorded; the full key is never returned.
    key_prefix: str | None
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    allowed_models: list[str] | None
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, key: APIKey) -> "KeyInfo":
        return cls(
            id=str(key.id),
            key_prefix=str(key.key_prefix) if key.key_prefix else None,
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat(),
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            allowed_models=list(key.allowed_models) if key.allowed_models is not None else None,
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )


class UpdateKeyRequest(BaseModel):
    """Request model for updating a key."""

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    # Tri-state via model_fields_set: absent = unchanged, null = clear to
    # unrestricted, [] = deny all, list = restrict. A plain default cannot tell
    # "absent" from "explicit null", so the handler checks model_fields_set.
    allowed_models: list[str] | None = None
    metadata: dict[str, Any] | None = None


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_key(
    request: CreateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CreateKeyResponse:
    """Create a new API key.

    Requires master key authentication.

    If user_id is provided, the key will be associated with that user (creates user if it doesn't exist).
    If user_id is not provided, a new user will be created automatically and the key will be associated with it.
    """
    try:
        allowed_models = validate_allowed_models(config, request.allowed_models)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    api_key = generate_api_key()
    key_hash = hash_key(api_key)
    key_id = uuid.uuid4()

    if request.user_id:
        result = await db.execute(select(User).where(User.user_id == request.user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                user_id=request.user_id,
                alias=f"User {request.user_id}",
            )
            db.add(user)
        elif user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{request.user_id}' has been deleted. Recreate via POST /v1/users first.",
            )
        user_id = request.user_id
    else:
        user_id = f"apikey-{key_id}"
        user = User(
            user_id=user_id,
            alias=f"Virtual user for API key: {request.key_name or 'unnamed'}",
        )
        db.add(user)

    db_key = APIKey(
        id=str(key_id),
        key_hash=key_hash,
        key_prefix=key_prefix(api_key),
        key_name=request.key_name,
        user_id=user_id,
        expires_at=request.expires_at,
        allowed_models=allowed_models,
        metadata_=request.metadata,
    )

    db.add(db_key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(db_key)

    key_info = KeyInfo.from_model(db_key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=api_key,
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[KeyInfo]:
    """List all API keys.

    Requires master key authentication.
    """
    result = await db.execute(select(APIKey).offset(skip).limit(limit))
    keys = result.scalars().all()

    return [KeyInfo.from_model(key) for key in keys]


@router.get("/{key_id}", dependencies=[Depends(verify_master_key)])
async def get_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeyInfo:
    """Get details of a specific API key.

    Requires master key authentication.
    """
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    return KeyInfo.from_model(key)


@router.patch("/{key_id}", dependencies=[Depends(verify_master_key)])
async def update_key(
    key_id: str,
    request: UpdateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> KeyInfo:
    """Update an API key.

    Requires master key authentication.
    """
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    if request.key_name is not None:
        key.key_name = request.key_name
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.expires_at is not None:
        key.expires_at = request.expires_at
    # Tri-state: only touch the allow-list when the field was supplied. A supplied
    # null clears to unrestricted; [] denies all; a list restricts.
    if "allowed_models" in request.model_fields_set:
        try:
            key.allowed_models = validate_allowed_models(config, request.allowed_models)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if request.metadata is not None:
        key.metadata_ = request.metadata

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(key)

    return KeyInfo.from_model(key)


@router.post("/{key_id}/rotate", dependencies=[Depends(verify_master_key)])
async def rotate_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CreateKeyResponse:
    """Rotate an API key's secret in place.

    Requires master key authentication.

    Generates a new secret for the same key row (id, user, name, expiry, and
    metadata are preserved) and returns the new raw key once, using the same
    response shape as key creation. The previous secret stops authenticating
    immediately; there is no grace window.
    """
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    new_api_key = generate_api_key()
    key.key_hash = hash_key(new_api_key)
    key.key_prefix = key_prefix(new_api_key)
    key.last_used_at = None

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(key)

    key_info = KeyInfo.from_model(key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=new_api_key,
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete (revoke) an API key.

    Requires master key authentication.
    """
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    await db.delete(key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
