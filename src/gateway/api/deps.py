import secrets
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import hash_key
from gateway.core.config import API_KEY_HEADER, LEGACY_API_KEY_HEADERS, GatewayConfig
from gateway.core.database import create_session, get_db
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.models.entities import APIKey
from gateway.services.file_store import FileStore
from gateway.services.log_writer import LogWriter
from gateway.services.master_key_service import hash_master_key

# Legacy module-level fallback. Config now lives on ``app.state.config`` (set in
# ``create_app``); ``get_config`` reads from the request's app state and only
# falls back to this shim for callers that set it directly (see ``set_config``).
_config: GatewayConfig | None = None
_LAST_USED_UPDATE_INTERVAL_SECONDS = 300


def _as_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as a timezone-aware datetime in UTC.

    SQLite stores ``DateTime(timezone=True)`` columns as naive strings and
    returns them naive on read. PostgreSQL returns them as aware. Normalising
    here keeps the subtraction/comparison call sites identical across both
    backends — a naive value is *assumed* to be UTC, which matches how the
    gateway writes them (always ``datetime.now(UTC)``).
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def set_config(config: GatewayConfig) -> None:
    """Set the legacy module-level config fallback.

    Compatibility shim. Config is stored on ``app.state.config`` by
    ``create_app``; this only updates the module-level fallback that
    ``get_config`` consults when no request-scoped app state is available.
    """
    global _config  # noqa: PLW0603
    _config = config


def get_config(request: Request) -> GatewayConfig:
    """Return the config for the current app from ``request.app.state``.

    Every shared resource (rate limiter, log writer, file store) lives on
    ``app.state``; config does too, so two apps in one process no longer share
    a single instance. Falls back to the legacy module-level shim only when the
    app state has no config attached.
    """
    config: GatewayConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        config = _config
    if config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return config


def reset_config() -> None:
    """Reset the legacy module-level config fallback. Intended for testing only."""
    global _config  # noqa: PLW0603
    _config = None


def _extract_bearer_token(request: Request, config: GatewayConfig) -> str:
    """Extract and validate Bearer token from request header.

    Checks the canonical Otari-Key header first, then the legacy
    AnyLLM-Key / X-AnyLLM-Key aliases (back-compat), then falls back
    to the standard Authorization header.
    """
    auth_header = request.headers.get(API_KEY_HEADER)
    if not auth_header:
        for legacy in LEGACY_API_KEY_HEADERS:
            auth_header = request.headers.get(legacy)
            if auth_header:
                break
    if not auth_header:
        auth_header = request.headers.get("Authorization")

    if auth_header:
        if not auth_header.startswith("Bearer "):
            record_auth_failure("invalid_format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid header format. Expected 'Bearer <token>'",
            )
        return auth_header[7:]

    record_auth_failure("missing_credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing {API_KEY_HEADER} or Authorization header",
    )


async def _verify_and_update_api_key(db: AsyncSession, token: str) -> APIKey:
    """Verify API key token and update last_used_at."""
    try:
        key_hash = hash_key(token)
    except ValueError as e:
        record_auth_failure("invalid_format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key format: {e}",
        ) from e

    try:
        result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    except SQLAlchemyError as e:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from e
    api_key = result.scalar_one_or_none()

    if not api_key:
        record_auth_failure("invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not api_key.is_active:
        record_auth_failure("inactive_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
        )

    expires_at = _as_utc(api_key.expires_at)
    if expires_at is not None and expires_at < datetime.now(UTC):
        record_auth_failure("expired_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    now = datetime.now(UTC)
    last_used_at = _as_utc(api_key.last_used_at)
    should_update_last_used = (
        last_used_at is None or (now - last_used_at).total_seconds() >= _LAST_USED_UPDATE_INTERVAL_SECONDS
    )

    if should_update_last_used:
        await _bump_last_used_at(api_key.id, now)

    return api_key


async def _bump_last_used_at(api_key_id: str, now: datetime) -> None:
    """Record an API key's last use on a short-lived, separate session.

    The bump runs outside the request's transaction so it never commits the
    caller's session or leaves it in a dirty state, and a failure is logged
    rather than swallowed. It is best-effort: throttled by
    ``_LAST_USED_UPDATE_INTERVAL_SECONDS`` and never fails the request.
    """
    try:
        async with create_session() as session:
            await session.execute(update(APIKey).where(APIKey.id == api_key_id).values(last_used_at=now))
            await session.commit()
    except SQLAlchemyError:
        logger.warning("Failed to update last_used_at for API key %s", api_key_id, exc_info=True)


def _is_valid_master_key(token: str, config: GatewayConfig) -> bool:
    """Check if token matches the master key (operator-set plaintext or generated hash)."""
    if config.master_key is not None and secrets.compare_digest(token, config.master_key):
        return True
    stored_hash = config._master_key_hash
    if stored_hash is not None:
        return secrets.compare_digest(hash_master_key(token), stored_hash)
    return False


async def verify_api_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> APIKey:
    """Verify API key from Otari-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)
    return await _verify_and_update_api_key(db, token)


async def verify_master_key(
    request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Verify master key from Otari-Key header.

    Args:
        request: FastAPI request object
        config: Gateway configuration

    Raises:
        HTTPException: If master key is not configured or invalid

    """
    if config.master_key is None and config._master_key_hash is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Master key not configured. Set OTARI_MASTER_KEY environment variable.",
        )

    token = _extract_bearer_token(request, config)

    if not _is_valid_master_key(token, config):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid master key",
        )


async def verify_api_key_or_master_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> tuple[APIKey | None, bool]:
    """Verify either API key or master key from Otari-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        Tuple of (APIKey object or None, is_master_key boolean)

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)

    if _is_valid_master_key(token, config):
        return None, True

    api_key = await _verify_and_update_api_key(db, token)
    return api_key, False


async def get_db_if_needed(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AsyncGenerator[AsyncSession | None, None]:
    """Get a database session in standalone mode, otherwise return None."""
    if config.is_hybrid_mode:
        yield None
        return

    async for db in get_db():
        yield db


def get_log_writer(request: Request) -> LogWriter:
    writer: LogWriter = request.app.state.log_writer
    return writer


def get_file_store(request: Request) -> FileStore:
    """Return the configured blob store for uploaded files (standalone mode)."""
    store: FileStore = request.app.state.file_store
    return store


__all__ = [
    "get_config",
    "get_db",
    "reset_config",
    "set_config",
    "get_db_if_needed",
    "get_file_store",
    "get_log_writer",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_master_key",
]
