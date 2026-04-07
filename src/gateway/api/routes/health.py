from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from gateway.api.deps import get_config, get_db
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.version import __version__

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(config: GatewayConfig = Depends(get_config)) -> dict[str, str]:
    """General health check endpoint.

    Returns basic health status. For infrastructure monitoring,
    use /health/readiness or /health/liveness instead.
    """
    payload: dict[str, str] = {"status": "healthy"}
    if config.is_platform_mode:
        payload["mode"] = "platform"
    return payload


@router.get("/liveness")
async def health_liveness() -> str:
    """Liveness probe endpoint.

    Simple check to verify the process is alive and responding.
    Used by Kubernetes/container orchestrators for liveness probes.

    Returns:
        Plain text "I'm alive!" message

    """
    return "I'm alive!"


@router.get("/readiness")
async def health_readiness(config: GatewayConfig = Depends(get_config)) -> dict[str, Any]:
    """Readiness probe endpoint.

    Checks if the gateway is ready to serve requests by validating:
    - Database connectivity
    - Service availability

    Used by Kubernetes/container orchestrators for readiness probes.
    Returns HTTP 503 if any dependency is unavailable.

    Returns:
        dict: Status object with health details

    Raises:
        HTTPException: 503 if service is not ready

    """
    if config.is_platform_mode:
        return {
            "status": "healthy",
            "mode": "platform",
            "version": __version__,
        }

    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            db.execute(text("SELECT 1"))
            db_status = "connected"
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Database connectivity check failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "unavailable",
                "version": __version__,
            },
        ) from e
    return {
        "status": "healthy",
        "database": db_status,
        "version": __version__,
    }
