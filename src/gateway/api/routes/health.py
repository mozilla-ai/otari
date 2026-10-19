from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db
from gateway.log_config import logger
from gateway.version import __version__

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, str]:
    """General health check endpoint.

    Returns basic health status. For infrastructure monitoring,
    use /health/readiness or /health/liveness instead.
    """
    return {"status": "healthy"}


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
async def health_readiness(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
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
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error("Database connectivity check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
