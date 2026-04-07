from fastapi import APIRouter, HTTPException, status

_DISABLED_DETAIL = "This endpoint is not available in platform mode. Manage this resource via the platform UI."

router = APIRouter(tags=["platform-mode"])


def _raise_disabled() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)


@router.api_route("/v1/users/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/users", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def users_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/keys/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/keys", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def keys_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/budgets/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/budgets", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def budgets_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/spend/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/spend", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def spend_disabled() -> None:
    _raise_disabled()
