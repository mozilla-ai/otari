from fastapi import APIRouter, HTTPException, status

_DISABLED_DETAIL = "This endpoint is not available in hybrid mode. Manage this resource via the platform UI."

router = APIRouter(tags=["hybrid-mode"])


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


@router.api_route("/v1/usage/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/usage", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def usage_disabled() -> None:
    _raise_disabled()
