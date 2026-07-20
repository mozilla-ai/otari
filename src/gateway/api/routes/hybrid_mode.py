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


# The admin-dashboard management surface (settings, aliases, providers, pricing)
# is standalone-only for the same reason as the resources above: in hybrid mode
# these are owned by the platform. Stubbed so an operator hitting them gets the
# same "manage via the platform UI" hint instead of a bare 404.
@router.api_route("/v1/settings/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/settings", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def settings_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/aliases/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/aliases", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def aliases_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/providers/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/providers", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def providers_disabled() -> None:
    _raise_disabled()


@router.api_route("/v1/pricing/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@router.api_route("/v1/pricing", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def pricing_disabled() -> None:
    _raise_disabled()
