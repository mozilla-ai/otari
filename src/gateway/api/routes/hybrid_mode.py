from fastapi import APIRouter, HTTPException, status

_DISABLED_DETAIL = "This endpoint is not available in hybrid mode. Manage this resource via the platform UI."

# The method list every stub below shares. ``HEAD`` is enumerated rather than
# inherited, for the reason ``hosted_mode`` spells out: FastAPI derives it from
# ``GET`` only for a route that leaves its methods unspecified, so leaving it off
# answered 405, which says the path is served here and only the verb was wrong.
# One constant rather than the same literal on twenty-two decorators, so a verb
# added here reaches every stub in this file instead of the subset somebody
# remembered. ``hosted_mode`` keeps its own copy: the lists agree today and for
# the same reason, but a shared one would tie two modules together for a
# three-word literal.
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# Not published, for the reason ``hosted_mode`` gives: a refusal is a
# deployment posture, not an operation a client can call.
router = APIRouter(tags=["hybrid-mode"], include_in_schema=False)


def _raise_disabled() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)


@router.api_route("/users/{path:path}", methods=_METHODS)
@router.api_route("/users", methods=_METHODS)
async def users_disabled() -> None:
    _raise_disabled()


@router.api_route("/keys/{path:path}", methods=_METHODS)
@router.api_route("/keys", methods=_METHODS)
async def keys_disabled() -> None:
    _raise_disabled()


@router.api_route("/budgets/{path:path}", methods=_METHODS)
@router.api_route("/budgets", methods=_METHODS)
async def budgets_disabled() -> None:
    _raise_disabled()


@router.api_route("/usage/{path:path}", methods=_METHODS)
@router.api_route("/usage", methods=_METHODS)
async def usage_disabled() -> None:
    _raise_disabled()


# The admin-dashboard management surface (settings, aliases, providers, pricing)
# is standalone-only for the same reason as the resources above: in hybrid mode
# these are owned by the platform. Stubbed so an operator hitting them gets the
# same "manage via the platform UI" hint instead of a bare 404.
@router.api_route("/settings/{path:path}", methods=_METHODS)
@router.api_route("/settings", methods=_METHODS)
async def settings_disabled() -> None:
    _raise_disabled()


@router.api_route("/aliases/{path:path}", methods=_METHODS)
@router.api_route("/aliases", methods=_METHODS)
async def aliases_disabled() -> None:
    _raise_disabled()


@router.api_route("/providers/{path:path}", methods=_METHODS)
@router.api_route("/providers", methods=_METHODS)
async def providers_disabled() -> None:
    _raise_disabled()


@router.api_route("/pricing/{path:path}", methods=_METHODS)
@router.api_route("/pricing", methods=_METHODS)
async def pricing_disabled() -> None:
    _raise_disabled()


# Tenancy is the clearest case of the rule above: a hybrid gateway holds no
# tenancy state at all, because the platform's control plane is where its
# organizations and workspaces live.
@router.api_route("/organizations/{path:path}", methods=_METHODS)
@router.api_route("/organizations", methods=_METHODS)
async def organizations_disabled() -> None:
    _raise_disabled()


@router.api_route("/workspaces/{path:path}", methods=_METHODS)
@router.api_route("/workspaces", methods=_METHODS)
async def workspaces_disabled() -> None:
    _raise_disabled()


# Invitations are tenancy, same reasoning as organizations/workspaces above: a
# hybrid gateway holds no membership state to accept an invitation into, so
# there is nothing this stub could hand off to even if it tried.
@router.api_route("/invitations/{path:path}", methods=_METHODS)
@router.api_route("/invitations", methods=_METHODS)
async def invitations_disabled() -> None:
    _raise_disabled()
