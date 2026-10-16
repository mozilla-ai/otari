"""The end users of the service key a request authenticates with.

A service presents its own key here and reaches only the end users that key
bills, by the ids it names them by, so it holds neither the master key nor its
key's id. The operator's routes under ``/keys/{key_id}/end-users`` still reach
any key's end users.

An end user is named in the path, ``/end-users/{external_id}``, or by the
``external_id`` query parameter on ``/end-users`` itself. An id may contain
'/', which proxies and HTTP clients can merge or collapse in a path, so the
query form is the one that names every end user intact. It sits on the
collection's own path, which no ``{external_id:path}`` route matches, so no
end user's id can shadow it.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status

from gateway.api.deps import BudgetServiceDep, ServiceKey, verify_service_key
from gateway.api.routes.users import TOTAL_COUNT_HEADER
from gateway.models.api_keys import APIKey
from gateway.schemas.budgets import EndUserBudgetPublic, EndUserPublic, EndUserPut, EndUserUpdate
from gateway.services.budgets import BudgetService, check_path_external_id


async def _require_available_owner(api_key: ServiceKey, budgets: BudgetServiceDep) -> None:
    """Refuse a key whose user is blocked or deleted, as its end users' requests are."""
    await budgets.require_end_user_owner(api_key=api_key)


# On the router, so a route added here later cannot be reached with another credential.
_GATES = [Depends(verify_service_key), Depends(_require_available_owner)]
router = APIRouter(prefix="/end-users", tags=["end-users"], dependencies=_GATES)
budgets_router = APIRouter(prefix="/end-user-budgets", tags=["end-users"], dependencies=_GATES)

ExternalId = Annotated[str, Path(description="The id the service names the end user by in a request's user field")]
ExternalIdParam = Annotated[
    str,
    Query(
        min_length=1,
        description="The id the service names the end user by in a request's user field, any id, '/' included",
    ),
]
ExternalIdFilter = Annotated[
    str | None,
    Query(min_length=1, description="Only the end user the service names with this `user` value: zero or one."),
]


@router.get("")
async def list_end_users(
    response: Response,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    budget_id: Annotated[str | None, Query(description="Only end users on this budget.")] = None,
    blocked: Annotated[bool | None, Query(description="Only end users that are, or are not, blocked.")] = None,
    external_id: ExternalIdFilter = None,
    include_total: Annotated[
        bool, Query(description="Also count every matching end user, in the Otari-Total-Count response header.")
    ] = False,
) -> list[EndUserPublic]:
    """List the end users of the calling service key, oldest first.

    Requires a service key. End users belong to the key's user, so every service key of one user lists the same
    end users. A deleted end user is not listed. ``external_id`` reads one end user by any id, one containing '/'
    included, as a list of zero or one.
    """
    if include_total:
        total = await budgets.count_end_users(
            api_key=api_key, external_id=external_id, budget_id=budget_id, blocked=blocked
        )
        response.headers[TOTAL_COUNT_HEADER] = str(total)
    return await budgets.list_end_users(
        api_key=api_key, external_id=external_id, budget_id=budget_id, blocked=blocked, skip=skip, limit=limit
    )


@router.put(
    "",
    responses={status.HTTP_201_CREATED: {"model": EndUserPublic, "description": "The end user was created"}},
)
async def put_end_user_by_query(
    external_id: ExternalIdParam,
    request: EndUserPut,
    response: Response,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
) -> EndUserPublic:
    """Put the end user named by ``external_id`` on a budget from the key's list, creating it if needed.

    The same as ``PUT /end-users/{external_id}``, for any id, one containing '/' included.
    """
    return await _put(external_id, request, response, api_key, budgets)


@router.patch("")
async def update_end_user_by_query(
    external_id: ExternalIdParam,
    request: EndUserUpdate,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
) -> EndUserPublic:
    """Block, unblock or move the end user named by ``external_id``.

    The same as ``PATCH /end-users/{external_id}``, for any id, one containing '/' included.
    """
    return await _update(external_id, request, api_key, budgets)


@router.get("/{external_id:path}")
async def get_end_user(external_id: ExternalId, api_key: ServiceKey, budgets: BudgetServiceDep) -> EndUserPublic:
    """Get an end user of the calling service key by the id the service names it by.

    Requires a service key. An id no end user of the key's user goes by answers 404. An id with a '.' or '..'
    segment is refused with 400; ``GET /end-users?external_id=`` reads any end user.
    """
    check_path_external_id(external_id)
    return await budgets.get_end_user(api_key=api_key, external_id=external_id)


@router.put(
    "/{external_id:path}",
    responses={status.HTTP_201_CREATED: {"model": EndUserPublic, "description": "The end user was created"}},
)
async def put_end_user(
    external_id: ExternalId,
    request: EndUserPut,
    response: Response,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
) -> EndUserPublic:
    """Put an end user of the calling service key on a budget from the key's list, creating it if needed.

    Requires a service key. Answers 201 when the end user was created. An end user already on the budget keeps its
    current period, so repeating the call changes nothing. A budget that is not on the key's ``end_user_budget_ids``
    is refused with 403 and ``end_user_budget_not_allowed``. An id with a '.' or '..' segment is refused with 400;
    ``PUT /end-users?external_id=`` takes any id.
    """
    check_path_external_id(external_id)
    return await _put(external_id, request, response, api_key, budgets)


@router.patch("/{external_id:path}")
async def update_end_user(
    external_id: ExternalId,
    request: EndUserUpdate,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
) -> EndUserPublic:
    """Block, unblock or move an end user of the calling service key.

    Requires a service key. A move restarts the end user's period on the new budget but keeps what it has spent and
    used so far. The budget must be on the key's ``end_user_budget_ids``, or the move is refused with 403 and
    ``end_user_budget_not_allowed``. An id with a '.' or '..' segment is refused with 400;
    ``PATCH /end-users?external_id=`` takes any id.
    """
    check_path_external_id(external_id)
    return await _update(external_id, request, api_key, budgets)


@budgets_router.get("")
async def list_end_user_budgets(api_key: ServiceKey, budgets: BudgetServiceDep) -> list[EndUserBudgetPublic]:
    """List the budgets the calling service key may put its end users on, with how many end users each holds.

    Requires a service key. The counts cover the end users of the key's user, blocked ones included and deleted
    ones not.
    """
    return await budgets.list_end_user_budgets(api_key=api_key)


async def _put(
    external_id: str, request: EndUserPut, response: Response, api_key: APIKey, budgets: BudgetService
) -> EndUserPublic:
    end_user, created = await budgets.put_end_user(
        api_key=api_key, external_id=external_id, budget_id=request.budget_id
    )
    if created:
        response.status_code = status.HTTP_201_CREATED
    return end_user


async def _update(external_id: str, request: EndUserUpdate, api_key: APIKey, budgets: BudgetService) -> EndUserPublic:
    return await budgets.update_end_user(
        api_key=api_key, external_id=external_id, blocked=request.blocked, budget_id=request.budget_id
    )
