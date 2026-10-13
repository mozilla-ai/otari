"""The end users of the service key a request authenticates with.

A service presents its own key here and reaches only the end users that key
bills, by the ids it names them by, so it holds neither the master key nor its
key's id. The operator's routes under ``/keys/{key_id}/end-users`` still reach
any key's end users.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status

from gateway.api.deps import BudgetServiceDep, ServiceKey, verify_service_key
from gateway.api.routes.users import TOTAL_COUNT_HEADER
from gateway.schemas.budgets import EndUserBudgetPublic, EndUserPublic, EndUserPut, EndUserUpdate

# On the router, so a route added here later cannot be reached with another credential.
router = APIRouter(prefix="/end-users", tags=["end-users"], dependencies=[Depends(verify_service_key)])
budgets_router = APIRouter(prefix="/end-user-budgets", tags=["end-users"], dependencies=[Depends(verify_service_key)])

ExternalId = Annotated[str, Path(description="The id the service names the end user by in a request's user field")]


@router.get("")
async def list_end_users(
    response: Response,
    api_key: ServiceKey,
    budgets: BudgetServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    budget_id: Annotated[str | None, Query(description="Only end users on this budget.")] = None,
    blocked: Annotated[bool | None, Query(description="Only end users that are, or are not, blocked.")] = None,
    include_total: Annotated[
        bool, Query(description="Also count every matching end user, in the Otari-Total-Count response header.")
    ] = False,
) -> list[EndUserPublic]:
    """List the end users of the calling service key, oldest first.

    Requires a service key. End users belong to the key's user, so every service key of one user lists the same
    end users. A deleted end user is not listed.
    """
    if include_total:
        total = await budgets.count_end_users(api_key=api_key, budget_id=budget_id, blocked=blocked)
        response.headers[TOTAL_COUNT_HEADER] = str(total)
    return await budgets.list_end_users(api_key=api_key, budget_id=budget_id, blocked=blocked, skip=skip, limit=limit)


@router.get("/{external_id:path}")
async def get_end_user(external_id: ExternalId, api_key: ServiceKey, budgets: BudgetServiceDep) -> EndUserPublic:
    """Get an end user of the calling service key by the id the service names it by.

    Requires a service key. An id no end user of the key's user goes by answers 404.
    """
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
    is refused with 403 and ``end_user_budget_not_allowed``.
    """
    end_user, created = await budgets.put_end_user(
        api_key=api_key, external_id=external_id, budget_id=request.budget_id
    )
    if created:
        response.status_code = status.HTTP_201_CREATED
    return end_user


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
    ``end_user_budget_not_allowed``.
    """
    return await budgets.update_end_user(
        api_key=api_key, external_id=external_id, blocked=request.blocked, budget_id=request.budget_id
    )


@budgets_router.get("")
async def list_end_user_budgets(api_key: ServiceKey, budgets: BudgetServiceDep) -> list[EndUserBudgetPublic]:
    """List the budgets the calling service key may put its end users on, with how many end users each holds.

    Requires a service key. The counts cover the end users of the key's user, blocked ones included and deleted
    ones not.
    """
    return await budgets.list_end_user_budgets(api_key=api_key)
