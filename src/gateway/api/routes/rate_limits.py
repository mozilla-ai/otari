"""The ``rate_limits`` rules, managed from the dashboard (``/api/v1/rate-limits``).

Lists every rule in effect, and adds, changes and removes the stored ones. A
rule from config.yml is listed and read-only here. Operator-gated and standalone
only, since a hybrid gateway refuses ``rate_limits``.
"""

from fastapi import APIRouter, Depends, Response, status

from gateway.api.deps import RateLimitServiceDep, require_deployment_operator
from gateway.schemas.rate_limits import (
    RateLimitRuleCreate,
    RateLimitRulePublic,
    RateLimitRulesPublic,
    RateLimitRuleUpdate,
)

router = APIRouter(
    prefix="/rate-limits",
    tags=["rate-limits"],
    dependencies=[Depends(require_deployment_operator)],
)


@router.get("")
async def list_rate_limit_rules(service: RateLimitServiceDep) -> RateLimitRulesPublic:
    """List every rule in effect: the config.yml rules, then the ones stored here."""
    return await service.list_rules()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_rate_limit_rule(service: RateLimitServiceDep, body: RateLimitRuleCreate) -> RateLimitRulePublic:
    """Add a rule. It applies from the next request on this replica, and on every replica within 30 seconds."""
    return await service.create_rule(body)


@router.patch("/{name}")
async def update_rate_limit_rule(
    service: RateLimitServiceDep, name: str, body: RateLimitRuleUpdate
) -> RateLimitRulePublic:
    """Change a stored rule. Requests it already counted stay counted. A config.yml rule answers 409."""
    return await service.update_rule(name, body)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rate_limit_rule(service: RateLimitServiceDep, name: str) -> Response:
    """Remove a stored rule. A config.yml rule answers 409."""
    await service.delete_rule(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
