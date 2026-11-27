"""Builders for the usage domain's services."""

from typing import Annotated

from fastapi import Depends

from gateway.api.deps._base import get_unit_of_work
from gateway.api.deps.api_keys import get_api_key_service
from gateway.api.deps.organizations import get_user_access
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.usage_and_telemetry import UsageLogRepository
from gateway.services.usage_and_telemetry import UsageService


def get_usage_service(uow: Annotated[UnitOfWork, Depends(get_unit_of_work)]) -> UsageService:
    """Build the usage service on the request's Unit of Work; its labels come from the key and user domains."""
    return UsageService(uow, UsageLogRepository(uow), keys=get_api_key_service(uow), users=get_user_access(uow))


UsageServiceDep = Annotated[UsageService, Depends(get_usage_service)]
