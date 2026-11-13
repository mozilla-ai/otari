"""The sandbox environments a caller's workspace may run code in, with what a run in each costs."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

from gateway.core.config import GatewayConfig
from gateway.schemas.tools import CodeExecutionEnvironment, CodeExecutionEnvironmentList
from gateway.services.sandbox_backend import code_execution_rate_keys
from gateway.services.tools._code_execution_policy import WorkspaceCodeExecutionPolicies, leaves_a_served_tool


class CodeExecutionEnvironmentPrices(Protocol):
    """Where a key's runs are scoped and priced."""

    async def workspace_of(self, api_key_id: str | None) -> uuid.UUID:
        """The workspace whose code execution policy applies to the key's runs."""
        ...

    async def organization_of(self, api_key_id: str | None) -> uuid.UUID | None:
        """The organization whose prices apply to the key's runs."""
        ...

    async def unit_rate(self, rate_keys: Sequence[str], *, organization_id: uuid.UUID | None) -> Decimal | None:
        """The deployment-paid price of one run under these rate keys, the first priced one winning."""
        ...


class CodeExecutionEnvironmentService:
    """Which sandbox environments a key's workspace is offered, the default among them, and their prices."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        policies: WorkspaceCodeExecutionPolicies,
        prices: CodeExecutionEnvironmentPrices,
    ) -> None:
        self._config = config
        self._policies = policies
        self._prices = prices

    async def list_offered(
        self, api_key_id: str | None, *, skip: int = 0, limit: int = 100
    ) -> CodeExecutionEnvironmentList:
        """A page of the environments the key's workspace may name, in the operator's order.

        A workspace whose policy is off, or leaves no tool the sandbox serves, is offered none; one that pins an
        environment is offered that one alone, and none once the pin is off the operator's list. A deployment with
        no sandbox offers none, and one that names no image has none to choose. With no key, a master key or a
        dashboard session, which belongs to no single workspace, is offered the deployment's whole list.
        """
        if not self._config.sandbox_configured():
            return CodeExecutionEnvironmentList(data=[])
        policy = (
            await self._policies.for_request(await self._prices.workspace_of(api_key_id))
            if api_key_id is not None
            else None
        )
        offered = self._config.pinnable_sandbox_images()
        default = self._config.effective_sandbox_image()
        if policy is not None:
            if not policy.enabled or not leaves_a_served_tool(policy.tools):
                offered = ()
            elif policy.image is not None:
                offered = (policy.image,) if policy.image in offered else ()
                default = policy.image
        page = offered[skip : skip + limit]
        if not page:
            return CodeExecutionEnvironmentList(data=[])
        organization_id = await self._prices.organization_of(api_key_id) if api_key_id is not None else None
        environments: list[CodeExecutionEnvironment] = []
        for image in page:
            described = self._config.sandbox_environments.get(image)
            # One lookup per environment: the page is the operator's own image list, a handful of entries.
            unit_rate = await self._prices.unit_rate(code_execution_rate_keys(image), organization_id=organization_id)
            environments.append(
                CodeExecutionEnvironment(
                    name=image,
                    label=(described.label if described is not None else None) or image,
                    description=described.description if described is not None else None,
                    price_per_run=float(unit_rate) if unit_rate is not None else None,
                    default=image == default,
                )
            )
        return CodeExecutionEnvironmentList(data=environments)
