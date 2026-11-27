"""How a request reaches each tool the gateway runs: code execution, MCP, web search and fetch."""

import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Annotated, Any, cast

import httpx
from fastapi import Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    UnitOfWorkDep,
    get_config,
    get_log_writer,
    get_unit_of_work,
)
from gateway.api.deps.api_keys import (
    get_api_key_service,
)
from gateway.api.deps.auth import (
    verify_catalog_reader,
)
from gateway.api.deps.budgets import (
    get_budget_service,
)
from gateway.api.deps.ports import (
    BillingPortDep,
    CodeExecutionPortDep,
)
from gateway.core.config import (
    GatewayConfig,
)
from gateway.core.database import get_db, release_session
from gateway.core.error_codes import USER_MISMATCH, error_headers
from gateway.core.settings.tools import ToolKind
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work_with_session
from gateway.exceptions.tools_exceptions import DirectToolCallError, FundsCheckFailedError
from gateway.inflight import track_request
from gateway.models.api_keys import APIKey
from gateway.ports.billing_port import (
    BillingPort,
    InsufficientFundsError,
    ToolUse,
)
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.rate_limit import admit_rate_limit_rules, check_rate_limit, rate_limit_headers
from gateway.repositories.tenancy import (
    WorkspaceRepository,
)
from gateway.repositories.tools import (
    McpServerAliasRepository,
    McpUserConnectionRepository,
    OrgWebSearchKeyRepository,
    SearchToolRepository,
    WorkspaceCodeExecutionPolicyRepository,
    WorkspaceMcpServerOverrideRepository,
    WorkspaceMcpServerRepository,
    WorkspaceWebSearchKeyOverrideRepository,
)
from gateway.repositories.usage_and_telemetry import (
    new_usage_row,
)
from gateway.services.api_keys import (
    API_KEY_NO_USER_DETAIL,
    API_KEY_VALIDATION_FAILED_DETAIL,
    MASTER_KEY_USER_REQUIRED_DETAIL,
    USER_MISMATCH_DETAIL,
    ApiKeyService,
    names_end_user,
    resolve_user_id,
)
from gateway.services.budgets import (
    BudgetScopeRequest,
    BudgetService,
    ReservationHandle,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.inference import (
    DeploymentCharge,
    admit_deployment_paid,
    describe_port_error,
)
from gateway.services.log_writer import LogWriter
from gateway.services.playground_service import LocalToolOffer, PlaygroundToolOffer
from gateway.services.pricing_service import (
    resolve_tool_rate,
)
from gateway.services.tenancy import (
    OrganizationService,
)
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServers
from gateway.services.tenancy.workspace_web_search_service import WorkspaceWebSearchPolicies, WorkspaceWebSearchService
from gateway.services.tool_settings_service import load_overrides, stage_override
from gateway.services.tools import (
    CodeExecutionEnvironmentService,
    DirectCaller,
    DirectToolAvailability,
    DirectToolCall,
    DirectToolLedger,
    DirectToolRate,
    DirectToolService,
    DirectToolUsage,
    McpToolListings,
    McpUserTokens,
    McpWorkspaceServers,
    OrganizationMcpServerWorkspaceService,
    SearchCatalogCaller,
    SearchToolService,
    ToolRateInForce,
    WebSearchKeyService,
    WorkspaceCodeExecutionPolicies,
    WorkspaceCodeExecutionPolicyService,
    WorkspaceSearchKeys,
    web_search_configured,
)
from gateway.services.workspace_scope import (
    organization_for_key_id,
    workspace_for_key_id,
)
from gateway.types.tool_charges import RateKey


def get_workspace_code_execution_policy_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceCodeExecutionPolicyService:
    """Build the workspace code execution policy service on the request's Unit of Work.

    ``sandbox_configured`` is the check ``GET /api/v1/tools`` makes before it advertises code
    execution, so the page and discovery agree. ``default_enabled`` is what the request path gives a
    workspace with no policy. ``allowed_images`` is the operator's curated list. All are properties
    of the running gateway, handed in rather than read by the service.
    """
    return WorkspaceCodeExecutionPolicyService(
        uow,
        WorkspaceCodeExecutionPolicyRepository(uow),
        WorkspaceAccess(db, OrganizationService(db, membership_listener=None)),
        sandbox_configured=config.sandbox_configured(),
        default_enabled=config.code_execution_workspace_default,
        allowed_images=config.pinnable_sandbox_images(),
    )


WorkspaceCodeExecutionPolicyServiceDep = Annotated[
    WorkspaceCodeExecutionPolicyService, Depends(get_workspace_code_execution_policy_service)
]


async def get_workspace_code_execution_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceCodeExecutionPolicies:
    """Build the request path's reader of a workspace's code execution policy, on the request's session."""
    return WorkspaceCodeExecutionPolicies(
        WorkspaceCodeExecutionPolicyRepository(db), on_without_policy=config.code_execution_workspace_default
    )


WorkspaceCodeExecutionPoliciesDep = Annotated[
    WorkspaceCodeExecutionPolicies, Depends(get_workspace_code_execution_policies)
]


class _CodeExecutionEnvironmentPrices:
    """The environment listing's scope and price lookups, on the request's session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def workspace_of(self, api_key_id: str | None) -> uuid.UUID:
        return await workspace_for_key_id(self._db, api_key_id)

    async def organization_of(self, api_key_id: str | None) -> uuid.UUID | None:
        return await organization_for_key_id(self._db, api_key_id)

    async def unit_rate(self, rate_keys: Sequence[str], *, organization_id: uuid.UUID | None) -> Decimal | None:
        # The sandbox is the deployment's, so a run is deployment-paid and only the operator's rate counts.
        rate = await resolve_tool_rate(self._db, rate_keys, deployment_paid=True, organization_id=organization_id)
        return rate.unit_rate if rate is not None else None


def get_code_execution_environment_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    policies: WorkspaceCodeExecutionPoliciesDep,
) -> CodeExecutionEnvironmentService:
    """Build the sandbox environment listing on the request's session."""
    return CodeExecutionEnvironmentService(config=config, policies=policies, prices=_CodeExecutionEnvironmentPrices(db))


CodeExecutionEnvironmentServiceDep = Annotated[
    CodeExecutionEnvironmentService, Depends(get_code_execution_environment_service)
]


async def get_workspace_mcp_servers(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceMcpServers:
    """Build the request path's reader of a workspace's MCP servers, on the request's session."""
    return WorkspaceMcpServers(db, McpUserTokens(McpUserConnectionRepository(db), release=lambda: release_session(db)))


WorkspaceMcpServersDep = Annotated[WorkspaceMcpServers, Depends(get_workspace_mcp_servers)]


def get_mcp_tool_listings(request: Request) -> McpToolListings | None:
    """The app's kept MCP tool listings, or ``None`` outside the app's lifespan, where discovery keeps nothing."""
    listings: McpToolListings | None = getattr(request.app.state, "mcp_tool_listings", None)
    return listings


McpToolListingsDep = Annotated[McpToolListings | None, Depends(get_mcp_tool_listings)]


def get_web_provider_client(request: Request) -> httpx.AsyncClient:
    """The client the web tools' providers are called through, which the app's lifespan builds and closes."""
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    if client is None:
        msg = "No web provider client: the app's lifespan has not started, or has ended"
        raise RuntimeError(msg)
    return client


WebProviderClientDep = Annotated[httpx.AsyncClient, Depends(get_web_provider_client)]


async def get_optional_web_provider_client(request: Request) -> httpx.AsyncClient | None:
    """The app's web provider client, or ``None`` outside the app's lifespan, where a provider opens its own.

    For ``ToolPorts``, which every completion route takes, so a request that runs
    no web tool never depends on the client.
    """
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    return client


OptionalWebProviderClientDep = Annotated[httpx.AsyncClient | None, Depends(get_optional_web_provider_client)]


def get_web_search_key_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WebSearchKeyService:
    """Build the organization web search key service on the request's Unit of Work.

    The config tells a workspace without a key of a kind what its calls run on.
    """
    organizations = OrganizationService(db, membership_listener=None)
    return WebSearchKeyService(
        uow,
        keys=OrgWebSearchKeyRepository(uow),
        overrides=WorkspaceWebSearchKeyOverrideRepository(uow),
        organizations=organizations,
        workspaces=WorkspaceAccess(db, organizations),
        lock_workspace=WorkspaceRepository(db).lock,
        config=config,
    )


WebSearchKeyServiceDep = Annotated[WebSearchKeyService, Depends(get_web_search_key_service)]


def get_organization_mcp_server_workspaces(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationMcpServerWorkspaceService:
    """Build the service showing an organization's MCP server across its workspaces, on the request's Unit of Work."""
    return OrganizationMcpServerWorkspaceService(
        uow,
        servers=WorkspaceMcpServerRepository(uow),
        overrides=WorkspaceMcpServerOverrideRepository(uow),
        organizations=OrganizationService(db, membership_listener=None),
        lock_workspace=WorkspaceRepository(db).lock,
    )


OrganizationMcpServerWorkspacesDep = Annotated[
    OrganizationMcpServerWorkspaceService, Depends(get_organization_mcp_server_workspaces)
]


def get_workspace_search_keys(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceSearchKeys:
    """Build the resolver of the keys a workspace uses, on the request's session."""
    return WorkspaceSearchKeys(
        workspaces=WorkspaceRepository(db), overrides=WorkspaceWebSearchKeyOverrideRepository(db)
    )


def get_workspace_web_search_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchService:
    """Build the service on the request's session.

    The backend-presence check, with the workspace's organization key, is the
    one admission makes when it decides whether a request may declare
    ``otari_web_search``, and the default is the one the request path applies
    to a workspace with no row, so the page and the request path agree.
    """
    return WorkspaceWebSearchService(
        db,
        web_search_configured=web_search_configured(config),
        default_enabled=config.web_search_workspace_default,
        has_search_key=get_workspace_search_keys(db).has_search_key,
    )


WorkspaceWebSearchServiceDep = Annotated[WorkspaceWebSearchService, Depends(get_workspace_web_search_service)]


async def get_workspace_web_search_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchPolicies:
    """Build the request path's reader of a workspace's web search policy, on the request's session."""
    return WorkspaceWebSearchPolicies(
        db, search_keys=get_workspace_search_keys(db), on_without_policy=config.web_search_workspace_default
    )


WorkspaceWebSearchPoliciesDep = Annotated[WorkspaceWebSearchPolicies, Depends(get_workspace_web_search_policies)]


async def _deployment_tool_rate(
    session: AsyncSession, rate_key: RateKey, organization_id: uuid.UUID | None
) -> ToolRateInForce | None:
    """The rate one of the deployment's search or fetch tools settles at on its own account, as settlement reads it."""
    rate = await resolve_tool_rate(session, [rate_key], deployment_paid=True, organization_id=organization_id)
    if rate is None:
        return None
    return ToolRateInForce(rate.unit_rate, under_old_key=rate.old_key)


def build_search_tool_service(uow: UnitOfWork, session: AsyncSession, *, config: GatewayConfig) -> SearchToolService:
    """Build the search-tool service on ``uow``.

    ``session`` is the session ``uow`` runs on. The runtime tool settings are
    read and staged through it, so a default a create stores lands in the
    create's commit, and the price list is read through it.
    """
    return SearchToolService(
        uow,
        config=config,
        tools=SearchToolRepository(uow),
        read_settings=lambda: load_overrides(session, report_invalid=False),
        stage_setting=lambda key, value: stage_override(session, key, value),
        read_rate=lambda rate_key, organization_id: _deployment_tool_rate(session, rate_key, organization_id),
    )


def get_search_tool_service(
    uow: UnitOfWorkDep,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SearchToolService:
    """Build the search-tool service on the request's unit of work."""
    return build_search_tool_service(uow, db, config=config)


SearchToolServiceDep = Annotated[SearchToolService, Depends(get_search_tool_service)]


class _KeySearchCatalogCaller:
    """An API key reading the search tool catalog, answered by its workspace's web policy and its tool lists."""

    def __init__(
        self,
        config: GatewayConfig,
        api_key: APIKey,
        web_policies: WorkspaceWebSearchPolicies,
        api_keys: ApiKeyService,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._web_policies = web_policies
        self._api_keys = api_keys

    async def web_search_allowed(self) -> bool:
        allowed, _policy = await self._web_policies.access(self._api_key.workspace_id)
        return allowed

    async def tool_allowlist(self) -> list[str] | None:
        return await self._api_keys.tool_allowlist(self._config, self._api_key)


def get_search_catalog_caller(
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
    web_policies: Annotated[WorkspaceWebSearchPolicies, Depends(get_workspace_web_search_policies)],
    api_keys: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SearchCatalogCaller | None:
    """The API key reading the search tool catalog, or ``None`` for the master key or a dashboard session."""
    api_key, _ = auth
    if api_key is None:
        return None
    return _KeySearchCatalogCaller(config, api_key, web_policies, api_keys)


SearchCatalogCallerDep = Annotated[SearchCatalogCaller | None, Depends(get_search_catalog_caller)]


async def get_search_tool_tester(
    uow: UnitOfWorkDep,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SearchToolService:
    """Build the search-tool service for a connection test, once authentication's database work has ended.

    A test calls a provider, so no pooled connection is held across the call.
    """
    await release_session(db)
    return build_search_tool_service(uow, db, config=config)


SearchToolTesterDep = Annotated[SearchToolService, Depends(get_search_tool_tester)]


def build_search_tool_refresh(config: GatewayConfig) -> Callable[[], Awaitable[set[str]]]:
    """Build the search-tool refresher's tick, each on a unit of work of its own."""

    async def refresh() -> set[str]:
        async with create_unit_of_work_with_session() as (session, uow):
            return await build_search_tool_service(uow, session, config=config).refresh()

    return refresh


def get_playground_tool_offer(config: Annotated[GatewayConfig, Depends(get_config)]) -> PlaygroundToolOffer:
    """Return what this deployment offers a Playground message."""
    return LocalToolOffer(config)


PlaygroundToolOfferDep = Annotated[PlaygroundToolOffer, Depends(get_playground_tool_offer)]


@dataclass(frozen=True, kw_only=True)
class ToolPorts:
    """The ports and policies a request's tools reach, and the client their web providers are called through.

    ``web_provider_client`` is the app's; ``None`` outside its lifespan, or where
    a test builds this by hand, and a provider then opens its own for the call.
    """

    code_execution: CodeExecutionPort | None
    code_execution_policy: WorkspaceCodeExecutionPolicies
    mcp_server: WorkspaceMcpServers
    web_search_policy: WorkspaceWebSearchPolicies
    web_provider_client: httpx.AsyncClient | None = None


async def get_tool_ports(
    code_execution: CodeExecutionPortDep,
    code_execution_policy: WorkspaceCodeExecutionPoliciesDep,
    mcp_server: WorkspaceMcpServersDep,
    web_search_policy: WorkspaceWebSearchPoliciesDep,
    web_provider_client: OptionalWebProviderClientDep,
) -> ToolPorts:
    """Resolve the ports and policies a request's tools reach."""
    return ToolPorts(
        code_execution=code_execution,
        code_execution_policy=code_execution_policy,
        mcp_server=mcp_server,
        web_search_policy=web_search_policy,
        web_provider_client=web_provider_client,
    )


ToolPortsDep = Annotated[ToolPorts, Depends(get_tool_ports)]


class _DirectToolAccounts:
    """The direct tool service's accounts: the api-keys rules for who a call bills, and the budgets service."""

    def __init__(self, config: GatewayConfig, api_keys: ApiKeyService, budgets: BudgetService) -> None:
        self._config = config
        self._api_keys = api_keys
        self._budgets = budgets

    def names_end_user(self, caller: DirectCaller) -> bool:
        return names_end_user(caller.api_key, caller.user)

    def billed_user(self, caller: DirectCaller) -> str:
        return resolve_user_id(
            user_id_from_request=caller.user,
            api_key=caller.api_key,
            is_master_key=caller.is_master_key,
            master_key_error=DirectToolCallError(
                MASTER_KEY_USER_REQUIRED_DETAIL, status_code=status.HTTP_400_BAD_REQUEST
            ),
            no_api_key_error=DirectToolCallError(
                API_KEY_VALIDATION_FAILED_DETAIL, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            no_user_error=DirectToolCallError(
                API_KEY_NO_USER_DETAIL, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            forbidden_user_error=DirectToolCallError(
                f"{USER_MISMATCH_DETAIL} To bill each end user separately, use a service key.",
                status_code=status.HTTP_403_FORBIDDEN,
                headers=error_headers(USER_MISMATCH),
            ),
            reject_mismatch=self._config.reject_user_mismatch,
        )

    async def end_user(
        self, api_key: APIKey, external_id: str, *, requested_budget_id: str | None
    ) -> tuple[str, str | None]:
        resolved = await self._budgets.resolve_end_user(
            api_key=api_key, external_id=external_id, requested_budget_id=requested_budget_id
        )
        return resolved.user_id, resolved.budget_id

    async def tool_refusal(self, api_key: APIKey | None, *, kind: ToolKind, provider: str, tool: str) -> str | None:
        return await self._api_keys.tool_refusal(self._config, api_key, kind=kind, provider=provider, tool=tool)


def _ledger_on(db: AsyncSession, config: GatewayConfig, billing: BillingPort) -> DirectToolLedger:
    """The direct tool service's ledger: the budget, pricing and funding functions that take the request's session.

    Its methods close over that session.
    """

    class Ledger:
        async def workspace_of(self, api_key_id: str | None) -> uuid.UUID:
            return await workspace_for_key_id(db, api_key_id)

        async def organization_of(self, api_key_id: str | None) -> uuid.UUID | None:
            return await organization_for_key_id(db, api_key_id)

        async def unit_rate(
            self, call: DirectToolCall[Any], *, organization_id: uuid.UUID | None
        ) -> DirectToolRate | None:
            rate = await resolve_tool_rate(
                db,
                call.rate_keys,
                label=call.label,
                deployment_paid=call.deployment_paid,
                organization_id=organization_id,
            )
            return DirectToolRate(rate.unit_rate, rate.source) if rate is not None else None

        async def hold(
            self, user_id: str, amount: Decimal, *, api_key: APIKey | None, provider: str, counts_toward_budget: bool
        ) -> object:
            return await reserve_budget(
                db,
                user_id,
                amount,
                # Not the pricing key: ``model`` only drives reserve_budget's free-model
                # shortcut through any-llm, which a tool is not, and a tool named like an
                # any-llm provider could skip the reservation through it.
                model=None,
                strategy=config.budget_strategy,
                counts_toward_budget=counts_toward_budget,
                # A tool is not a model, but it is billed to a workspace like one, so
                # the scope carries the tool's provider as the narrowing axis.
                scope=BudgetScopeRequest(api_key=api_key, provider_instance=provider),
            )

        async def refund(self, hold: object) -> None:
            await refund_reservation(db, cast(ReservationHandle, hold))

        async def settle(self, hold: object, cost: Decimal) -> None:
            await reconcile_reservation(db, cast(ReservationHandle, hold), cost)

        async def admit_funding(
            self,
            *,
            organization_id: uuid.UUID | None,
            api_key_id: str | None,
            user_id: str,
            request_id: str,
            tool: str,
        ) -> object | None:
            try:
                return await admit_deployment_paid(
                    billing,
                    deployment_paid=True,
                    billed=None,
                    organization_id=organization_id,
                    api_key_id=api_key_id,
                    user_id=user_id,
                    request_id=request_id,
                    tools=frozenset({tool}),
                )
            except InsufficientFundsError:
                raise
            except Exception as exc:
                # Ends the transaction the port may have left failed, rolling it back
                # when it cannot commit, so the refund that follows has a usable session.
                await release_session(db)
                raise FundsCheckFailedError(describe_port_error(exc)) from exc

        async def settle_funding(
            self, funding: object, cost: Decimal, *, tools: Mapping[str, ToolUse], model: str, description: str
        ) -> None:
            await cast(DeploymentCharge, funding).settle(
                cost, tools=tools, model=model, description=description, expects_cost=False, savepoint=db.begin_nested
            )

        async def release(self) -> bool:
            return await release_session(db)

    return Ledger()


class _DirectToolGates:
    """The direct tool service's gates: the request's rate limits and in-flight entry, and the usage log writer."""

    def __init__(self, request: Request, config: GatewayConfig, budgets: BudgetService, log_writer: LogWriter) -> None:
        self._request = request
        self._config = config
        self._budgets = budgets
        self._log_writer = log_writer

    async def check_rate_limit(self, user_id: str) -> dict[str, str]:
        info = await check_rate_limit(self._request, user_id)
        return rate_limit_headers(info) if info else {}

    async def admit_rules(
        self, *, api_key_id: str | None, user_id: str, budget_exempt: bool, provider: str, tool: str
    ) -> None:
        grant = await admit_rate_limit_rules(
            self._request,
            key_id=api_key_id,
            user_id=user_id,
            estimated_tokens=0,
            budget_limits=None
            if budget_exempt
            else await self._budgets.minute_limits(user_id, strategy=self._config.budget_strategy),
        )
        if grant is not None:
            # A `per: model` rule names a tool by its pricing key, <provider>:<tool>.
            await grant.admit_model(provider, tool)

    def track(self, *, endpoint: str, model: str, provider: str, user_id: str, api_key_id: str | None) -> None:
        track_request(
            self._request, endpoint=endpoint, model=model, provider=provider, user_id=user_id, api_key_id=api_key_id
        )

    async def record(self, usage: DirectToolUsage) -> None:
        await self._log_writer.put(new_usage_row(**vars(usage)))


def get_direct_tool_service(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    budgets: Annotated[BudgetService, Depends(get_budget_service)],
    api_keys: Annotated[ApiKeyService, Depends(get_api_key_service)],
    policies: WorkspaceWebSearchPoliciesDep,
    billing: BillingPortDep,
    web_client: WebProviderClientDep,
) -> DirectToolService:
    """Build the request's direct tool service; its ledger and its billing port share the request's session."""
    return DirectToolService(
        config=config,
        accounts=_DirectToolAccounts(config, api_keys, budgets),
        ledger=_ledger_on(db, config, billing),
        gates=_DirectToolGates(request, config, budgets, log_writer),
        policies=policies,
        web_client=web_client,
    )


DirectToolServiceDep = Annotated[DirectToolService, Depends(get_direct_tool_service)]


def get_direct_tool_availability(
    config: Annotated[GatewayConfig, Depends(get_config)],
    api_keys: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> DirectToolAvailability:
    """Whether the direct search and fetch would admit a caller, checked as the direct tool service checks it."""
    return DirectToolAvailability(config, partial(api_keys.tool_refusal, config))


DirectToolAvailabilityDep = Annotated[DirectToolAvailability, Depends(get_direct_tool_availability)]


def get_mcp_workspace_servers(db: Annotated[AsyncSession, Depends(get_db)]) -> McpWorkspaceServers:
    """Build the reader of the MCP servers a workspace reaches, its organization's too, on the request's session."""
    return McpWorkspaceServers(
        WorkspaceMcpServerRepository(db), WorkspaceMcpServerOverrideRepository(db), McpServerAliasRepository(db)
    )


McpWorkspaceServersDep = Annotated[McpWorkspaceServers, Depends(get_mcp_workspace_servers)]
