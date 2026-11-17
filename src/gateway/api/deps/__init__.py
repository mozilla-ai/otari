import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Annotated, Any, cast

import httpx
from fastapi import BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    ContainerDep,
    MembershipListenerDep,
    PortSessionDep,
    UnitOfWorkDep,
    get_config,
    get_container,
    get_enabled_features,
    get_log_writer,
    get_membership_listener,
    get_unit_of_work,
    reset_config,
    set_config,
)
from gateway.api.deps._caller import (
    CallerOrganization,
    CallerUsageScope,
    CallerWorkspaceQuery,
    CallerWorkspaces,
)
from gateway.api.deps.api_keys import (
    ApiKeyFormatPortDep,
    ApiKeyServiceDep,
    FederationTokenExchangeDep,
    build_federation_token_exchange,
    get_api_key_service,
    get_federation_token_exchange,
)
from gateway.api.deps.auth import (
    AUTH_FAILURES,
    SERVICE_KEY_REQUIRED_DETAIL,
    CallingKey,
    ServiceKey,
    catalog_reader_operates_deployment,
    catalog_reader_organization,
    cookie_may_authenticate,
    extract_credential_token,
    get_session_identity,
    is_valid_master_key,
    misdirected_key_detail,
    record_auth_failure,
    refuse_ended_session,
    require_deployment_operator,
    verify_api_key,
    verify_api_key_or_master_key,
    verify_api_key_or_master_key_for_upload,
    verify_catalog_reader,
    verify_catalog_reader_or_public,
    verify_master_key,
    verify_service_key,
)
from gateway.api.deps.budgets import (
    BudgetServiceDep,
    build_budget_service,
    get_budget_service,
)
from gateway.api.deps.identity import (
    CurrentIdentity,
    get_current_identity,
    require_deployment_operator_or_absent,
)
from gateway.api.deps.organizations import get_user_access
from gateway.api.deps.ports import (
    AgentModelRecommenderPortDep,
    BillingPortDep,
    CodeExecutionPortDep,
    EntitlementPortDep,
    GrowthSignalPortDep,
    IdentityProviderPortDep,
    ModelProviderPortDep,
    PersonGrowthSignalPortDep,
    RequestSessionBillingPortDep,
    RequestSessionModelProviderPortDep,
    SignupInvitePortDep,
    TelemetryStoragePortDep,
    billing_port_on,
    get_agent_model_recommender_port,
    get_api_key_format_port,
    get_billing_port,
    get_code_execution_port,
    get_entitlement_port,
    get_feedback_delivery_port,
    get_growth_signal_port,
    get_identity_provider_port,
    get_model_provider_port,
    get_person_growth_signal_port,
    get_request_session_billing_port,
    get_request_session_model_provider_port,
    get_signup_invite_port,
    get_telemetry_storage_port,
    model_provider_port_on,
)
from gateway.api.deps.traces import (
    TRACE_CONTENT_HEADER_INVALID_DETAIL,
    DatabaseSessionOwners,
    NoSessionOwners,
    OrganizationBreakGlass,
    PlatformBreakGlass,
    TraceServiceDep,
    VisibleWorkspaces,
    VisibleWorkspacesDep,
    asks_to_keep_trace_content,
    build_trace_service,
    get_organization_break_glass,
    get_organization_trace_scope,
    get_platform_break_glass,
    get_trace_service,
    get_visible_workspaces,
    is_workspace_member,
    require_signed_in_reader,
    routed_trace_content,
    workspace_keeps_trace_content,
    workspace_members,
    workspaces_keeping_content_for,
)
from gateway.core.config import (
    API_ROOT,
    OAUTH_PROVIDERS,
    GatewayConfig,
)
from gateway.core.database import get_db, release_session
from gateway.core.settings.tools import ToolKind
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work, create_unit_of_work_with_session
from gateway.exceptions.tools_exceptions import DirectToolCallError, FundsCheckFailedError
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import MANAGEMENT_ROLES
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.billing_port import (
    BillingPort,
    InsufficientFundsError,
    ToolUse,
)
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.file_storage_port import FileStoragePort
from gateway.ports.model_provider_port import HostedModels
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.ports.usage_signal_port import UsageSignalPort
from gateway.rate_limit import admit_rate_limit_rules, check_rate_limit, rate_limit_headers
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.catalog import ModelMetadataOverrideRepository
from gateway.repositories.files import FileRepositories
from gateway.repositories.health_checks import (
    HealthCheckRecoveryRepository,
    HealthCheckResultRepository,
    HealthCheckRunRepository,
    HealthCheckSettingsRepository,
)
from gateway.repositories.identity import BlockedEmailRepository, IdentityRepositories, OAuthPendingStateRepository
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.repositories.inference import InferenceRepositories
from gateway.repositories.mcp_catalog import McpCatalogServerRepository, McpCatalogSyncStateRepository
from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.overview.overview_repository import OverviewRepository
from gateway.repositories.pricing import ModelPricingRepository, OrganizationModelPricingRepository
from gateway.repositories.providers import (
    HostedProviderModelRepository,
    HostedProviderRepository,
    OrgProviderKeyModelRepository,
    PlatformConnectRepository,
    ProviderEndpointRepository,
    WorkspaceHostedAccessRepository,
    WorkspaceProviderPolicyRepository,
)
from gateway.repositories.rate_limits import RateLimitRuleRepository
from gateway.repositories.routing import AgentRecommenderRepository
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationGuardrailDefinitionRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpServerRequestRepository,
    McpUserConnectionRepository,
    OrgWebSearchKeyRepository,
    SearchToolRepository,
    WorkspaceCodeExecutionPolicyRepository,
    WorkspaceMcpServerLogoRepository,
    WorkspaceMcpServerRepository,
    WorkspaceWebSearchKeyOverrideRepository,
)
from gateway.repositories.update_check import UpdateCheckStateRepository
from gateway.repositories.usage_and_telemetry import (
    UsageLogRepository,
    new_usage_row,
)
from gateway.repositories.users_repository import get_active_user
from gateway.services.api_keys import (
    API_KEY_NO_USER_DETAIL,
    API_KEY_VALIDATION_FAILED_DETAIL,
    MASTER_KEY_USER_REQUIRED_DETAIL,
    USER_MISMATCH_DETAIL,
    ApiKeyDeactivationListener,
    ApiKeyService,
    names_end_user,
    resolve_user_id,
)
from gateway.services.batch_service import claim_batch_accounting
from gateway.services.budgets import (
    BudgetScopeRequest,
    BudgetService,
    ReservationHandle,
    reconcile_reservation,
    record_external_spend,
    refund_reservation,
    reserve_budget,
)
from gateway.services.catalog import ModelMetadataOverrideService
from gateway.services.code_execution import SandboxContainerRegistry
from gateway.services.feedback import FeedbackService
from gateway.services.files import FileBackends, FileService, SandboxFileBridge, StagedFile
from gateway.services.health_checks import HealthCheckService, UnavailableModelListener
from gateway.services.identity import EmailBlocklist, IdentityService, OAuthAppService, OIDCConnectionService
from gateway.services.inference import (
    BatchResultsService,
    DeploymentCharge,
    InferenceService,
    UnavailableModelReporter,
    VideoJobService,
    VideoRateLookup,
    WorkspaceFunding,
    admit_deployment_paid,
    describe_port_error,
)
from gateway.services.log_writer import LogWriter
from gateway.services.mail import Mailer
from gateway.services.maintenance_mode_service import is_maintenance_mode
from gateway.services.mcp_catalog import McpCatalogService
from gateway.services.model_discovery_service import DiscoveryDial, HostedDiscovery
from gateway.services.oauth_service import provider_label, redirect_uri
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.organizations import (
    DeliveryRule,
    ExpiredInvitationCleanup,
    InvitationDelivery,
    InvitationFactsReader,
    InvitationMailLimit,
    OrganizationClaimedDomains,
)
from gateway.services.overview import OverviewService
from gateway.services.playground_service import LocalToolOffer, PlaygroundToolOffer
from gateway.services.pricing import DeploymentPricingService
from gateway.services.pricing_refresh_service import ModelsDevPriceUpdates
from gateway.services.pricing_service import (
    RequestPrices,
    ResolvedPricing,
    resolve_model_pricing,
    resolve_tool_rate,
)
from gateway.services.providers import (
    HostedCatalogUpdate,
    HostedProviderService,
    HostedSources,
    OrgProviderModelService,
    PlatformClient,
    PlatformConnectService,
    ProviderEndpointService,
    ScheduledRun,
    WorkspaceHostedAccessService,
    hosted_models_for,
    refresh_provider_endpoint_cache,
    refresh_workspace_hosted_access_cache,
)
from gateway.services.rate_limits import RateLimitService
from gateway.services.routing import (
    AgentRecommenderService,
    AutomaticRouter,
    RoutingService,
)
from gateway.services.tenancy import (
    DomainAutoJoinListener,
    InvitationClaimSettings,
    OrganizationDomainService,
    OrganizationService,
    VerifiedDomainLookup,
    organization_guardrail_runner,
)
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.deployment_user_service import DeploymentAdministrationAccess, DeploymentUserService
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService, refresh_org_provider_cache
from gateway.services.tenancy.organization_guardrail_definition_service import (
    OrganizationGuardrailDefinitionService,
)
from gateway.services.tenancy.organization_guardrail_service import OrganizationGuardrailService
from gateway.services.tenancy.workspace_activation_service import WorkspaceActivationService
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpConnectionEligibility,
    WorkspaceMcpServers,
    WorkspaceMcpServerService,
)
from gateway.services.tenancy.workspace_service import WorkspaceService
from gateway.services.tenancy.workspace_web_search_service import WorkspaceWebSearchPolicies, WorkspaceWebSearchService
from gateway.services.tool_settings_service import load_overrides, stage_override
from gateway.services.tools import (
    CodeExecutionEnvironmentService,
    DirectCaller,
    DirectToolCall,
    DirectToolLedger,
    DirectToolRate,
    DirectToolService,
    DirectToolUsage,
    McpConnectionService,
    McpServerLogoService,
    McpServerRequestService,
    McpSignedInServers,
    McpUserTokens,
    SearchCatalogCaller,
    SearchToolService,
    ToolRateInForce,
    WebSearchKeyService,
    WorkspaceCodeExecutionPolicies,
    WorkspaceCodeExecutionPolicyService,
    WorkspaceMcpServerUrls,
    WorkspaceSearchKeys,
    web_search_configured,
)
from gateway.services.update_check import UpdateCheckService
from gateway.services.usage_and_telemetry import UsageService, VideoUsageWriter
from gateway.services.workspace_scope import (
    default_workspace_id,
    organization_for_key_id,
    organization_for_workspace_id,
    workspace_for_key_id,
)
from gateway.types.tool_charges import RateKey
from gateway.types.traces import (
    ContentReader,
    ShareAudience,
    TraceScope,
)


def build_file_service(uow: UnitOfWork, backends: FileBackends, config: GatewayConfig) -> FileService:
    """Build Files operations for a scoped output request or cleanup job."""

    async def reject_unscoped_upload() -> uuid.UUID:
        raise RuntimeError("Unscoped uploads are not supported in this context; specify a workspace.")

    return FileService(uow, FileRepositories.on(uow), backends, config, reject_unscoped_upload)


def _file_backends(request: Request) -> FileBackends | None:
    """The backends this build bound for files, or ``None`` where it has none.

    A deployment with files off starts without a store it could not build.
    """
    storage: FileStoragePort | None = getattr(request.app.state, "file_store", None)
    provider_files: ProviderFilePort | None = getattr(request.app.state, "provider_files", None)
    if storage is None or provider_files is None:
        return None
    return FileBackends(storage=storage, provider_files=provider_files)


def build_inference_service(uow: UnitOfWork, config: GatewayConfig) -> InferenceService:
    """Build the inference service for a request or a worker job, such as the idempotency sweep.

    It receives the Unit of Work and its own domain's repositories over it, never the session.
    """
    return InferenceService(
        uow,
        InferenceRepositories.on(uow),
        config,
        country_limited_providers=build_hosted_provider_service(uow, config=config).names_limited_to_countries,
    )


def build_sandbox_file_bridge(
    *,
    raw_request: Request,
    config: GatewayConfig,
    uow: UnitOfWork,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    inputs: list[StagedFile],
) -> SandboxFileBridge | None:
    """The file bridge a completion request's sandbox session gets, or ``None``.

    Built by the route once the billed user and workspace are resolved, on the request's own Unit of Work.
    It is ``None`` when files are disabled.
    Produced files are announced under ``public_base_url`` where the deployment knows its address,
    and otherwise under the one the request arrived on.
    """
    backends = _file_backends(raw_request)
    if not config.files_enabled or backends is None or user_id is None or workspace_id is None:
        return None
    base = (config.public_base_url or str(raw_request.base_url)).rstrip("/")
    return SandboxFileBridge(
        backends=backends,
        config=config,
        files=build_file_service(uow, backends, config),
        user_id=user_id,
        workspace_id=workspace_id,
        inputs=inputs,
        base_url=f"{base}{API_ROOT}/files",
    )


def build_sandbox_container_registry(
    *,
    config: GatewayConfig,
    uow: UnitOfWork,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    port: CodeExecutionPort | None,
    may_read_stored_files: bool,
) -> SandboxContainerRegistry | None:
    """The registry a completion request's sandbox session is held in, or ``None``.

    ``None`` is every case in which a sandbox cannot outlive its request: a
    deployment with no sandbox at all; and one that turned reuse off with a zero idle TTL. The
    lease is scoped to the billed user and workspace, and to the adapter this
    build runs, so a deployment that changes providers starts fresh.

    NOTE: ``may_read_stored_files`` has no default, so a caller cannot drop the check by accident.
    A container keeps what earlier runs staged into it, so re-entering one reads those files.
    """
    if user_id is None or workspace_id is None or port is None or config.sandbox_container_idle_ttl_sec <= 0:
        return None
    return SandboxContainerRegistry(
        uow=uow,
        user_id=user_id,
        workspace_id=workspace_id,
        may_read_stored_files=may_read_stored_files,
        provider=port.label,
        idle_ttl_s=config.sandbox_container_idle_ttl_sec,
        max_lifetime_s=config.sandbox_container_max_lifetime_sec,
    )


def build_model_metadata_override_service(uow: UnitOfWork) -> ModelMetadataOverrideService:
    """Build the model details service on ``uow``, for the request dependency and the services that compose it."""
    return ModelMetadataOverrideService(uow, overrides=ModelMetadataOverrideRepository(uow))


def get_model_metadata_override_service(uow: UnitOfWorkDep) -> ModelMetadataOverrideService:
    return build_model_metadata_override_service(uow)


ModelMetadataOverrideServiceDep = Annotated[ModelMetadataOverrideService, Depends(get_model_metadata_override_service)]


async def get_maintenance_mode(db: Annotated[AsyncSession, Depends(get_db)]) -> bool:
    """Return whether new dashboard sign-ins are frozen right now."""
    return await is_maintenance_mode(db)


MaintenanceModeDep = Annotated[bool, Depends(get_maintenance_mode)]


def get_inference_service(
    uow: UnitOfWorkDep, config: Annotated[GatewayConfig, Depends(get_config)]
) -> InferenceService:
    """Build the inference service on the request's unit of work."""
    return build_inference_service(uow, config)


InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]


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


def get_mcp_signed_in_servers(db: Annotated[AsyncSession, Depends(get_db)]) -> McpSignedInServers:
    """Build the read of which per-user MCP servers a user has signed in to, on the request's session."""
    return McpSignedInServers(McpUserConnectionRepository(db))


McpSignedInServersDep = Annotated[McpSignedInServers, Depends(get_mcp_signed_in_servers)]


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


def get_playground_tool_offer(config: Annotated[GatewayConfig, Depends(get_config)]) -> PlaygroundToolOffer:
    """Return what this deployment offers a Playground message."""
    return LocalToolOffer(config)


PlaygroundToolOfferDep = Annotated[PlaygroundToolOffer, Depends(get_playground_tool_offer)]


# =============================================================================
# Composition root
# =============================================================================
#
# Every port is resolved here and nowhere else: a dependency names the port and
# asks the container for whichever adapter this build bound to it, so no route
# or service ever names a concrete adapter (ARCHITECTURE.md, rule 5).


async def get_workspace_mcp_servers(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceMcpServers:
    """Build the request path's reader of a workspace's MCP servers, on the request's session."""
    return WorkspaceMcpServers(db, McpUserTokens(McpUserConnectionRepository(db), release=lambda: release_session(db)))


def get_web_provider_client(request: Request) -> httpx.AsyncClient:
    """The client the web tools' providers are called through, which the app's lifespan builds and closes."""
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    if client is None:
        msg = "No web provider client: the app's lifespan has not started, or has ended"
        raise RuntimeError(msg)
    return client


async def get_optional_web_provider_client(request: Request) -> httpx.AsyncClient | None:
    """The app's web provider client, or ``None`` outside the app's lifespan, where a provider opens its own.

    For ``ToolPorts``, which every completion route takes, so a request that runs
    no web tool never depends on the client.
    """
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    return client


def get_overview_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> OverviewService:
    """Build the dashboard overview's summary service on the request's session.

    Assembled here rather than in the route, because a route does not name a
    session type (``scripts/check_architecture.py``, rule 12).
    """
    return OverviewService(
        OverviewRepository(db),
        OrganizationService(db, membership_listener=None),
        DeploymentAdministrationAccess(db),
        # The listener is for writes; this service only reads, and the same
        # pairing is what `get_workspace_service` builds.
        WorkspaceService(db, uow=uow, membership_listener=membership_listener),
    )


OverviewServiceDep = Annotated[OverviewService, Depends(get_overview_service)]


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the request's organization service.

    It reads only. A membership write needs the listener this pairing leaves unset.
    """
    return OrganizationService(db, membership_listener=None)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


def get_deployment_user_service(
    db: Annotated[AsyncSession, Depends(get_db)], uow: UnitOfWorkDep
) -> DeploymentUserService:
    """Wire account changes and self-service key deactivation to the request's single transaction."""
    return DeploymentUserService(db, uow=uow, keys=ApiKeyDeactivationListener(ApiKeyRepositories.on(uow)))


DeploymentUserServiceDep = Annotated[DeploymentUserService, Depends(get_deployment_user_service)]


def get_deployment_administration_access(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentAdministrationAccess:
    """Read deployment administration authority without constructing mutation collaborators."""
    return DeploymentAdministrationAccess(db)


DeploymentAdministrationAccessDep = Annotated[
    DeploymentAdministrationAccess, Depends(get_deployment_administration_access)
]


def get_membership_organization_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> OrganizationService:
    """Build the request's organization service for writing memberships and creating workspaces."""
    return OrganizationService(db, membership_listener=membership_listener, uow=uow)


MembershipOrganizationServiceDep = Annotated[OrganizationService, Depends(get_membership_organization_service)]


def get_organization_domain_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationDomainService:
    """Build the email-domain service on the request's session."""
    return OrganizationDomainService(db)


OrganizationDomainServiceDep = Annotated[OrganizationDomainService, Depends(get_organization_domain_service)]


def get_invitation_claim_settings(config: Annotated[GatewayConfig, Depends(get_config)]) -> InvitationClaimSettings:
    """Return what this deployment asks of, and offers, an invitee who chooses a first password from an invitation link.

    The terms it publishes, and the ways in a refusal can offer instead: its sign-in providers by display name,
    and a reset link where mail can send one.
    """
    return InvitationClaimSettings(
        terms_url=config.terms_url,
        sign_in_providers=tuple(provider_label(name) for name in config.oauth_providers),
        can_send_links=Mailer(config).can_send_links,
    )


InvitationClaimSettingsDep = Annotated[InvitationClaimSettings, Depends(get_invitation_claim_settings)]


def get_invitation_mail_limit(
    config: Annotated[GatewayConfig, Depends(get_config)], container: ContainerDep
) -> InvitationMailLimit:
    """Return the invitation mail limits, counted in the rate-limit store this build bound."""
    return InvitationMailLimit(
        container.resolve(RateLimitStorePort, None),
        per_organization=config.invitation_mail_limit_per_organization,
        per_address=config.invitation_mail_limit_per_address,
    )


InvitationMailLimitDep = Annotated[InvitationMailLimit, Depends(get_invitation_mail_limit)]


def get_organization_guardrail_definition_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationGuardrailDefinitionService:
    """Build an organization's guardrail-definition service on the request's Unit of Work.

    Assembled here for the reason the overview service above is: a route does not
    name a session type (``scripts/check_architecture.py``, rule 12).

    The session is here only for the role gate. The service itself holds none,
    reaching the database through its repository and committing in the blocks it
    opens; `OrganizationService` is still in the old shape and takes a session,
    and both are the request's one session, so a block commits what the gate
    read.

    The runner is wired in here rather than imported by the service, because the
    runner imports the service for the function that undoes the secret split and
    the pair would otherwise form a cycle. This is the composition root, which
    is where that join belongs anyway.
    """
    return OrganizationGuardrailDefinitionService(
        definitions=OrganizationGuardrailDefinitionRepository(uow),
        organizations=OrganizationService(db, membership_listener=None),
        uow=uow,
        build_state=organization_guardrail_runner.build_state,
        rebuild=organization_guardrail_runner.rebuild_definition,
        handle=organization_guardrail_runner.handle,
    )


OrganizationGuardrailDefinitionServiceDep = Annotated[
    OrganizationGuardrailDefinitionService, Depends(get_organization_guardrail_definition_service)
]


def get_workspace_funding(db: PortSessionDep, billing: BillingPortDep) -> WorkspaceFunding:
    """Read a workspace's organization's funds the way a deployment-paid request's gate reads them."""
    return WorkspaceFunding(billing, partial(organization_for_workspace_id, db))


WorkspaceFundingDep = Annotated[WorkspaceFunding, Depends(get_workspace_funding)]
WorkspaceMcpServersDep = Annotated[WorkspaceMcpServers, Depends(get_workspace_mcp_servers)]


def get_workspace_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> WorkspaceService:
    return WorkspaceService(db, uow=uow, membership_listener=membership_listener)


WorkspaceServiceDep = Annotated[WorkspaceService, Depends(get_workspace_service)]


def get_workspace_activation_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    key_format: ApiKeyFormatPortDep,
) -> WorkspaceActivationService:
    return WorkspaceActivationService(db, config, key_format)


WorkspaceActivationServiceDep = Annotated[WorkspaceActivationService, Depends(get_workspace_activation_service)]


def get_mcp_connection_service(
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> McpConnectionService:
    """Build the service that runs a user's sign-in to a per-user MCP server, on the request's Unit of Work."""
    return McpConnectionService(
        uow,
        WorkspaceMcpServerRepository(uow),
        McpUserConnectionRepository(uow),
        McpOAuthFlowRepository(uow),
        config,
        eligibility=WorkspaceMcpConnectionEligibility(
            uow,
            WorkspaceRepository(uow),
            WorkspaceMemberRepository(uow),
            OrganizationMemberRepository(uow),
            UserRepository(uow),
        ),
        users=get_user_access(uow),
    )


McpConnectionServiceDep = Annotated[McpConnectionService, Depends(get_mcp_connection_service)]


def get_org_provider_key_service(
    db: Annotated[AsyncSession, Depends(get_db)], config: Annotated[GatewayConfig, Depends(get_config)]
) -> OrgProviderKeyService:
    return OrgProviderKeyService(db, config)


OrgProviderKeyServiceDep = Annotated[OrgProviderKeyService, Depends(get_org_provider_key_service)]


def get_organization_guardrail_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationGuardrailService:
    return OrganizationGuardrailService(db)


OrganizationGuardrailServiceDep = Annotated[OrganizationGuardrailService, Depends(get_organization_guardrail_service)]


def get_identity_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    identity_provider: IdentityProviderPortDep,
    membership_listener: MembershipListenerDep,
) -> IdentityService:
    """Build the identity service on the request's Unit of Work.

    Its own organization service is built on the same Unit of Work, not the deployment's shared
    read-only one, because provisioning an SSO sign-in's account writes through it. It carries
    the membership listener because the sign-in paths built here grant a workspace membership,
    which has to be announced.
    """
    organizations = OrganizationService(db, membership_listener=membership_listener, uow=uow)
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=identity_provider,
        domain_claims=VerifiedDomainLookup(OrganizationDomainRepository(uow)),
        organizations=organizations,
        sign_in_listener=DomainAutoJoinListener(
            OrganizationDomainRepository(uow), OrganizationMemberRepository(uow), organizations
        ),
        session_ttl_hours=config.dashboard_session_ttl_hours,
        terms_url=config.terms_url,
        mailer=Mailer(config),
        open_signup=config.open_signup,
        password_reset_expiry_hours=config.password_reset_expiry_hours,
    )


IdentityServiceDep = Annotated[IdentityService, Depends(get_identity_service)]


def get_oidc_connection_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    organizations: OrganizationServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OIDCConnectionService:
    """Build the organization OIDC connection service on the request's Unit of Work.

    ``sign_in_states`` is the one repository built on the session instead, for the reason the
    service's own attribute gives: the sign-in client is used after the block that built it.

    ``sso_redirect_uri`` is None where this deployment has no public address,
    because a path alone is not registerable at a provider.
    """
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=organizations,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
        sign_in_states=OAuthPendingStateRepository(db),
        allow_private_hosts=config.oidc_allow_private_hosts,
        passkey_rp_id=config.webauthn_relying_party.rp_id if config.webauthn_relying_party else None,
        offered_providers=frozenset(
            provider for provider in OAUTH_PROVIDERS if config.oauth_client_credentials(provider) is not None
        ),
        sso_redirect_uri=redirect_uri(config, "sso") if config.public_base_url else None,
    )


OIDCConnectionServiceDep = Annotated[OIDCConnectionService, Depends(get_oidc_connection_service)]


def get_org_provider_model_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    model_provider: RequestSessionModelProviderPortDep,
) -> OrgProviderModelService:
    """Build the offered-models service on the request's session and unit of work.

    The service itself names neither the session nor SQLAlchemy, so its
    repositories and its cache-refresh callable are assembled here. The unit of
    work and the services built on the session are over the *same* session (see
    ``deps.get_unit_of_work``), so a block's commit also settles what they staged.
    """
    return OrgProviderModelService(
        uow,
        config=config,
        organizations=OrganizationService(db, membership_listener=None),
        provider_keys=OrgProviderKeyService(db, config),
        org_pricing=OrganizationPricingService(db, config, model_provider=model_provider),
        models=OrgProviderKeyModelRepository(uow),
        keys=OrgProviderKeyRepository(db),
        metadata_overrides=build_model_metadata_override_service(uow),
        refresh_overlay=lambda: refresh_org_provider_cache(db),
    )


OrgProviderModelServiceDep = Annotated[OrgProviderModelService, Depends(get_org_provider_model_service)]


def get_provider_endpoint_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ProviderEndpointService:
    """Build the owned provider endpoint service on the request's session and unit of work."""

    async def resolve_workspace(workspace_id: uuid.UUID | None) -> uuid.UUID | None:
        if workspace_id is None:
            return await default_workspace_id(db)
        return workspace_id if await WorkspaceRepository(db).get(workspace_id) is not None else None

    async def user_is_active(user_id: str) -> bool:
        return await get_active_user(db, user_id) is not None

    return ProviderEndpointService(
        uow,
        config=config,
        endpoints=ProviderEndpointRepository(uow),
        resolve_workspace=resolve_workspace,
        user_is_active=user_is_active,
        refresh_cache=lambda: refresh_provider_endpoint_cache(uow),
    )


ProviderEndpointServiceDep = Annotated[ProviderEndpointService, Depends(get_provider_endpoint_service)]


def get_workspace_access(
    db: Annotated[AsyncSession, Depends(get_db)],
    organizations: OrganizationServiceDep,
) -> WorkspaceAccess:
    """Build the workspace visibility and management checks on the request's session."""
    return WorkspaceAccess(db, organizations)


WorkspaceAccessDep = Annotated[WorkspaceAccess, Depends(get_workspace_access)]


def get_workspace_hosted_access_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    access: WorkspaceAccessDep,
    org_provider_keys: OrgProviderKeyServiceDep,
    model_provider: RequestSessionModelProviderPortDep,
) -> WorkspaceHostedAccessService:
    """Build the workspace hosted provider access service on the request's unit of work.

    The hosted port answers which providers an organization may use, and the
    organization's own keys which of them a workspace never asks the port
    about. Both are other domains' answers, handed in as callables.
    """

    async def hosted_models(organization_id: uuid.UUID | None) -> HostedModels:
        return await hosted_models_for(model_provider, organization_id=organization_id)

    async def byo_providers(workspace_id: uuid.UUID) -> frozenset[str]:
        return await org_provider_keys.get_workspace_byo_providers(workspace_id=workspace_id)

    return WorkspaceHostedAccessService(
        uow,
        rules=WorkspaceHostedAccessRepository(uow),
        policies=WorkspaceProviderPolicyRepository(uow),
        access=access,
        hosted_models=hosted_models,
        byo_providers=byo_providers,
        refresh_cache=lambda: refresh_workspace_hosted_access_cache(uow),
    )


WorkspaceHostedAccessServiceDep = Annotated[WorkspaceHostedAccessService, Depends(get_workspace_hosted_access_service)]


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


def get_workspace_search_keys(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceSearchKeys:
    """Build the resolver of the keys a workspace uses, on the request's session."""
    return WorkspaceSearchKeys(
        workspaces=WorkspaceRepository(db), overrides=WorkspaceWebSearchKeyOverrideRepository(db)
    )


async def get_workspace_web_search_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchPolicies:
    """Build the request path's reader of a workspace's web search policy, on the request's session."""
    return WorkspaceWebSearchPolicies(
        db, search_keys=get_workspace_search_keys(db), on_without_policy=config.web_search_workspace_default
    )


def build_deployment_pricing_service(uow: UnitOfWork) -> DeploymentPricingService:
    """The one assembly, shared by the hosted-providers service and startup's config pricing seed."""
    return DeploymentPricingService(
        uow, pricing=ModelPricingRepository(uow), overrides=OrganizationModelPricingRepository(uow)
    )


def build_hosted_provider_service(uow: UnitOfWork, *, config: GatewayConfig) -> HostedProviderService:
    """Build the hosted-providers service on ``uow``.

    The one assembly, used by the request dependency below and handed to the
    composition root for the model provider adapter, the way the membership
    listener's builder is. The deployment price list is reached through the
    pricing domain's own service on the same unit of work, so a rate written
    while offering a model lands in the same transaction as the offer.
    """
    providers = HostedProviderRepository(uow)
    return HostedProviderService(
        uow,
        config=config,
        providers=providers,
        sources=HostedSources(config, providers),
        models=HostedProviderModelRepository(uow),
        endpoints=ProviderEndpointRepository(uow),
        pricing=build_deployment_pricing_service(uow),
        live_byo_pairs=OrgProviderKeyRepository(uow).live_provider_pairs,
        organization_exists=OrganizationRepository(uow).exists,
        metadata_overrides=build_model_metadata_override_service(uow),
    )


def get_hosted_provider_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> HostedProviderService:
    """Build the hosted-providers service on the request's unit of work."""
    return build_hosted_provider_service(uow, config=config)


HostedProviderServiceDep = Annotated[HostedProviderService, Depends(get_hosted_provider_service)]


def get_platform_client(config: Annotated[GatewayConfig, Depends(get_config)]) -> PlatformClient:
    """The otari.ai endpoints a connection uses, at the configured ``platform_url``."""
    return PlatformClient(config.platform_url)


def get_platform_connect_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    hosted: HostedProviderServiceDep,
    client: Annotated[PlatformClient, Depends(get_platform_client)],
) -> PlatformConnectService:
    """Build the otari.ai connect service on the request's unit of work."""
    return PlatformConnectService(uow, attempts=PlatformConnectRepository(uow), hosted=hosted, client=client)


PlatformConnectServiceDep = Annotated[PlatformConnectService, Depends(get_platform_connect_service)]


def get_agent_recommender_service(uow: Annotated[UnitOfWork, Depends(get_unit_of_work)]) -> AgentRecommenderService:
    """Build the subagent router service on the request's unit of work."""
    return AgentRecommenderService(uow, AgentRecommenderRepository(uow))


AgentRecommenderServiceDep = Annotated[AgentRecommenderService, Depends(get_agent_recommender_service)]


def build_hosted_catalog_update(uow: UnitOfWork, *, config: GatewayConfig) -> HostedCatalogUpdate:
    """Build the catalog update on ``uow``; its price steps open sessions of their own."""
    return HostedCatalogUpdate(build_hosted_provider_service(uow, config=config), ModelsDevPriceUpdates(config))


def get_hosted_catalog_update(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> HostedCatalogUpdate:
    """The catalog update for one request, on its Unit of Work."""
    return build_hosted_catalog_update(uow, config=config)


HostedCatalogUpdateDep = Annotated[HostedCatalogUpdate, Depends(get_hosted_catalog_update)]


def build_scheduled_catalog_update(config: GatewayConfig) -> ScheduledRun:
    """Build the catalog update worker's run, each on a Unit of Work of its own."""

    async def run(policy: str, interval_seconds: float) -> str | None:
        async with create_unit_of_work() as uow:
            return await build_hosted_catalog_update(uow, config=config).run_scheduled(policy, interval_seconds)

    return run


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


def build_hosted_discovery(config: GatewayConfig) -> HostedDiscovery:
    """Build the discovery worker's read of the hosted providers' dials.

    Each call reads in a unit of work of its own, which closes before any dial
    runs, so no session is held across an upstream call.
    """

    async def dials() -> dict[str, DiscoveryDial]:
        async with create_unit_of_work() as uow:
            return await build_hosted_provider_service(uow, config=config).discovery_dials()

    return dials


def get_rate_limit_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RateLimitService:
    """Build the rate-limit rules service on the request's unit of work."""
    return RateLimitService(uow, RateLimitRuleRepository(uow), config)


RateLimitServiceDep = Annotated[RateLimitService, Depends(get_rate_limit_service)]


def build_health_check_service(uow: UnitOfWork, config: GatewayConfig) -> HealthCheckService:
    """Build the health checks service on a request's Unit of Work, or the worker's own.

    The service is handed the Unit of Work, never the session under it, with its own domain's repositories and
    the providers domain's service for the hosted roster, as DOMAINS.md's constructor injection asks.
    """
    return HealthCheckService(
        uow,
        HealthCheckSettingsRepository(uow),
        HealthCheckRunRepository(uow),
        HealthCheckResultRepository(uow),
        HealthCheckRecoveryRepository(uow),
        build_hosted_provider_service(uow, config=config),
        config,
    )


def get_health_check_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> HealthCheckService:
    """The request's service. The poll builds its own through ``build_health_check_service``."""
    return build_health_check_service(uow, config)


HealthCheckServiceDep = Annotated[HealthCheckService, Depends(get_health_check_service)]


def build_mcp_catalog_service(
    uow: UnitOfWork, config: GatewayConfig, workspaces: WorkspaceAccess | None = None
) -> McpCatalogService:
    """Build the MCP catalog service on a request's Unit of Work, or the sync worker's own.

    The worker passes no workspace checks: it only syncs, and who may ask for a sync is a request's question.
    """
    return McpCatalogService(
        uow,
        McpCatalogServerRepository(uow),
        McpCatalogSyncStateRepository(uow),
        workspaces,
        config,
        workspace_servers=WorkspaceMcpServerUrls(WorkspaceMcpServerRepository(uow)),
    )


def get_mcp_catalog_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspaces: WorkspaceAccessDep,
) -> McpCatalogService:
    """The request's service. The sync worker builds its own through ``build_mcp_catalog_service``."""
    return build_mcp_catalog_service(uow, config, workspaces)


async def master_key_caller_operates_deployment(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    _master_key: Annotated[str | None, Depends(verify_master_key)],
) -> bool:
    """Whether a caller :func:`verify_master_key` admitted operates the deployment.

    The question :func:`require_deployment_operator` asks, answered rather than enforced, for a route
    that admits every signed-in caller and holds an operator to a looser rule. A session is put to
    ``DeploymentAdministrationAccess.has_administration_access``; without one, the caller holds the master key.
    """
    if session_identity is not None:
        return await DeploymentAdministrationAccess(db).has_administration_access(session_identity)
    return True


OperatesDeployment = Annotated[bool, Depends(master_key_caller_operates_deployment)]
McpCatalogServiceDep = Annotated[McpCatalogService, Depends(get_mcp_catalog_service)]


def get_mcp_server_logo_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    workspaces: WorkspaceAccessDep,
    catalog: McpCatalogServiceDep,
) -> McpServerLogoService:
    """Build the MCP server logo service on the request's Unit of Work, with the catalog as its logo source."""
    return McpServerLogoService(
        uow, WorkspaceMcpServerRepository(uow), WorkspaceMcpServerLogoRepository(uow), workspaces, catalog
    )


McpServerLogoServiceDep = Annotated[McpServerLogoService, Depends(get_mcp_server_logo_service)]


def get_mcp_server_request_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    workspaces: WorkspaceAccessDep,
    catalog: McpCatalogServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    background_tasks: BackgroundTasks,
) -> McpServerRequestService:
    """Build the service for members' MCP server requests on the request's Unit of Work.

    Its mail goes out as a background task of the request, after the response.
    """
    return McpServerRequestService(
        uow,
        McpServerRequestRepository(uow),
        WorkspaceMcpServerRepository(uow),
        workspaces,
        catalog,
        Mailer(config),
        send_later=background_tasks.add_task,
    )


McpServerRequestServiceDep = Annotated[McpServerRequestService, Depends(get_mcp_server_request_service)]


def get_workspace_mcp_server_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    api_keys: ApiKeyServiceDep,
    connections: McpConnectionServiceDep,
    requests: McpServerRequestServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceMcpServerService:
    """Build the service on the request's session, with the keys its access lists name and the sign-ins it starts."""
    return WorkspaceMcpServerService(db, api_keys, connections=connections, requests=requests, config=config)


WorkspaceMcpServerServiceDep = Annotated[WorkspaceMcpServerService, Depends(get_workspace_mcp_server_service)]


def build_unavailable_model_reporter(config: GatewayConfig) -> UnavailableModelReporter:
    """Build the app's reporter of models their providers refuse to serve, with the health checks listening."""
    return UnavailableModelReporter(UnavailableModelListener(lambda uow: build_health_check_service(uow, config)))


def get_unavailable_model_reporter(request: Request) -> UnavailableModelReporter:
    """Return the reporter the app was built with."""
    reporter: UnavailableModelReporter = request.app.state.unavailable_model_reporter
    return reporter


def build_invitation_delivery(uow: UnitOfWork, config: GatewayConfig) -> InvitationDelivery:
    """Build the sender of due invitation emails on a worker job's Unit of Work."""
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return InvitationDelivery(
        uow,
        OrganizationRepositories.on(uow),
        rule=DeliveryRule(domains, EmailBlocklist(uow, BlockedEmailRepository(uow))),
        facts=InvitationFactsReader(
            domains=domains,
            organizations=OrganizationRepository(uow),
            users=UserRepository(uow),
            workspaces=WorkspaceRepository(uow),
        ),
        mailer=Mailer(config),
    )


def build_invitation_cleanup(uow: UnitOfWork) -> ExpiredInvitationCleanup:
    """Build the remover of expired invitations on a worker job's Unit of Work."""
    return ExpiredInvitationCleanup(uow, OrganizationRepositories.on(uow))


def build_update_check_service(uow: UnitOfWork, config: GatewayConfig) -> UpdateCheckService:
    """Build the update check service on a request's Unit of Work, or the worker's own."""
    return UpdateCheckService(uow, UpdateCheckStateRepository(uow), config)


def get_update_check_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> UpdateCheckService:
    """The request's service. The poll builds its own through ``build_update_check_service``."""
    return build_update_check_service(uow, config)


UpdateCheckServiceDep = Annotated[UpdateCheckService, Depends(get_update_check_service)]


def get_request_prices(db: AsyncSession, organization_id: uuid.UUID | None) -> RequestPrices:
    """Build the request pricing cache with its batched pricing repository."""
    return RequestPrices(organization_id, OrganizationModelPricingRepository(db))


def get_automatic_router(request: Request) -> AutomaticRouter:
    """Build one automatic router per application worker."""
    app_state = getattr(request.scope.get("app"), "state", None)
    router = getattr(app_state, "automatic_router", None)
    if router is None:
        router = AutomaticRouter()
        if app_state is not None:
            app_state.automatic_router = router
    return router


def get_usage_service(uow: UnitOfWorkDep) -> UsageService:
    """Build the usage service on the request's Unit of Work; its labels come from the key and user domains."""
    return UsageService(uow, UsageLogRepository(uow), keys=get_api_key_service(uow), users=get_user_access(uow))


UsageServiceDep = Annotated[UsageService, Depends(get_usage_service)]


class _BatchAccountingClaim:
    """Claims a recorded batch's one-time results accounting on the request's session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def claim_accounting(self, batch_id: str) -> bool:
        return await claim_batch_accounting(self._db, batch_id)


def get_batch_results_service(db: PortSessionDep, uow: UnitOfWorkDep) -> BatchResultsService:
    """Build the batch-results accounting service on the request's Unit of Work.

    The claim, the usage rows and the spend fold land on the request's session,
    in one commit, as a video job's settlement does.
    """
    return BatchResultsService(
        uow,
        _BatchAccountingClaim(db),
        UsageLogRepository(uow),
        fold_spend=partial(record_external_spend, db),
    )


BatchResultsServiceDep = Annotated[BatchResultsService, Depends(get_batch_results_service)]


def get_routing_service(
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    organizations: OrganizationServiceDep,
) -> RoutingService:
    """Build the routing service, and the usage and key services it reads through, on the request's Unit of Work."""
    return RoutingService(
        config, get_usage_service(uow), get_api_key_service(uow), uow=uow, organizations=organizations
    )


RoutingServiceDep = Annotated[RoutingService, Depends(get_routing_service)]


WorkspaceWebSearchPoliciesDep = Annotated[WorkspaceWebSearchPolicies, Depends(get_workspace_web_search_policies)]
WebProviderClientDep = Annotated[httpx.AsyncClient, Depends(get_web_provider_client)]
OptionalWebProviderClientDep = Annotated[httpx.AsyncClient | None, Depends(get_optional_web_provider_client)]


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


def require_capability(capability: str) -> Callable[[EntitlementPort], Awaitable[None]]:
    """Build a dependency that refuses a request unless the deployment is entitled.

    The gate every contributed router is mounted behind. A refusal carries the
    same status and body as a request for a path nothing serves (404 with the
    framework's wording), so the response alone does not reveal whether the
    surface exists. The surface itself stays mounted: it appears in the OpenAPI
    document, and a request with an unsupported method meets the router's 405
    before this gate. The refusal is logged, so an operator can tell "not
    entitled" from "not mounted" where the client cannot.
    """

    async def _require(entitlements: EntitlementPortDep) -> None:
        if capability not in await entitlements.entitlements():
            logger.warning("Refused a request to a surface gated on capability %r: not entitled", capability)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    return _require


def _video_rate_lookup(db: AsyncSession) -> VideoRateLookup:
    """The rate a video job settles at, read on the request's session."""

    async def rate(
        provider: str, model: str, *, organization_id: uuid.UUID | None, deployment_paid: bool
    ) -> ResolvedPricing | None:
        return await resolve_model_pricing(
            db, provider, model, organization_id=organization_id, deployment_paid=deployment_paid
        )

    return rate


def get_video_job_service(request: Request, uow: UnitOfWorkDep, db: PortSessionDep) -> VideoJobService:
    """Build the video job service on the request's Unit of Work.

    Its settlement row and the spend it folds land on the request's session, in
    the commit that claims the job.
    """
    return VideoJobService(
        uow,
        InferenceRepositories.on(uow).video_jobs,
        usage=VideoUsageWriter(uow, get_container(request).resolve(UsageSignalPort, db)),
        fold_spend=partial(record_external_spend, db),
        rate=_video_rate_lookup(db),
    )


VideoJobServiceDep = Annotated[VideoJobService, Depends(get_video_job_service)]


def get_file_store(request: Request) -> FileStoragePort:
    """Return the configured blob store for uploaded files."""
    store: FileStoragePort = request.app.state.file_store
    return store


def get_file_service(
    request: Request,
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> FileService:
    """Build the request's files service on the request's Unit of Work."""
    backends = _file_backends(request)
    if backends is None:
        raise RuntimeError("Files are served only where this build binds a file store")
    return FileService(uow, FileRepositories.on(uow), backends, config, lambda: default_workspace_id(db))


FileServiceDep = Annotated[FileService, Depends(get_file_service)]


async def get_optional_file_service(
    request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
) -> FileService | None:
    """Return the request's files service, or ``None`` where files are unavailable.

    The counterpart of ``get_file_service``, for a completion route.
    """
    backends = _file_backends(request)
    if backends is None:
        return None
    return FileService(uow, FileRepositories.on(uow), backends, config, lambda: default_workspace_id(db))


OptionalFileServiceDep = Annotated[FileService | None, Depends(get_optional_file_service)]


async def get_deployment_trace_scope(
    workspace_ids: CallerWorkspaces,
    organization_id: CallerOrganization,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> TraceScope:
    """The scope of the deployment-operator trace routes: every workspace in the caller's organization.

    The operator reads a session's content where it is their own, where its user shared it with the
    operator, and, as an owner or admin of that organization too, where its user shared it with them.
    A header master key names no person to record a read against, so it reads no content at all.
    """
    if session_identity is None:
        return TraceScope.workspaces(workspace_ids)
    audiences: set[ShareAudience] = {"deployment_operator"}
    membership = await OrganizationMemberRepository(db).get_active_by_organization_and_user(
        organization_id, identity.id
    )
    if membership is not None and membership.role in MANAGEMENT_ROLES:
        audiences.add("organization_admins")
    return TraceScope.workspaces(
        workspace_ids,
        reader=ContentReader(
            user_id=identity.id, email=identity.email or str(identity.id), audiences=frozenset(audiences)
        ),
    )


def get_feedback_service() -> FeedbackService:
    return FeedbackService()


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
            forbidden_user_error=DirectToolCallError(USER_MISMATCH_DETAIL, status_code=status.HTTP_403_FORBIDDEN),
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


def get_oauth_app_service(
    request: Request,
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    key_format: ApiKeyFormatPortDep,
    organizations: OrganizationServiceDep,
    workspaces: WorkspaceServiceDep,
) -> OAuthAppService:
    """Build app authorization using the external authorization API origin as issuer."""
    return OAuthAppService(
        uow,
        OAuthAppGrantRepository(uow),
        get_api_key_service(uow),
        key_format,
        organizations,
        workspaces,
        apps=config.oauth_apps,
        ui_link=config.ui_link,
        issuer=str(request.base_url.replace(path="", query="", fragment="")).rstrip("/"),
    )


OAuthAppServiceDep = Annotated[OAuthAppService, Depends(get_oauth_app_service)]


__all__ = [
    "AUTH_FAILURES",
    "AgentModelRecommenderPortDep",
    "AgentRecommenderServiceDep",
    "ApiKeyFormatPortDep",
    "ApiKeyServiceDep",
    "BatchResultsServiceDep",
    "BillingPortDep",
    "BudgetServiceDep",
    "CallerOrganization",
    "CallerUsageScope",
    "CallerWorkspaceQuery",
    "CallerWorkspaces",
    "CallingKey",
    "CodeExecutionPortDep",
    "ContainerDep",
    "CurrentIdentity",
    "DatabaseSessionOwners",
    "DirectToolServiceDep",
    "EntitlementPortDep",
    "FederationTokenExchangeDep",
    "FileServiceDep",
    "GrowthSignalPortDep",
    "HealthCheckServiceDep",
    "HostedProviderServiceDep",
    "IdentityProviderPortDep",
    "IdentityServiceDep",
    "InferenceServiceDep",
    "MaintenanceModeDep",
    "McpCatalogServiceDep",
    "McpServerRequestServiceDep",
    "MembershipListenerDep",
    "ModelMetadataOverrideServiceDep",
    "ModelProviderPortDep",
    "NoSessionOwners",
    "OAuthAppServiceDep",
    "OIDCConnectionServiceDep",
    "OperatesDeployment",
    "OptionalFileServiceDep",
    "OrgProviderModelServiceDep",
    "OrganizationBreakGlass",
    "OverviewServiceDep",
    "PersonGrowthSignalPortDep",
    "PlatformBreakGlass",
    "PlatformConnectServiceDep",
    "PortSessionDep",
    "RateLimitServiceDep",
    "RequestSessionBillingPortDep",
    "RequestSessionModelProviderPortDep",
    "RoutingServiceDep",
    "SERVICE_KEY_REQUIRED_DETAIL",
    "SearchCatalogCallerDep",
    "SearchToolServiceDep",
    "SearchToolTesterDep",
    "ServiceKey",
    "SignupInvitePortDep",
    "TRACE_CONTENT_HEADER_INVALID_DETAIL",
    "TelemetryStoragePortDep",
    "ToolPorts",
    "ToolPortsDep",
    "TraceServiceDep",
    "UnitOfWorkDep",
    "UpdateCheckServiceDep",
    "UsageServiceDep",
    "VisibleWorkspaces",
    "VisibleWorkspacesDep",
    "WebProviderClientDep",
    "WorkspaceMcpServersDep",
    "WorkspaceWebSearchPoliciesDep",
    "asks_to_keep_trace_content",
    "billing_port_on",
    "build_budget_service",
    "build_federation_token_exchange",
    "build_health_check_service",
    "build_invitation_cleanup",
    "build_invitation_delivery",
    "build_mcp_catalog_service",
    "build_trace_service",
    "build_unavailable_model_reporter",
    "build_update_check_service",
    "catalog_reader_operates_deployment",
    "catalog_reader_organization",
    "cookie_may_authenticate",
    "extract_credential_token",
    "get_agent_model_recommender_port",
    "get_api_key_format_port",
    "get_api_key_service",
    "get_billing_port",
    "get_budget_service",
    "get_code_execution_port",
    "get_config",
    "get_container",
    "get_current_identity",
    "get_db",
    "get_direct_tool_service",
    "get_enabled_features",
    "get_entitlement_port",
    "get_federation_token_exchange",
    "get_feedback_delivery_port",
    "get_feedback_service",
    "get_file_store",
    "get_growth_signal_port",
    "get_identity_provider_port",
    "get_log_writer",
    "get_membership_listener",
    "get_model_provider_port",
    "get_oauth_app_service",
    "get_organization_break_glass",
    "get_organization_trace_scope",
    "get_person_growth_signal_port",
    "get_platform_break_glass",
    "get_platform_client",
    "get_request_session_billing_port",
    "get_request_session_model_provider_port",
    "get_session_identity",
    "get_signup_invite_port",
    "get_telemetry_storage_port",
    "get_trace_service",
    "get_unavailable_model_reporter",
    "get_unit_of_work",
    "get_user_access",
    "get_visible_workspaces",
    "is_valid_master_key",
    "is_workspace_member",
    "misdirected_key_detail",
    "model_provider_port_on",
    "record_auth_failure",
    "refuse_ended_session",
    "require_capability",
    "require_deployment_operator",
    "require_deployment_operator_or_absent",
    "require_signed_in_reader",
    "reset_config",
    "routed_trace_content",
    "set_config",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_api_key_or_master_key_for_upload",
    "verify_catalog_reader",
    "verify_catalog_reader_or_public",
    "verify_master_key",
    "verify_service_key",
    "workspace_keeps_trace_content",
    "workspace_members",
    "workspaces_keeping_content_for",
]
