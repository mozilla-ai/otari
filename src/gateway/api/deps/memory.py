"""How a request reaches agent memory: its service, its caller, and what memory a request through the gateway gets."""

import asyncio
import secrets
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import ContainerDep, get_config, get_container, get_unit_of_work
from gateway.api.deps.auth import extract_credential_token, verify_api_key
from gateway.api.deps.traces import _person, is_workspace_member
from gateway.container import Container
from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, create_session, describe_database_error, get_db
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions import TenancyError
from gateway.exceptions.memory_exceptions import MEMORY_OFF, MemoryToolsRefusedError
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import MANAGEMENT_ROLES
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.content_key_port import ContentKeyPort
from gateway.ports.memory_ranking_port import MemoryRankingPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.repositories.memory import MemoryRepositories
from gateway.repositories.tenancy import OrganizationMemberRepository, WorkspaceMemberRepository, WorkspaceRepository
from gateway.repositories.usage_and_telemetry import UsageLogRepository
from gateway.services.guardrails import GuardrailsNotReachableError, deployment_guardrails_url
from gateway.services.memory import (
    MEMORY_ENDPOINT_PREFIX,
    ExperimentArm,
    GuardrailOutcome,
    GuardrailsUnavailableError,
    MemoryCaller,
    MemoryCallUsage,
    MemoryModels,
    MemoryService,
    MemoryToolBackend,
    ModelUse,
    RecallVectors,
    SessionUsage,
    ToolOutcome,
    WorkspaceKeys,
    call_memory_tool,
    recall_for_turn,
)
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError
from gateway.services.tenancy import OrganizationService, organization_guardrail_runner
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_guardrail_service import (
    ResolvedOrganizationGuardrail,
    resolve_organization_guardrails,
    run_organization_guardrails,
)


def verify_memory_service_token(request: Request, config: Annotated[GatewayConfig, Depends(get_config)]) -> None:
    """Admit only a gateway of this deployment to the memory service, by ``memory_service_token``.

    Neither an API key nor the master key opens it: the service answers about any workspace, and only a gateway,
    which has already decided what each member may recall, is trusted to ask.
    """
    expected = config.memory_service_token
    token = extract_credential_token(request)
    if not config.memory_engine or not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a gateway of this deployment.")


def memory_keys_for(config: GatewayConfig, keys: ContentKeyPort | None) -> WorkspaceKeys:
    """The workspaces' memory data keys over ``keys``, rotated as the deployment says."""
    return WorkspaceKeys(keys, rotation=timedelta(days=config.memory_key_rotation_days))


class _MemoryUsage:
    """Memory's model calls as usage rows of their workspace, on the service's own Unit of Work."""

    def __init__(self, usage: UsageLogRepository) -> None:
        self._usage = usage

    async def record(self, workspace_id: uuid.UUID, api_key_id: str | None, use: ModelUse) -> None:
        await self._usage.record_side_call(
            workspace_id=workspace_id,
            api_key_id=api_key_id,
            model=use.model,
            provider=use.provider,
            endpoint=use.endpoint,
            succeeded=use.succeeded,
            prompt_tokens=use.prompt_tokens,
            completion_tokens=use.completion_tokens,
            latency_ms=use.latency_ms,
        )


class _MemoryLedger:
    """A workspace's usage rows, as the savings experiment reads them, in the service's own blocks."""

    def __init__(self, uow: UnitOfWork, usage: UsageLogRepository) -> None:
        self._uow = uow
        self._usage = usage

    async def sessions(self, workspace_id: uuid.UUID, since: datetime, until: datetime) -> dict[str, SessionUsage]:
        async with self._uow:
            totals = await self._usage.session_totals(
                workspace_id, since=since, until=until, excluding_endpoints=MEMORY_ENDPOINT_PREFIX
            )
        return {
            row.session: SessionUsage(
                requests=row.requests,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cost=float(row.cost) if row.cost is not None else None,
            )
            for row in totals
        }

    async def memory_calls(
        self, workspace_id: uuid.UUID, since: datetime, until: datetime
    ) -> dict[str, MemoryCallUsage]:
        async with self._uow:
            totals = await self._usage.endpoint_totals(
                workspace_id, since=since, until=until, endpoints=MEMORY_ENDPOINT_PREFIX
            )
        return {row.endpoint: MemoryCallUsage(row.calls, row.input_tokens, row.output_tokens) for row in totals}


class _MemoryMandate:
    """The guardrails an organization mandates for one workspace, run over a proposal as over a request's input."""

    def __init__(
        self, config: GatewayConfig, organization_id: uuid.UUID, mandates: list[ResolvedOrganizationGuardrail]
    ) -> None:
        self._config = config
        self._organization_id = organization_id
        self._mandates = mandates

    async def check(self, text: str) -> GuardrailOutcome:
        try:
            verdict = await run_organization_guardrails(
                self._mandates,
                text,
                organization_id=self._organization_id,
                default_url=deployment_guardrails_url(self._config),
                held=organization_guardrail_runner.handle,
            )
        except GuardrailsNotReachableError as exc:
            # The reason names the profile and the endpoint, for the operator, and never what was checked.
            logger.warning("A mandated guardrail could not check a memory proposal: %s", exc)
            raise GuardrailsUnavailableError from None
        return GuardrailOutcome(
            blocked=tuple(result.profile for result in verdict.flagged if result.mode == "block"),
            monitored=tuple(result.profile for result in verdict.flagged if result.mode != "block"),
            unchecked=tuple(result.profile for result in verdict.results if result.status != "evaluated"),
        )


class _MemoryGuardrails:
    """The guardrails an organization mandates for a workspace, read as the request path reads them."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def mandated(self, organization_id: uuid.UUID, workspace_id: uuid.UUID) -> _MemoryMandate | None:
        try:
            # On a session of its own, closed before any check runs, so no transaction idles over one.
            async with create_session() as session:
                mandates = await resolve_organization_guardrails(
                    session, organization_id=organization_id, workspace_id=workspace_id, for_memory=True
                )
        except (SecretBoxUnavailableError, SecretDecryptionError) as exc:
            # As on a request, a mandate whose credential will not open is not run unauthenticated.
            logger.error(
                "Organization guardrail credential could not be decrypted for organization %s: %s",
                organization_id,
                exc,
            )
            raise GuardrailsUnavailableError from None
        return _MemoryMandate(self._config, organization_id, mandates) if mandates else None


class _MemoryManagers:
    """The management checks every workspace and organization policy uses, for memory's switches."""

    def __init__(self, organizations: OrganizationService, access: WorkspaceAccess) -> None:
        self._organizations = organizations
        self._access = access

    async def require_manager(self, user: TenancyUser, workspace_id: uuid.UUID) -> uuid.UUID:
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)
        return workspace.organization_id

    async def require_organization_manager(self, user: TenancyUser) -> uuid.UUID:
        organization = await self._organizations.get_active_organization_for_user(user)
        await self._organizations.require_active_organization_management_access(user=user, organization=organization)
        return organization.id


def build_memory_service(
    uow: UnitOfWork,
    config: GatewayConfig,
    keys: WorkspaceKeys,
    container: Container,
    db: AsyncSession | None = None,
    *,
    recall_vectors: RecallVectors | None = None,
    ranking: MemoryRankingPort | None = None,
) -> MemoryService:
    """Build the memory service on a request's Unit of Work, or the consolidator's own.

    ``container`` serves the hosted providers memory's models may run on, read on the same Unit of Work.
    ``db`` is the request's session, for the management checks; the consolidator has no caller to check.
    ``recall_vectors`` is the process's opened embeddings, for automatic recall and erasing what it holds.
    ``ranking`` is the memory service recall and search rank on; None ranks in this process.
    """
    managers = None
    if db is not None:
        organizations = OrganizationService(db, membership_listener=None)
        managers = _MemoryManagers(organizations, WorkspaceAccess(db, organizations))
    return MemoryService(
        uow,
        MemoryRepositories.on(uow),
        keys,
        MemoryModels(config, container.resolve(ModelProviderPort, db, uow=uow)),
        _MemoryUsage(UsageLogRepository(uow)),
        config,
        managers,
        ledger=_MemoryLedger(uow, UsageLogRepository(uow)),
        recall_vectors=recall_vectors,
        ranking=ranking,
        guardrails=_MemoryGuardrails(config),
    )


def get_memory_keys(request: Request) -> WorkspaceKeys:
    """This app's memory data keys, built on first use and held for every request, so opened keys are shared."""
    keys: WorkspaceKeys | None = getattr(request.app.state, "memory_keys", None)
    if keys is None:
        config: GatewayConfig = request.app.state.config
        keys = memory_keys_for(config, get_container(request).resolve(ContentKeyPort, None))
        request.app.state.memory_keys = keys
    return keys


def get_recall_vectors(request: Request) -> RecallVectors | None:
    """This app's opened recall embeddings, held for every request; None where the deployment holds none.

    A gateway that hands ranking to a memory service holds none: the service does.
    """
    config: GatewayConfig = request.app.state.config
    if not config.memory_recall_cache or config.memory_service_url:
        return None
    held: RecallVectors | None = getattr(request.app.state, "memory_recall_vectors", None)
    if held is None:
        held = RecallVectors()
        request.app.state.memory_recall_vectors = held
    return held


def get_memory_ranking(request: Request) -> MemoryRankingPort | None:
    """The memory service this app hands recall's ranking to; None where it ranks in its own process."""
    config: GatewayConfig = request.app.state.config
    if not config.memory_service_url:
        return None
    ranking: MemoryRankingPort = get_container(request).resolve(MemoryRankingPort, None)
    return ranking


def get_memory_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    keys: Annotated[WorkspaceKeys, Depends(get_memory_keys)],
    container: ContainerDep,
    recall_vectors: Annotated[RecallVectors | None, Depends(get_recall_vectors)],
    ranking: Annotated[MemoryRankingPort | None, Depends(get_memory_ranking)],
) -> MemoryService:
    """The request's memory service. The consolidator builds its own through ``build_memory_service``."""
    return build_memory_service(uow, config, keys, container, db, recall_vectors=recall_vectors, ranking=ranking)


NOT_A_MEMBER = "Memory is for workspace members, and this key names none of its workspace."


async def memory_caller_for(
    db: AsyncSession, api_key: APIKey, *, session_id: str | None = None, model: str | None = None
) -> MemoryCaller | None:
    """The member a key speaks for, in the key's own workspace; None for a key that names no active member.

    A service key, or a billing user no person stands behind, names none. Memory never takes the workspace
    or the member from a request body.
    """
    workspace = await WorkspaceRepository(db).get(api_key.workspace_id)
    person = _person(api_key.user_id)
    if (
        workspace is None
        or person is None
        or not await is_workspace_member(db, api_key.workspace_id, api_key.user_id, workspace)
    ):
        return None
    membership = await OrganizationMemberRepository(db).get_active_by_organization_and_user(
        workspace.organization_id, person
    )
    seat = await WorkspaceMemberRepository(db).get_by_workspace_and_user(api_key.workspace_id, person)
    organization_manager = membership is not None and membership.role in MANAGEMENT_ROLES
    workspace_role = seat.role if seat is not None and seat.status == "active" else None
    return MemoryCaller(
        workspace_id=api_key.workspace_id,
        member_id=person,
        is_manager=organization_manager or workspace_role in MANAGEMENT_ROLES,
        # A workspace viewer recalls what the workspace knows and writes nothing to it.
        can_write=organization_manager or workspace_role != "viewer",
        api_key_id=api_key.id,
        session_id=session_id[:255] if session_id else None,
        model=model[:255] if model else None,
    )


async def get_memory_caller(
    request: Request,
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MemoryCaller:
    """The member a key speaks for on the MCP endpoint.

    Raises:
        HTTPException: 403 when the key names no active member of its workspace.
    """
    session_id = request.headers.get("Otari-Session-Id") or request.headers.get("Mcp-Session-Id")
    caller = await memory_caller_for(db, api_key, session_id=session_id)
    if caller is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_A_MEMBER)
    return caller


# Longest automatic recall may hold a request up, embedding included.
MEMORY_RECALL_BUDGET_S = 3.0


@dataclass(frozen=True)
class AdmittedMemory:
    """What memory one request through the gateway gets.

    ``tools`` are memory's tools for its loop, ``recall`` what memory holds on its newest turn, given that turn
    and the user's turns before it (the text to add, and how many entries), and ``arm`` the experiment arm of
    its session.
    """

    tools: MemoryToolBackend | None
    recall: Callable[[str, str | None], Awaitable[tuple[str | None, int]]] | None
    arm: ExperimentArm | None


class MemoryToolAdmission:
    """Decides whether a request through the gateway gets memory's tools, and builds them for its caller.

    A request asks for them by declaring ``{"type": "otari_memory"}`` or sending ``Otari-Memory: on``, and
    gets them unasked where its workspace offers memory to every request, unless it sends ``Otari-Memory: off``.
    Asking where memory cannot be had is refused; not asking never is.
    """

    def __init__(
        self,
        config: GatewayConfig,
        keys: WorkspaceKeys,
        container: Container,
        recall_vectors: RecallVectors | None = None,
        ranking: MemoryRankingPort | None = None,
    ) -> None:
        self._config = config
        self._keys = keys
        self._container = container
        self._recall_vectors = recall_vectors
        self._ranking = ranking

    async def admit(
        self,
        db: AsyncSession,
        api_key: APIKey | None,
        *,
        asked: bool,
        declined: bool,
        session_label: str | None = None,
        model: str | None = None,
    ) -> AdmittedMemory | None:
        """What memory this request gets, or None when it gets none.

        ``session_label`` is the trace id of the request's session: its experiment arm, and the session its
        proposals name. A session in the experiment's control arm gets no memory, even where it asked.

        Raises:
            MemoryToolsRefusedError: the request asked for memory's tools and its caller cannot have them.
        """
        if declined:
            return None
        caller = await memory_caller_for(db, api_key, session_id=session_label, model=model) if api_key else None
        if caller is None:
            if asked:
                raise MemoryToolsRefusedError(NOT_A_MEMBER)
            return None
        async with create_unit_of_work() as uow:
            service = build_memory_service(uow, self._config, self._keys, self._container)
            offer = await service.offer(caller, session_label)
        if not offer.on:
            if asked:
                raise MemoryToolsRefusedError(MEMORY_OFF)
            return None
        if offer.arm == "control":
            return AdmittedMemory(tools=None, recall=None, arm="control")
        tools = MemoryToolBackend(caller, self._run) if asked or offer.every_request else None
        recall = partial(self._recall, caller) if offer.auto_recall else None
        if tools is None and recall is None and offer.arm is None:
            return None
        return AdmittedMemory(tools=tools, recall=recall, arm=offer.arm)

    async def _run(self, caller: MemoryCaller, name: str, arguments: dict[str, Any]) -> ToolOutcome | None:
        # A Unit of Work per call: the request's own session is released before the provider is called.
        async with create_unit_of_work() as uow:
            service = build_memory_service(
                uow,
                self._config,
                self._keys,
                self._container,
                recall_vectors=self._recall_vectors,
                ranking=self._ranking,
            )
            return await call_memory_tool(service, caller, name, arguments)

    async def _recall(self, caller: MemoryCaller, query: str, context: str | None) -> tuple[str | None, int]:
        # Best effort: a request is served without what memory holds rather than failed or held up for it.
        try:
            async with asyncio.timeout(MEMORY_RECALL_BUDGET_S), create_unit_of_work() as uow:
                service = build_memory_service(
                    uow,
                    self._config,
                    self._keys,
                    self._container,
                    recall_vectors=self._recall_vectors,
                    ranking=self._ranking,
                )
                return await recall_for_turn(service, caller, query, context)
        except TimeoutError:
            logger.warning("Automatic recall took longer than %ss; the request goes without it", MEMORY_RECALL_BUDGET_S)
        except TenancyError as exc:
            logger.info("Automatic recall found nothing to hand over: %s", exc.message)
        except DATABASE_ERRORS as exc:
            logger.warning("Automatic recall failed on the database: %s", describe_database_error(exc))
        return None, 0


def get_memory_tool_admission(request: Request) -> MemoryToolAdmission | None:
    """What admits memory's tools to a request, or None where the deployment does not offer memory."""
    config: GatewayConfig = request.app.state.config
    if not config.memory_enabled:
        return None
    return MemoryToolAdmission(
        config,
        get_memory_keys(request),
        get_container(request),
        get_recall_vectors(request),
        get_memory_ranking(request),
    )


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
MemoryCallerDep = Annotated[MemoryCaller, Depends(get_memory_caller)]
