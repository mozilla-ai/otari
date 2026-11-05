"""Direct code execution: run code in the sandbox without a model.

``otari_code_execution`` runs a model's code mid-completion. This endpoint runs
the caller's, as ``POST /api/v1/search`` runs a query, under the same
admission: the workspace's code execution policy (the switch, the timeout
ceiling, the pinned image), the held-container rules, the
``otari:code_execution`` rate and the budgets.

Every run that reached the sandbox is billed, whatever the code did: the
sandbox ran it either way, and a caller here, unlike a model in the tool loop,
asked for that run and reads its exit status itself.

Files move through the Files API, as on the completion routes: ``file_ids``
names uploads to put in the sandbox's working directory before the code runs,
and each file the run writes is stored as a new file and returned with its
``file_id``.

The usage row is logged as ``model="code_execution"`` and ``provider="otari"``,
the halves of the pricing key, under ``endpoint="/v1/code/execute"``, a frozen
label rather than a path. A request the gateway refuses writes a row too,
through the shared :func:`gateway.api.routes._pipeline.log_gateway_rejection`.
"""

import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    BillingPortDep,
    CodeExecutionPortDep,
    OptionalFileServiceDep,
    UnitOfWorkDep,
    WorkspaceCodeExecutionPoliciesDep,
    build_sandbox_container_registry,
    build_sandbox_file_bridge,
    get_budget_service,
    get_config,
    get_db,
    get_log_writer,
    verify_api_key_or_master_key,
)
from gateway.api.request_id import set_request_id
from gateway.api.routes._passthrough import resolve_passthrough_user_id
from gateway.api.routes._pipeline import (
    SANDBOX_UNAVAILABLE_DETAIL,
    SANDBOX_UNREACHABLE_DETAIL,
    _elapsed_ms,
    log_gateway_rejection,
)
from gateway.core.config import END_USER_BUDGET_HEADER, GatewayConfig
from gateway.core.database import release_session
from gateway.core.error_codes import INSUFFICIENT_FUNDS, PRICING_REQUIRED, error_headers
from gateway.core.metered_pricing import quantize_cost
from gateway.exceptions import TenancyError
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.usage import UsageLog
from gateway.ports.billing_port import FUNDS_CHECK_UNAVAILABLE_DETAIL, InsufficientFundsError
from gateway.ports.code_execution_port import (
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.rate_limit import admit_rate_limit_rules, check_rate_limit, rate_limit_headers
from gateway.schemas.tools import CodeExecuteRequest, CodeExecuteResponse, CodeExecutionOutputFile
from gateway.schemas.usage_and_telemetry import TOOL_METER_NAMESPACE
from gateway.services.api_keys import names_end_user
from gateway.services.budgets import (
    BudgetScopeRequest,
    BudgetService,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.code_execution import ContainerLease
from gateway.services.files import FileScope, StagedFile, sandbox_path_for
from gateway.services.inference import DeploymentCharge, admit_deployment_paid, describe_port_error
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import resolve_tool_rate
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, DEFAULT_EXEC_TIMEOUT_S, SandboxBackend
from gateway.services.tools import CONTAINER_GONE_DETAIL_TEMPLATE, declare_code_execution, unpriced_tool_detail
from gateway.services.workspace_scope import organization_for_key_id, workspace_for_key_id
from gateway.types.tool_charges import GATEWAY_TOOL_PRICING_PROVIDER, gateway_tool_pricing_key

router = APIRouter(prefix="/code", tags=["code-execution"])

CODE_EXECUTE_ENDPOINT = "/v1/code/execute"
SANDBOX_NOT_CONFIGURED_DETAIL = "code execution is not available: no sandbox is configured on this gateway"
FILES_NOT_ENABLED_DETAIL = "file_ids needs the Files API, which this gateway has turned off"
FILE_NOT_FOUND_DETAIL_TEMPLATE = "File '{file_id}' was not found"
# A direct run is admitted as this declaration, so the workspace's policy reads it as the in-loop tool's.
_DECLARATION: list[dict[str, Any]] = [{"type": "otari_code_execution"}]
_PRICING_KEY = gateway_tool_pricing_key(CODE_EXECUTION_TOOL_NAME)


@router.post("/execute")
async def execute_code(
    raw_request: Request,
    response: Response,
    request: CodeExecuteRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
    policies: WorkspaceCodeExecutionPoliciesDep,
    port: CodeExecutionPortDep,
    files: OptionalFileServiceDep,
    billing: BillingPortDep,
) -> CodeExecuteResponse:
    """Run code in the gateway's sandbox and return its output.

    The run is admitted as an ``otari_code_execution`` call is: a workspace whose
    code execution policy is off is refused, and its execution ceiling and pinned
    image apply. It is priced under ``otari:code_execution`` and reserved against
    the caller's budgets. ``file_ids`` puts uploads in the working directory, and
    each file the code writes comes back with a ``file_id``. ``container: "auto"``
    holds the sandbox after the run and names it in ``Otari-Container-Id``;
    sending that id back runs in the same sandbox, with its files and
    interpreter state.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend bind to the key's own user. A ``user`` field
      naming a different user is rejected with 403 (or ignored, when the key's
      own ``reject_user_mismatch`` is false, or the deployment-wide setting is
      disabled and the key does not override it); it is never billed to that
      user.
    - Service key: a ``user`` field names one of the key owner's end users,
      created on first use on the budget ``Otari-End-User-Budget`` names (or
      the key's default), and is billed and rate limited as that end user.
    """
    started_at = time.monotonic()
    request_id = str(uuid.uuid4())
    set_request_id(raw_request, response, request_id)
    api_key, _ = auth_result
    api_key_id = api_key.id if api_key else None
    budget_exempt = api_key is not None and api_key.exclude_from_budget

    if api_key is not None and request.user and names_end_user(api_key, request.user):
        rate_limit_info = await check_rate_limit(raw_request, str(api_key.user_id))
        end_user = await budget_service.resolve_end_user(
            api_key=api_key,
            external_id=request.user,
            requested_budget_id=raw_request.headers.get(END_USER_BUDGET_HEADER) or None,
        )
        user_id = end_user.user_id
        if end_user.budget_id is not None:
            response.headers[END_USER_BUDGET_HEADER] = end_user.budget_id
    else:
        user_id = resolve_passthrough_user_id(auth_result, request.user, reject_mismatch=config.reject_user_mismatch)
        rate_limit_info = await check_rate_limit(raw_request, user_id)

    async def refuse(detail: str, status_code: int, headers: dict[str, str] | None = None) -> HTTPException:
        """Log a run the gateway turned away, and return the error to raise."""
        await log_gateway_rejection(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=CODE_EXECUTION_TOOL_NAME,
            provider=GATEWAY_TOOL_PRICING_PROVIDER,
            endpoint=CODE_EXECUTE_ENDPOINT,
            detail=detail,
            status_code=status_code,
            started_at=started_at,
            request_id=request_id,
        )
        return HTTPException(status_code=status_code, detail=detail, headers=headers)

    if port is None:
        raise await refuse(SANDBOX_NOT_CONFIGURED_DETAIL, status.HTTP_400_BAD_REQUEST)

    workspace_id = await workspace_for_key_id(db, api_key_id)
    file_bridge = build_sandbox_file_bridge(
        raw_request=raw_request,
        config=config,
        uow=uow,
        user_id=user_id,
        workspace_id=workspace_id,
        inputs=[],
    )
    if request.file_ids:
        if file_bridge is None or files is None:
            raise await refuse(FILES_NOT_ENABLED_DETAIL, status.HTTP_400_BAD_REQUEST)
        # The master key acts deployment-wide, so it reaches an upload in any workspace.
        scope = FileScope(user_id, None if api_key is None else workspace_id)
        for file_id in dict.fromkeys(request.file_ids):
            staged = await files.staged_upload(file_id, scope)
            if staged is None:
                raise await refuse(FILE_NOT_FOUND_DETAIL_TEMPLATE.format(file_id=file_id), status.HTTP_400_BAD_REQUEST)
            file_bridge.inputs.append(_named_for_the_sandbox(staged, file_bridge.inputs))

    # Admitted before any reservation, so a refused run costs nothing to be told about.
    try:
        declared = await declare_code_execution(
            _DECLARATION,
            container_id=request.container,
            code_execution_header=None,
            config=config,
            resolve_policy=lambda: policies.for_request(workspace_id),
            container_registry=build_sandbox_container_registry(
                config=config, uow=uow, user_id=user_id, workspace_id=workspace_id, port=port
            ),
        )
        admitted = await declared.admit(claimed=False, beside_sandbox=False)
    except TenancyError as exc:
        raise await refuse(exc.message, exc.status_code) from exc

    claimed = admitted.container_lease
    containers = admitted.containers

    async def give_back_claim() -> None:
        """Hand back a held container this run claimed and will now never enter."""
        if claimed is None or containers is None:
            return
        try:
            await containers.release(claimed.container_id)
        except Exception:  # noqa: BLE001 - the claim expires on its own either way
            logger.warning("could not release container claim %s", claimed.container_id, exc_info=True)

    try:
        # The sandbox is the deployment's, so the run is deployment-paid and only the operator's rate counts.
        organization_id = await organization_for_key_id(db, api_key_id)
        rate = await resolve_tool_rate(db, (_PRICING_KEY,), deployment_paid=True, organization_id=organization_id)
        unit_rate = rate.unit_rate if rate is not None else Decimal(0)
        rate_limit_grant = await admit_rate_limit_rules(
            raw_request,
            key_id=api_key_id,
            user_id=user_id,
            estimated_tokens=0,
            budget_limits=None
            if budget_exempt
            else await budget_service.minute_limits(user_id, strategy=config.budget_strategy),
        )
        if rate_limit_grant is not None:
            await rate_limit_grant.admit_model(GATEWAY_TOOL_PRICING_PROVIDER, CODE_EXECUTION_TOOL_NAME)
        reservation = await reserve_budget(
            db,
            user_id,
            unit_rate,
            # Not a model: ``model`` only drives reserve_budget's free-model shortcut through any-llm.
            model=None,
            strategy=config.budget_strategy,
            counts_toward_budget=not budget_exempt,
            scope=BudgetScopeRequest(api_key=api_key),
        )
    except BaseException:
        await give_back_claim()
        raise

    async def refused_after_reservation(
        detail: str, status_code: int, headers: dict[str, str] | None = None
    ) -> HTTPException:
        await refund_reservation(db, reservation)
        await give_back_claim()
        return await refuse(detail, status_code, headers)

    # After the budget checks, so a missing or exhausted budget answers first.
    if rate is None and config.require_pricing and not budget_exempt:
        raise await refused_after_reservation(
            unpriced_tool_detail(CODE_EXECUTION_TOOL_NAME),
            status.HTTP_402_PAYMENT_REQUIRED,
            error_headers(PRICING_REQUIRED),
        )
    funding: DeploymentCharge | None = None
    try:
        funding = await admit_deployment_paid(
            billing,
            deployment_paid=True,
            organization_id=organization_id,
            api_key_id=api_key_id,
            user_id=user_id,
            request_id=request_id,
        )
    except InsufficientFundsError as exc:
        raise await refused_after_reservation(
            exc.public_message, status.HTTP_402_PAYMENT_REQUIRED, error_headers(INSUFFICIENT_FUNDS)
        ) from exc
    except Exception as exc:
        logger.error(
            "BillingPort failed to admit a direct code execution: organization=%s request_id=%s: %s",
            organization_id,
            request_id,
            describe_port_error(exc),
        )
        # Ends the transaction the port may have left failed, so the refund has a usable session.
        await release_session(db)
        raise await refused_after_reservation(
            FUNDS_CHECK_UNAVAILABLE_DETAIL, status.HTTP_503_SERVICE_UNAVAILABLE
        ) from exc

    def usage_row(**outcome: Any) -> UsageLog:
        return UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=CODE_EXECUTION_TOOL_NAME,
            provider=GATEWAY_TOOL_PRICING_PROVIDER,
            endpoint=CODE_EXECUTE_ENDPOINT,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=not budget_exempt,
            deployment_paid=funding is not None,
            request_group_id=request_id,
            **outcome,
        )

    track_request(
        raw_request,
        endpoint=CODE_EXECUTE_ENDPOINT,
        model=CODE_EXECUTION_TOOL_NAME,
        provider=GATEWAY_TOOL_PRICING_PROVIDER,
        user_id=user_id,
        api_key_id=api_key_id,
    )
    # Frees the request's connection before the sandbox runs, and lands any credit the funding gate granted.
    if not await release_session(db) and funding is not None:
        logger.error("Could not commit a deployment-paid code execution admission: request_id=%s", request_id)
        raise await refused_after_reservation(FUNDS_CHECK_UNAVAILABLE_DETAIL, status.HTTP_503_SERVICE_UNAVAILABLE)

    backend = SandboxBackend(
        port=port,
        timeout_s=min(
            DEFAULT_EXEC_TIMEOUT_S,
            float(admitted.exec_timeout_s or DEFAULT_EXEC_TIMEOUT_S),
            float(request.timeout_seconds or DEFAULT_EXEC_TIMEOUT_S),
        ),
        image=admitted.session_image,
        allowed_tools=admitted.allowed_tools,
        files=file_bridge,
        files_base_url=file_bridge.base_url if file_bridge is not None else None,
        container=claimed,
        containers=containers,
    )
    try:
        async with backend:
            await backend.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": request.code})
            execution = backend.take_executions()[-1]
    except SandboxNotReachableError as exc:
        error = _sandbox_failure(exc, claimed)
        await log_writer.put(
            usage_row(
                status="error",
                error_message=type(exc).__name__,
                status_code=error.status_code,
                billing_meters=_tool_meters(billed=0, errors=1),
            )
        )
        await refund_reservation(db, reservation)
        if isinstance(exc, SandboxSessionGoneError) and claimed is not None and containers is not None:
            try:
                await containers.forget(claimed.container_id)
            except Exception:  # noqa: BLE001 - the next resume refuses it anyway
                logger.warning("could not drop gone container %s", claimed.container_id, exc_info=True)
        else:
            # A session that never opened never reached the backend's exit, which is what gives a claim back.
            await give_back_claim()
        logger.error("Direct code execution failed on %s: %s", port.label, exc)
        raise error from exc
    except BaseException:
        await refund_reservation(db, reservation)
        await give_back_claim()
        raise

    assert execution.result is not None  # a call that returned produced a result block
    cost = quantize_cost(unit_rate)
    await log_writer.put(
        usage_row(
            status="success",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=cost,
            billing_meters=_tool_meters(billed=1, errors=0, unit_rate=unit_rate if rate is not None else None),
        )
    )
    if funding is not None:
        await funding.settle(
            cost,
            model=CODE_EXECUTION_TOOL_NAME,
            description=_PRICING_KEY,
            expects_cost=False,
            savepoint=db.begin_nested,
        )
    await reconcile_reservation(db, reservation, cost)

    if backend.lease is not None:
        response.headers["Otari-Container-Id"] = backend.lease.container_id
        response.headers["Otari-Container-Expires-At"] = backend.lease.expires_at.isoformat()
    if rate_limit_info:
        for header, value in rate_limit_headers(rate_limit_info).items():
            response.headers[header] = value
    content = execution.result.content
    return CodeExecuteResponse(
        stdout=content.stdout,
        stderr=content.stderr,
        return_code=content.return_code,
        files=[
            CodeExecutionOutputFile(file_id=file_id, filename=filename)
            for filename, file_id in execution.file_ids.items()
        ],
    )


def _tool_meters(*, billed: int, errors: int, unit_rate: Decimal | None = None) -> dict[str, Any]:
    """The run as the tool loop records its calls, so the usage views' tool filter and breakdown count it."""
    meter: dict[str, float] = {"billed": billed, "errors": errors}
    if unit_rate is not None:
        meter["unit_rate"] = float(unit_rate)
    return {TOOL_METER_NAMESPACE: {CODE_EXECUTION_TOOL_NAME: meter}}


def _named_for_the_sandbox(staged: StagedFile, taken: list[StagedFile]) -> StagedFile:
    """``staged`` under the name it gets in the working directory, unique among ``taken``."""
    return replace(staged, filename=sandbox_path_for(staged.filename, {existing.filename for existing in taken}))


def _sandbox_failure(exc: SandboxNotReachableError, claimed: ContainerLease | None) -> HTTPException:
    """The answer to a run the sandbox could not take, worded as the completion routes word it."""
    if isinstance(exc, SandboxSessionGoneError) and claimed is not None:
        # The caller's id is stale, which is a request to fix rather than an outage to retry.
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=CONTAINER_GONE_DETAIL_TEMPLATE.format(container_id=claimed.container_id),
        )
    if isinstance(exc, SandboxUnavailableError):
        headers = {"Retry-After": exc.retry_after} if exc.retry_after is not None else None
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SANDBOX_UNAVAILABLE_DETAIL, headers=headers
        )
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=SANDBOX_UNREACHABLE_DETAIL)
