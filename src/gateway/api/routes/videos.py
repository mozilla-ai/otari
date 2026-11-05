"""OpenAI-shaped video generation: start a job, poll it, fetch its video, and list the caller's jobs.

A video takes minutes, so every provider returns a job first. ``POST /videos``
starts one through any-llm and returns it, ``GET /videos/{video_id}`` reads it
back from the provider that runs it, ``GET /videos/{video_id}/content`` serves
the finished video's bytes, and ``GET /videos`` lists the caller's own jobs
from the gateway's records.

Billing follows the batch pattern. Starting a job runs the pass-through
scaffold (auth, rate limit, model access, the budget reservation, the hosted
credential ladder, the usage row) with the estimate for the length asked for at
the model's per-second video rate, and settles at no cost: the length is known
only when the job completes. The first retrieval that finds the job completed
bills it once through the video job service, which owns the length, the price
and the claim. A model with no per-second video rate is refused before the job
starts.

A completed job is remembered on its record, so a later read answers from it
even where the provider keeps no job to look up, as DeepInfra does.
"""

import uuid
from decimal import Decimal
from typing import Annotated, Any

from any_llm import AnyLLM, LLMProvider, adownload_video, aretrieve_video, avideo_generation
from any_llm.types.video import VideoJob
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from gateway.api.deps import (
    PortSessionDep,
    VideoJobServiceDep,
    billing_port_on,
    get_config,
    get_log_writer,
    model_provider_port_on,
    verify_api_key_or_master_key,
)
from gateway.api.routes._helpers import country_gate_for, database_error
from gateway.api.routes._passthrough import provider_call_failure, resolve_provider_by_name, run_passthrough
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, release_session
from gateway.core.error_codes import PRICING_REQUIRED, error_headers
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.inference import VideoJobRecord
from gateway.models.pricing import ModelPricing
from gateway.schemas.inference import (
    VideoError,
    VideoGenerationRequest,
    VideoJobList,
    VideoJobResponse,
    VideoJobSummary,
)
from gateway.services.inference import VIDEO_BILLING, DeploymentCharge, VideoJobService
from gateway.services.log_writer import LogWriter
from gateway.services.pricing import estimate_video_cost, video_rate
from gateway.services.provider_kwargs import ResolvedProvider
from gateway.services.upstream_redaction import redact_upstream_message
from gateway.services.workspace_scope import organization_for_workspace_id, resolve_workspace_id

router = APIRouter(prefix="/videos", tags=["videos"])

# See chat.USAGE_ENDPOINT. A job writes one row when it starts and one when its
# completion is billed.
USAGE_ENDPOINT = "/v1/videos"
USAGE_ENDPOINT_COMPLETED = "/v1/videos/completed"

# What the estimate assumes when the request names no length: the longest
# default a served provider renders, so the reservation is never short.
ESTIMATED_SECONDS = 8

ConfigDep = Annotated[GatewayConfig, Depends(get_config)]
LogWriterDep = Annotated[LogWriter, Depends(get_log_writer)]


def no_video_price_detail(model: str) -> str:
    """The 402 body for a video model with no per-second price: what is missing and how to set it."""
    return (
        f"Model '{model}' has no per-second video price, so a video cannot be billed. Add a rates block that "
        f"prices video output per second (POST {API_ROOT}/pricing, or the pricing section of config.yml)."
    )


def job_response(job: VideoJob, *, provider: str) -> VideoJobResponse:
    """The job as the caller sees it: what the provider reports, under the instance that runs it.

    A failed job's message is the provider's own, redacted and capped as any upstream message is.
    """
    error = (
        None
        if job.error is None
        else VideoError(message=redact_upstream_message(job.error.message), code=job.error.code)
    )
    return VideoJobResponse(
        id=job.id,
        model=job.model,
        provider=provider,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        progress=job.progress,
        seconds=job.seconds,
        size=job.size,
        error=error,
    )


@router.post("", response_model=VideoJobResponse)
async def create_video(
    raw_request: Request,
    response: Response,
    request: VideoGenerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: PortSessionDep,
    config: ConfigDep,
    log_writer: LogWriterDep,
    jobs: VideoJobServiceDep,
) -> VideoJobResponse:
    """Start a video generation job.

    The job is returned at once. Poll ``GET /api/v1/videos/{video_id}`` until
    its status is ``completed``, then fetch the video from
    ``GET /api/v1/videos/{video_id}/content``. The model must have a per-second
    video price, from the catalog or set through the pricing API; one with none
    is refused with 402. The length asked for is reserved against the budget at
    that price while the job starts, and the job is billed when a retrieval
    finds it completed. A provider that renders inside this call returns the
    job already completed, billed, and ready for its content.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    api_key, _ = auth_result
    estimated_seconds = request.seconds if request.seconds is not None else ESTIMATED_SECONDS
    forward = request.model_dump(exclude_unset=True, exclude={"model", "prompt", "user"})

    async def call_provider(resolved: ResolvedProvider) -> VideoJob:
        return await avideo_generation(
            resolved.model, request.prompt, provider=resolved.provider, **resolved.kwargs, **forward
        )

    def refuse_model(resolved: ResolvedProvider) -> HTTPException | None:
        if getattr(AnyLLM.get_provider_class(resolved.provider), "SUPPORTS_VIDEO_GENERATION", False):
            return None
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Provider '{resolved.provider.value}' does not support video generation",
        )

    def refuse_unpriced(pricing: ModelPricing | None) -> HTTPException | None:
        if video_rate(pricing) is not None:
            return None
        return HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=no_video_price_detail(request.model),
            headers=error_headers(PRICING_REQUIRED),
        )

    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT,
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        request_params=request.model_fields_set,
        estimate=lambda pricing: estimate_video_cost(pricing, estimated_seconds),
        billing=VIDEO_BILLING,
        refuse_model=refuse_model,
        refuse_unpriced=refuse_unpriced,
        relabel=False,
    )
    job = outcome.result
    try:
        await jobs.record(
            video_id=job.id,
            provider=outcome.resolved.instance,
            model=outcome.resolved.model,
            user_id=outcome.user_id,
            api_key_id=api_key.id if api_key else None,
            workspace_id=outcome.workspace_id,
            seconds=request.seconds,
            counts_toward_budget=api_key is None or not api_key.exclude_from_budget,
        )
    except DATABASE_ERRORS as exc:
        raise database_error("Recording the video job", exc) from exc
    if job.status == "completed":
        # A provider that renders inside the start call reports the length now,
        # and may have no job to read back later, so the result is kept and the
        # bill lands here.
        record = await jobs.find(job.id)
        if record is not None:
            await jobs.remember_completed(record, job)
            await _settle(
                record,
                job,
                db=db,
                jobs=jobs,
                workspace_id=outcome.workspace_id,
                organization_id=await organization_for_workspace_id(db, outcome.workspace_id),
                deployment_charge=outcome.deployment_charge,
            )
    return job_response(job, provider=outcome.resolved.instance)


async def _settle(
    record: VideoJobRecord,
    job: VideoJob,
    *,
    db: PortSessionDep,
    jobs: VideoJobService,
    workspace_id: uuid.UUID,
    organization_id: uuid.UUID | None,
    deployment_charge: DeploymentCharge | None,
) -> None:
    """Bill a completed job once through the service, charging the deployment first where it paid."""
    charge_deployment = None
    if deployment_charge is not None:

        async def charge_deployment(cost: Decimal | None) -> None:
            await deployment_charge.settle(
                cost,
                model=record.model,
                description=f"{record.provider}:{record.model}",
                expects_cost=job.seconds is not None or record.seconds is not None,
                savepoint=db.begin_nested,
            )

    await jobs.settle_completed(
        record,
        reported_seconds=job.seconds,
        endpoint=USAGE_ENDPOINT_COMPLETED,
        workspace_id=workspace_id,
        organization_id=organization_id,
        deployment_paid=deployment_charge is not None,
        charge_deployment=charge_deployment,
    )


async def _retrieve(
    video_id: str,
    *,
    raw_request: Request,
    auth_result: tuple[APIKey | None, bool],
    db: PortSessionDep,
    config: GatewayConfig,
    jobs: VideoJobService,
) -> tuple[VideoJobRecord, VideoJob, LLMProvider, dict[str, Any]]:
    """The job as its record remembers it, else as the provider reports it, for a caller allowed to see it.

    A completed job is billed once. Also gives the implementation and
    credentials a further call on the job dispatches with.
    """
    api_key, is_master_key = auth_result
    record = jobs.authorize(
        await jobs.find(video_id),
        video_id,
        user_id=str(api_key.user_id) if api_key and api_key.user_id else None,
        workspace_id=api_key.workspace_id if api_key else None,
        is_master_key=is_master_key,
    )
    # The job's own workspace, so credentials and rates follow the organization that started it.
    workspace_id = record.workspace_id if record.workspace_id is not None else await resolve_workspace_id(db, api_key)
    organization_id = await organization_for_workspace_id(db, workspace_id)
    provider, kwargs, deployment_paid = await resolve_provider_by_name(
        config,
        record.provider,
        port=model_provider_port_on(raw_request, db),
        organization_id=organization_id,
        country_gate=country_gate_for(config, raw_request),
        workspace_id=workspace_id,
    )
    # The provider call can outlast a completion, and the billing after it
    # opens its own transaction, so the connection goes back to the pool here.
    await release_session(db)
    track_request(
        raw_request,
        endpoint=USAGE_ENDPOINT,
        model=record.model,
        provider=record.provider,
        user_id=record.user_id,
        api_key_id=api_key.id if api_key else None,
    )
    job = jobs.remembered(record)
    if job is None:
        try:
            job = await aretrieve_video(provider, video_id, **kwargs)
        except Exception as exc:
            logger.error("Video retrieve failed for %s (%s)", record.provider, type(exc).__name__)
            raise provider_call_failure(exc) from exc
        if job.status == "completed":
            await jobs.remember_completed(record, job)
    if job.status == "completed" and record.accounted_at is None:
        deployment_charge = None
        if deployment_paid and organization_id is not None:
            deployment_charge = DeploymentCharge(
                billing=billing_port_on(raw_request, db),
                organization_id=organization_id,
                api_key_id=record.api_key_id,
                actor_user_id=None,
                request_id=None,
            )
        await _settle(
            record,
            job,
            db=db,
            jobs=jobs,
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_charge=deployment_charge,
        )
    return record, job, provider, kwargs


@router.get("/{video_id}", response_model=VideoJobResponse)
async def retrieve_video(
    video_id: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: PortSessionDep,
    config: ConfigDep,
    jobs: VideoJobServiceDep,
) -> VideoJobResponse:
    """The job's status, from its record once it completed, else from the provider that runs it.

    The first retrieval that finds the job completed bills its length at the
    model's per-second video price. A key reads only the jobs its user started
    in its workspace; the master key reads any job.
    """
    record, job, _, _ = await _retrieve(
        video_id, raw_request=raw_request, auth_result=auth_result, db=db, config=config, jobs=jobs
    )
    return job_response(job, provider=record.provider)


@router.get(
    "/{video_id}/content",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {"content": {"video/*": {}}, "description": "The video, in the provider's media type"},
        status.HTTP_409_CONFLICT: {"description": "The job has not completed"},
        status.HTTP_502_BAD_GATEWAY: {"description": "LLM provider error"},
    },
)
async def retrieve_video_content(
    video_id: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: PortSessionDep,
    config: ConfigDep,
    jobs: VideoJobServiceDep,
) -> Response:
    """The finished video's bytes, in the provider's media type.

    Reads the job first, so a completed job is billed here as it would be on
    ``GET /api/v1/videos/{video_id}``. A job still running, or one that failed,
    is a 409 naming its status.
    """
    record, job, provider, kwargs = await _retrieve(
        video_id, raw_request=raw_request, auth_result=auth_result, db=db, config=config, jobs=jobs
    )
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Video '{video_id}' is not yet complete (status: {job.status}). "
                f"Call GET {API_ROOT}/videos/{video_id} to check the current status."
            ),
        )
    if not job.outputs:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="The provider reported the video complete and returned none"
        )
    output = job.outputs[0]
    try:
        content = await adownload_video(provider, output, **kwargs)
    except Exception as exc:
        logger.error("Video download failed for %s (%s)", record.provider, type(exc).__name__)
        raise provider_call_failure(exc) from exc
    return Response(content=content, media_type=output.mime_type)


@router.get("", response_model=VideoJobList)
async def list_videos(
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    jobs: VideoJobServiceDep,
    limit: Annotated[int, Query(ge=1, le=100, description="How many of the newest jobs to list")] = 20,
) -> VideoJobList:
    """The caller's video jobs, newest first, from the gateway's own records.

    A key sees the jobs its user started in its workspace; the master key sees
    every job. The records say what the gateway knows without asking a
    provider; a job's status is ``GET /api/v1/videos/{video_id}``'s.
    """
    api_key, is_master_key = auth_result
    if not is_master_key and api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    owner = None if is_master_key else api_key
    records = await jobs.page(
        user_id=None if owner is None else str(owner.user_id),
        workspace_id=None if owner is None else owner.workspace_id,
        limit=limit,
    )
    return VideoJobList(
        data=[
            VideoJobSummary(
                id=record.id,
                model=record.model,
                provider=record.provider,
                created_at=record.created_at,
                seconds=record.seconds,
                billed=record.accounted_at is not None,
            )
            for record in records
        ]
    )
