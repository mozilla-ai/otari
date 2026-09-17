"""The Playground: an in-product chat page, served to a dashboard session.

otari-ai#1947 asked for the Playground back after the hosted original was
retired (otari-ai#1920), and otari#663 is the port. This router is its whole
server side: one completion endpoint plus the page's own memory.

**How a completion here is authorized, and why it is not a loosened
``/chat/completions``.** The public completion route accepts an API key or the
deployment's master key and nothing else, because a keyless request resolves to
the *default* workspace, so honoring a dashboard cookie there would let any
signed-in member of any organization spend the default organization's BYO
credential and bill it (otari-ai#1880). That route is untouched. This one
authenticates the session, resolves the caller's own attribution user and proves
their membership of the workspace it will bill
(``playground_service.resolve_playground_principal``), and hands the result to
the same pipeline as a
:class:`~gateway.types.session_principal.SessionPrincipal`. Nothing about
routing, budget, guardrails, tools, pricing or settlement is reimplemented here:
``run_chat_completion`` is the same function the public route calls, so the two
paths cannot drift.

The hosted original solved this by minting a short-lived bearer and handing it
to the browser (otari-ai#1598). This does not, and that is the one deliberate
departure from what otari#663 wrote down: a credential in JavaScript is strictly
worse than no credential in JavaScript, and the constraint the issue was
protecting, that the browser never holds a durable key, is honored more strongly
by holding none at all. `docs/access-control.md` already says sessions are the
dashboard's credential.

**Everything else here is the page's memory, and it is per identity.** Consent,
saved transcripts, rated comparisons and pinned models all carry the caller's
own ``user_id`` as a query predicate rather than a post-load check, so another
identity's row answers exactly as a nonexistent one does. otari-ai#978 was a
comparisons list that trusted a client-supplied scope and leaked prompts and
model outputs across organizations; the predicate is what makes that
unreachable rather than merely fixed.

**Hosted serves the same page and forwards the one request that dispatches.**
A hosted control plane runs no inference (otari#822), so on that deployment the
completion below hands the resolved principal to
``services/playground_dispatch`` instead of to the local pipeline: it is
presented to ``config.data_plane_url`` as an API key that never leaves this
process, and the gateway there runs the pipeline and reports the usage that
debits the wallet. Everything else on this router is management traffic and is
served locally either way, so the page is whole rather than half-mounted.

Hybrid mode is the one deployment that serves none of this. It has no dashboard
session and no local tenancy to resolve one against, so
``main._register_core_routers`` mounts the router for standalone and hosted and
not for hybrid.
"""

import uuid
from typing import Annotated

import httpx
from any_llm.types.completion import ChatCompletion
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    CurrentIdentity,
    ModelProviderPortDep,
    get_config,
    get_db,
    get_log_writer,
    verify_master_key,
)
from gateway.api.routes.chat import ChatCompletionRequest, run_chat_completion
from gateway.core.config import GatewayConfig
from gateway.core.database import release_session
from gateway.core.surface import Surface
from gateway.log_config import logger
from gateway.models.playground import (
    PlaygroundComparisonCreate,
    PlaygroundComparisonsPublic,
    PlaygroundComparisonSummary,
    PlaygroundConsentPublic,
    PlaygroundConsentUpdate,
    PlaygroundConversationCreate,
    PlaygroundConversationsPublic,
    PlaygroundConversationSummary,
    PlaygroundFavoriteModelsPublic,
    PlaygroundFavoriteModelsUpdate,
    PlaygroundMessagesPublic,
)
from gateway.services import playground_dispatch, playground_service
from gateway.services.log_writer import LogWriter
from gateway.services.secret_box import SecretBoxUnavailableError
from gateway.types.session_principal import SessionPrincipal

router = APIRouter(
    prefix="/playground",
    tags=["playground"],
    # Authentication only, like the rest of the tenant-facing surface. What the
    # caller may reach is decided per request by the owner predicate on every
    # query and by the workspace resolver on the completion path, which is why
    # the deployment-operator gate does not belong here: the Playground is for
    # whoever signed in, not only for the operator.
    dependencies=[Depends(verify_master_key)],
)

# Published everywhere a dashboard session exists. A hosted control plane serves
# no inference of its own, which is why the completion below forwards rather than
# dispatches, and ``bootstrap.published_surfaces`` withholds the surface there
# while ``data_plane_url`` is unset: with nowhere to forward to, the page has
# nothing to offer.
SURFACE = Surface("playground")

_CONVERSATION_NOT_FOUND = "Conversation not found"
_COMPARISON_NOT_FOUND = "Comparison not found"

# A workspace parameter is optional everywhere it appears: the page sends the
# one its sidebar has selected, and a caller driving the API directly gets their
# organization's default. Either way it is resolved through the membership check
# in ``resolve_playground_workspace``, so naming one is
# never a way to reach a workspace the caller does not belong to.
# No default here, and the ``= None`` stays at each call site: FastAPI asserts on
# a ``Query`` default inside ``Annotated``.
_WORKSPACE_QUERY = Query(
    description=(
        "Workspace to act in. Defaults to the caller's organization's default workspace. "
        "A workspace the caller is not a member of answers 404, as a nonexistent one does."
    ),
)


class PlaygroundToolStatus(BaseModel):
    """Whether one gateway-run tool can be attached right now, and why not.

    Three states from two fields, which is what the composer's menu draws: a
    tool the deployment never configured is not offered, one the deployment
    configured and this workspace turned off is shown disabled with the reason,
    and an available one is a plain checkbox. A single boolean would collapse
    the first two, which is how a checkbox comes to look attachable and then
    fail at request time (otari-ai#1419).
    """

    configured: bool = Field(description="Whether this deployment has a backend for the tool at all.")
    enabled: bool = Field(description="Whether the caller's workspace may attach it.")
    reason: str | None = Field(
        default=None,
        description="Why it cannot be attached. Null when it can.",
    )


class PlaygroundMcpServer(BaseModel):
    """One of the workspace's MCP servers, as the tools menu lists it."""

    id: uuid.UUID
    name: str
    purpose_hint: str | None = None
    enabled: bool


class PlaygroundToolsResponse(BaseModel):
    """What the caller's workspace may attach to a Playground message."""

    web_search: PlaygroundToolStatus
    code_execution: PlaygroundToolStatus
    mcp_servers: list[PlaygroundMcpServer]


@router.post("/chat/completions", response_model=None)
async def playground_chat_completions(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: ChatCompletionRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    model_provider: ModelProviderPortDep,
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> ChatCompletion | StreamingResponse:
    """Run one chat completion for the signed-in caller.

    Streaming and non-streaming both, identically to ``POST
    /api/v1/chat/completions``: this resolves the principal and then calls the
    very same handler. The request is billed to the caller's own attribution
    user in the workspace they named (or their organization's default), against
    that user's budget, and writes the ordinary usage row with no ``api_key_id``,
    because there was no key.

    The body is ``ChatCompletionRequest`` unchanged, so the page sends the same
    request an SDK would and a model, tool or parameter the gateway gains is
    available here the day it lands. The workspace rides in the query string
    rather than in the body for that reason: a field added to the body would
    also have to be added to the pipeline's strip list, and a gateway-internal
    field that is not stripped is forwarded to the provider as a call kwarg.

    ``user`` in the body is the one field the pipeline will not read here: spend
    binds to the session's own attribution user, derived and never accepted.

    On a hosted control plane there is no local pipeline to call, so the same
    principal is forwarded to the data-plane gateway instead
    (:func:`_dispatch_to_data_plane`). The request and the response are the same
    either way.
    """
    principal = await playground_service.resolve_playground_principal(
        db,
        identity=identity,
        workspace_id=workspace_id,
    )
    if config.is_hosted_mode:
        return await _dispatch_to_data_plane(
            request=request,
            principal=principal,
            db=db,
            config=config,
        )
    return await run_chat_completion(
        raw_request=raw_request,
        response=response,
        background_tasks=background_tasks,
        request=request,
        db=db,
        config=config,
        log_writer=log_writer,
        model_provider=model_provider,
        session_principal=principal,
    )


_NO_DATA_PLANE_DETAIL = (
    "This deployment does not serve inference, and no data-plane gateway is configured for it. "
    "An operator sets data_plane_url to the gateway that serves this deployment's traffic."
)
_DATA_PLANE_UNREACHABLE_DETAIL = "The gateway that serves this deployment's inference could not be reached"
_NO_SECRET_KEY_DETAIL = "This deployment cannot store the credential it needs to reach its data-plane gateway"


async def _dispatch_to_data_plane(
    *,
    request: ChatCompletionRequest,
    principal: SessionPrincipal,
    db: AsyncSession,
    config: GatewayConfig,
) -> StreamingResponse:
    """Run this completion on the deployment's data plane rather than here.

    The hosted half of the route above. ``services/playground_dispatch`` owns why
    a forward and which key; this is the request-shaped part: resolve the key,
    release the session, and forward.

    The body is re-sent as the caller sent it (``exclude_unset``) rather than as
    this deployment's model defaults would fill it in, so a field the gateway
    understands and this build does not still arrives, and a default the data
    plane would have chosen is not overridden by one chosen here.

    The answer is streamed back whatever its shape. A non-streaming completion is
    one chunk of JSON carrying the upstream's own content type, so the page reads
    the same body it reads standalone and neither this nor the page has to branch
    on ``stream``.
    """
    if config.data_plane_url is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NO_DATA_PLANE_DETAIL)

    try:
        api_key = await playground_dispatch.resolve_dispatch_key(db, principal=principal)
    except SecretBoxUnavailableError:
        # The deployment stores provider credentials through the same key, so this
        # is a control plane that could not have served a completion anyway.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_NO_SECRET_KEY_DETAIL,
        ) from None
    # Released for the reason every dispatching route releases it: a pooled
    # connection must not be held across an upstream call, and a streamed answer
    # holds this one for as long as the model takes.
    await release_session(db)
    try:
        status_code, headers, body = await playground_dispatch.forward_completion(
            url=playground_dispatch.completions_url(config.data_plane_url),
            api_key=api_key,
            payload=request.model_dump(mode="json", exclude_unset=True),
        )
    except httpx.HTTPError as exc:
        # Logged with the cause and answered without it: the transport error names
        # this deployment's own hosts and ports, which is topology a tenant is not
        # owed and the error-detail boundary does not carry.
        logger.warning("Playground dispatch to the data plane failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_DATA_PLANE_UNREACHABLE_DETAIL,
        ) from None
    return StreamingResponse(body, status_code=status_code, headers=headers)


@router.get("/tools")
async def read_playground_tools(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> PlaygroundToolsResponse:
    """The gateway-run tools the caller's workspace may attach to a message.

    One read rather than the three the dashboard would otherwise make (the
    deployment's tool settings, the workspace's web-search row, its
    code-execution row), because the answer is a composition of them in a fixed
    direction: the deployment decides whether a tool exists and the workspace may
    only narrow that. Composing it here is what keeps the menu from offering
    something the request path would refuse.
    """
    resolved = await playground_service.resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)
    availability = await playground_service.resolve_tool_availability(db, config=config, workspace_id=resolved)
    return PlaygroundToolsResponse(
        web_search=_tool_status(availability.web_search),
        code_execution=_tool_status(availability.code_execution),
        mcp_servers=[
            PlaygroundMcpServer(
                id=server.id,
                name=server.name,
                purpose_hint=server.purpose_hint,
                enabled=server.enabled,
            )
            for server in availability.mcp_servers
        ],
    )


# ==============================================================================
# Content-retention consent
# ==============================================================================


@router.get("/consent")
async def read_playground_consent(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlaygroundConsentPublic:
    """What the caller has agreed the Playground may store.

    Both flags false for a caller who has never answered, and nothing is
    written: the page asks at the moment it needs the grant, so recording an
    answer on a page load would record one nobody gave.
    """
    return await playground_service.read_consent(db, user_id=identity.id)


@router.put("/consent")
async def update_playground_consent(
    update: PlaygroundConsentUpdate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlaygroundConsentPublic:
    """Grant or withdraw content retention, one flag at a time.

    An omitted flag is left as it was. Withdrawing blocks new saves and deletes
    nothing: what was stored with consent stays until its owner deletes it, which
    is what keeps a withdrawal from being a destructive action nobody asked for.
    """
    return await playground_service.update_consent(db, user_id=identity.id, update=update)


# ==============================================================================
# Saved conversations
# ==============================================================================


@router.get("/conversations")
async def list_playground_conversations(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> PlaygroundConversationsPublic:
    """The caller's own saved transcripts in one workspace, newest first.

    Not gated on consent: withdrawing it stops new saves, so a transcript saved
    while it was granted has to stay listable and deletable by its owner.
    """
    resolved = await playground_service.resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)
    return PlaygroundConversationsPublic(
        data=await playground_service.list_conversations(db, user_id=identity.id, workspace_id=resolved)
    )


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def save_playground_conversation(
    request: PlaygroundConversationCreate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlaygroundConversationSummary:
    """Save one transcript whole, for the caller, in a workspace they belong to.

    403 when content retention has not been granted. The page asks first, so
    reaching that is a client that skipped the prompt: a consent gate enforced
    only in the browser is not a consent gate.
    """
    resolved = await playground_service.resolve_playground_workspace(
        db, identity=identity, workspace_id=request.workspace_id
    )
    return await playground_service.save_conversation(db, user_id=identity.id, workspace_id=resolved, request=request)


@router.get("/conversations/{conversation_id}/messages")
async def read_playground_conversation_messages(
    conversation_id: uuid.UUID,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlaygroundMessagesPublic:
    """One saved transcript's turns, in order.

    404 for a transcript belonging to somebody else, the same answer an
    unknown id gets: the owner predicate is in the query, so the two are
    indistinguishable from here. An empty transcript is not a state a save can
    produce (the request requires at least one turn), so no rows means no row
    for this caller.
    """
    messages = await playground_service.read_conversation_messages(
        db, user_id=identity.id, conversation_id=conversation_id
    )
    if not messages:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_CONVERSATION_NOT_FOUND)
    return PlaygroundMessagesPublic(data=messages)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playground_conversation(
    conversation_id: uuid.UUID,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete one of the caller's saved transcripts, and its turns with it."""
    if not await playground_service.delete_conversation(db, user_id=identity.id, conversation_id=conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_CONVERSATION_NOT_FOUND)


# ==============================================================================
# Saved comparisons
# ==============================================================================


@router.get("/comparisons")
async def list_playground_comparisons(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> PlaygroundComparisonsPublic:
    """The caller's own rated comparisons in one workspace, newest first.

    Without the two answer bodies: the list shows a dozen rows and renders
    neither, and there is no detail endpoint because the page has no screen that
    reads one back. A comparison is a judgment that was recorded, not a
    transcript to resume.
    """
    resolved = await playground_service.resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)
    return PlaygroundComparisonsPublic(
        data=await playground_service.list_comparisons(db, user_id=identity.id, workspace_id=resolved)
    )


@router.post("/comparisons", status_code=status.HTTP_201_CREATED)
async def save_playground_comparison(
    request: PlaygroundComparisonCreate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PlaygroundComparisonSummary:
    """Record which of two models answered a question better.

    403 when comparison retention has not been granted; this is the flag with
    the wider disclosure, because the row keeps both models' full answers.
    """
    resolved = await playground_service.resolve_playground_workspace(
        db, identity=identity, workspace_id=request.workspace_id
    )
    return await playground_service.save_comparison(db, user_id=identity.id, workspace_id=resolved, request=request)


@router.delete("/comparisons/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playground_comparison(
    comparison_id: uuid.UUID,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete one of the caller's saved comparisons."""
    if not await playground_service.delete_comparison(db, user_id=identity.id, comparison_id=comparison_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_COMPARISON_NOT_FOUND)


# ==============================================================================
# Pinned models
# ==============================================================================


@router.get("/favorite-models")
async def read_playground_favorite_models(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> PlaygroundFavoriteModelsPublic:
    """The caller's pinned model keys in one workspace, in pinned order.

    Stored rather than kept in the browser, so a pin follows the person to their
    other devices; that is what the hosted original did and what makes the
    Favorites group in every picker worth having.
    """
    resolved = await playground_service.resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)
    return PlaygroundFavoriteModelsPublic(
        model_keys=await playground_service.list_favorite_models(db, user_id=identity.id, workspace_id=resolved)
    )


@router.put("/favorite-models")
async def replace_playground_favorite_models(
    update: PlaygroundFavoriteModelsUpdate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_id: Annotated[uuid.UUID | None, _WORKSPACE_QUERY] = None,
) -> PlaygroundFavoriteModelsPublic:
    """Replace the caller's pin list for one workspace.

    A replace rather than a toggle, because the order is part of the value and
    the client already holds the list it is rendering. Two tabs racing therefore
    resolve to one of the two lists rather than to an interleaving neither of
    them showed. Model keys are not validated against the catalog: a pinned model
    that leaves the catalog simply stops appearing in the picker, and refusing
    the write would make a stale pin unremovable.
    """
    resolved = await playground_service.resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)
    return PlaygroundFavoriteModelsPublic(
        model_keys=await playground_service.replace_favorite_models(
            db,
            user_id=identity.id,
            workspace_id=resolved,
            model_keys=update.model_keys,
        )
    )


def _tool_status(availability: playground_service.ToolAvailability) -> PlaygroundToolStatus:
    return PlaygroundToolStatus(
        configured=availability.configured,
        enabled=availability.enabled,
        reason=availability.reason,
    )
