"""What the Playground page needs from the database, and who it may ask for.

Two halves that look unrelated and are not. :func:`resolve_playground_principal`
is the one that matters: it turns a dashboard session into the principal a
completion may run as, and everything else here is the page's own memory
(consent, saved transcripts, rated comparisons, pinned models), which exists
only because that principal is a real, tenanted caller rather than an anonymous
browser.

The reader of a stored row is always its own owner. Every query below carries
the ``user_id`` predicate rather than checking ownership after loading, so a row
belonging to somebody else answers exactly as a nonexistent one does. That is
not an incidental style: the hosted original leaked prompts and model outputs
across organizations precisely because a list query trusted a client-supplied
scope (otari-ai#978), and a predicate that cannot be omitted is what stops that
from being reachable again.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyConflictError, TenancyForbiddenError
from gateway.exceptions.organizations_exceptions import WorkspaceNotFoundError
from gateway.models.playground import (
    MAX_FAVORITE_MODELS,
    MAX_SAVED_COMPARISONS,
    MAX_SAVED_CONVERSATIONS,
    PlaygroundComparison,
    PlaygroundComparisonCreate,
    PlaygroundComparisonSummary,
    PlaygroundConsent,
    PlaygroundConsentPublic,
    PlaygroundConsentUpdate,
    PlaygroundConversation,
    PlaygroundConversationCreate,
    PlaygroundConversationSummary,
    PlaygroundFavoriteModel,
    PlaygroundMessage,
    PlaygroundMessagePublic,
)
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tools import WorkspaceCodeExecutionPolicy, WorkspaceMcpServer, WorkspaceWebSearchConfig
from gateway.models.users import User
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import resolve_workspace_in_organization
from gateway.services.workspace_scope import organization_default_workspace_id
from gateway.types.session_principal import SessionPrincipal

RetainedContent = Literal["conversations", "comparisons"]

# Refusals here are raised as the tenancy error family rather than as
# ``HTTPException``, because this slice authorizes through ``services/tenancy/``
# and that family is what the handler registered in ``gateway.main`` renders.
# The status belongs to the condition rather than to the endpoint, which is the
# rule ``gateway.exceptions`` states.
_SPEND_IDENTITY_REVOKED = "Your spend identity has been deactivated on this deployment; ask an operator to restore it."
_NO_WORKSPACE = "You are not a member of a workspace on this deployment; ask an operator to add you to one."


# ==============================================================================
# The principal a Playground completion runs as
# ==============================================================================


async def resolve_playground_workspace(
    db: AsyncSession,
    *,
    identity: TenancyUser,
    workspace_id: uuid.UUID | None,
) -> uuid.UUID:
    """The workspace this caller is acting in, proved to be theirs.

    A named one goes through ``resolve_workspace_in_organization``, so a
    workspace in another organization, or one in this organization the caller
    does not belong to, answers 404 exactly as a nonexistent one does. Omitting
    it targets the organization's default workspace, put through the same check,
    which is what ``POST /api/v1/organizations/me/keys`` does and for the same
    reason: the default is a convenience, not a way past the membership test.

    Every Playground route resolves through here, reads included, so a read and a
    completion can never disagree about which workspace a caller reached.
    """
    organizations = OrganizationService(db, membership_listener=None)
    organization = await organizations.get_active_organization_for_user(identity)

    resolved = workspace_id
    if resolved is None:
        resolved = await organization_default_workspace_id(db, organization.id)
        if resolved is None:
            raise TenancyConflictError(_NO_WORKSPACE)
    try:
        workspace = await resolve_workspace_in_organization(
            db,
            user=identity,
            workspace_id=resolved,
            organization=organization,
            organizations=organizations,
        )
    except WorkspaceNotFoundError:
        # A caller who named nothing is not pointed at a parameter they did not
        # send: they are in an organization whose default workspace they are not
        # a member of, which an operator fixes, not a different request.
        if workspace_id is None:
            raise TenancyConflictError(_NO_WORKSPACE) from None
        raise
    return workspace.id


async def resolve_playground_principal(
    db: AsyncSession,
    *,
    identity: TenancyUser,
    workspace_id: uuid.UUID | None,
) -> SessionPrincipal:
    """The caller, as a principal the completion pipeline will accept.

    :func:`resolve_playground_workspace` settles which workspace, and this adds
    the other two things
    :class:`~gateway.types.session_principal.SessionPrincipal` requires before
    the pipeline will run a completion without an API key:

    * **Which user spend binds to.** The caller's own attribution row, keyed on
      their identity's UUID as a string, provisioned on first use exactly as the
      member key surface provisions it. Never read from the request: this path
      takes no ``user`` parameter at all, so there is nothing for an escalation
      to travel on.
    * **Which models they may name.** That user's own ``allowed_models``
      default. A session holds no key to narrow it with, so the user default
      *is* the effective list, and the pipeline's allow-list gate then binds
      here exactly as it would for one of their keys.

    Reached only from the completion route, which is why the spend-identity
    check and the provisioning live here and not in the workspace resolver: a
    read of somebody's own saved transcripts should not mint them a spend row,
    and an operator who revoked one should not have a page load revive it.

    The caller owns the commit: the attribution row is staged rather than
    committed, because the route that asked for this is about to run a
    completion and the request's transaction boundary is that route's.
    """
    resolved_workspace_id = await resolve_playground_workspace(db, identity=identity, workspace_id=workspace_id)

    # Deleting a spend identity is an operator's revocation
    # (``DELETE /api/v1/users`` soft-deletes the row and deactivates every key it
    # holds), and ``get_or_create_attribution_user`` *revives* a soft-deleted
    # row, which is right for the membership paths that call it and wrong here.
    # Refused rather than revived, exactly as ``POST
    # /api/v1/organizations/me/keys`` refuses the same owner.
    revoked = (await db.execute(select(User.deleted_at).where(User.user_id == str(identity.id)))).scalar_one_or_none()
    if revoked is not None:
        raise TenancyConflictError(_SPEND_IDENTITY_REVOKED)

    owner = await get_or_create_attribution_user(
        db,
        user_id=str(identity.id),
        alias=identity.full_name or identity.email,
    )
    return SessionPrincipal(
        user_id=owner.user_id,
        workspace_id=resolved_workspace_id,
        allowed_models=owner.allowed_models,
    )


# ==============================================================================
# Which tools this workspace can attach
# ==============================================================================


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    """Whether one gateway-run tool can be attached, and why not when it cannot.

    Two fields rather than one boolean, because the page draws three states and
    a boolean can only carry two. A tool the deployment never configured is not
    offered at all; one the deployment configured and this workspace turned off
    is shown disabled, with the reason, so somebody looking for it learns where
    it went instead of concluding the page is broken. That distinction is what
    otari-ai#1419 was: a checkbox that looked attachable and failed at request
    time.
    """

    configured: bool
    enabled: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class McpServerAvailability:
    """One of the workspace's MCP servers, as the tools menu lists it."""

    id: uuid.UUID
    name: str
    purpose_hint: str | None
    enabled: bool


@dataclass(frozen=True, slots=True)
class PlaygroundToolAvailability:
    """Everything the composer's tools menu renders."""

    web_search: ToolAvailability
    code_execution: ToolAvailability
    mcp_servers: list[McpServerAvailability]


_WORKSPACE_DISABLED = "Turned off for this workspace."
_NOT_CONFIGURED = "No backend is configured on this deployment."


async def resolve_tool_availability(
    db: AsyncSession,
    *,
    config: GatewayConfig,
    workspace_id: uuid.UUID,
) -> PlaygroundToolAvailability:
    """What the caller's workspace may attach to a Playground message.

    The deployment decides whether a tool exists and the workspace may only
    narrow that, which is the rule ``prepare_gateway_tools`` enforces at
    admission (`src/gateway/AGENTS.md`). This read composes the same two facts in
    the same direction, so the menu cannot offer something the request path would
    then refuse. It is a read of stored policy only and dials nothing: a live
    reachability probe belongs to the request that needs the backend, not to
    drawing a menu.
    """
    # The same question the request path asks, asked the same way, so the menu
    # cannot hide a tool a request would then be allowed to use.
    sandbox_configured = config.sandbox_configured()
    web_search_configured = config.web_search_configured()

    web_search_row = (
        await db.execute(
            select(WorkspaceWebSearchConfig.enabled).where(WorkspaceWebSearchConfig.workspace_id == workspace_id)
        )
    ).scalar_one_or_none()
    code_execution_row = (
        await db.execute(
            select(WorkspaceCodeExecutionPolicy.enabled).where(
                WorkspaceCodeExecutionPolicy.workspace_id == workspace_id
            )
        )
    ).scalar_one_or_none()
    servers = (
        (
            await db.execute(
                select(WorkspaceMcpServer)
                .where(WorkspaceMcpServer.workspace_id == workspace_id)
                .order_by(WorkspaceMcpServer.name)
            )
        )
        .scalars()
        .all()
    )

    return PlaygroundToolAvailability(
        web_search=_tool_availability(web_search_configured, web_search_row),
        code_execution=_tool_availability(sandbox_configured, code_execution_row),
        mcp_servers=[
            McpServerAvailability(
                id=server.id,
                name=server.name,
                purpose_hint=server.purpose_hint,
                enabled=server.enabled,
            )
            for server in servers
        ],
    )


def _tool_availability(configured: bool, workspace_enabled: bool | None) -> ToolAvailability:
    """Compose the deployment's answer with the workspace's narrowing of it.

    ``None`` for the workspace means no row, which means no narrowing: that is
    what keeps a deployment that configured nothing per workspace behaving as it
    did (#655/#678), and it must not read as "off".
    """
    if not configured:
        return ToolAvailability(configured=False, enabled=False, reason=_NOT_CONFIGURED)
    if workspace_enabled is False:
        return ToolAvailability(configured=True, enabled=False, reason=_WORKSPACE_DISABLED)
    return ToolAvailability(configured=True, enabled=True)


# ==============================================================================
# Content-retention consent
# ==============================================================================


async def read_consent(db: AsyncSession, *, user_id: uuid.UUID) -> PlaygroundConsentPublic:
    """This identity's consent, defaulting to nothing agreed.

    A missing row reads as both flags false and provisions nothing. Writing a
    row here would record an answer the person never gave, and the page asks for
    consent at the moment it needs it rather than on load.
    """
    row = (
        await db.execute(select(PlaygroundConsent).where(col(PlaygroundConsent.user_id) == user_id))
    ).scalar_one_or_none()
    return row.to_public() if row is not None else PlaygroundConsentPublic()


async def update_consent(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    update: PlaygroundConsentUpdate,
) -> PlaygroundConsentPublic:
    """Apply a partial consent change, creating the row on first grant.

    An omitted flag is left as it was, which is why the row is read before it is
    written: the page grants one flag at a time, and a write that defaulted the
    other would move a consent record on its own.
    """
    row = (
        await db.execute(select(PlaygroundConsent).where(col(PlaygroundConsent.user_id) == user_id))
    ).scalar_one_or_none()
    if row is None:
        row = PlaygroundConsent(user_id=user_id)
        db.add(row)
    if update.store_conversations is not None:
        row.store_conversations = update.store_conversations
    if update.store_comparisons is not None:
        row.store_comparisons = update.store_comparisons
    await db.commit()
    return row.to_public()


async def _require_consent(db: AsyncSession, *, user_id: uuid.UUID, kind: RetainedContent) -> None:
    """Refuse a save this identity has not agreed to.

    The page asks first and only calls the save endpoint after the grant lands,
    so reaching this is either a client that skipped the prompt or a caller
    driving the API directly. Either way the server is the one that has to say
    no: a consent gate enforced only in the browser is not a consent gate.
    """
    consent = await read_consent(db, user_id=user_id)
    granted = consent.store_conversations if kind == "conversations" else consent.store_comparisons
    if not granted:
        raise TenancyForbiddenError(f"Saving {kind} requires agreeing to content retention first.")


# ==============================================================================
# Saved conversations
# ==============================================================================


async def list_conversations(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> list[PlaygroundConversationSummary]:
    """This identity's saved transcripts in one workspace, newest first.

    The turn count is a **correlated** subquery, not a grouped one joined in.
    Both are one round trip, and the difference is what they read: grouping
    every row of ``playground_message`` aggregates the whole table, including
    every other person's transcripts, before the join throws almost all of it
    away. Correlated, each of the at-most-hundred rows is one seek on
    ``ix_playground_message_conversation_id``. Loading the turns themselves is
    the option neither of these is: the list renders a count and nothing else
    from them.
    """
    message_count = (
        select(func.count())
        .select_from(PlaygroundMessage)
        .where(col(PlaygroundMessage.conversation_id) == col(PlaygroundConversation.id))
        .correlate(PlaygroundConversation)
        .scalar_subquery()
    )
    rows = (
        await db.execute(
            select(PlaygroundConversation, message_count)
            .where(
                col(PlaygroundConversation.user_id) == user_id,
                col(PlaygroundConversation.workspace_id) == workspace_id,
            )
            .order_by(col(PlaygroundConversation.created_at).desc())
            .limit(MAX_SAVED_CONVERSATIONS)
        )
    ).all()
    return [
        PlaygroundConversationSummary(
            id=conversation.id,
            workspace_id=conversation.workspace_id,
            title=conversation.title,
            model=conversation.model,
            message_count=message_count,
            created_at=conversation.created_at,
        )
        for conversation, message_count in rows
    ]


async def save_conversation(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    request: PlaygroundConversationCreate,
) -> PlaygroundConversationSummary:
    """Store one transcript whole, pruning the oldest beyond the cap.

    ``workspace_id`` is the resolved one, not ``request.workspace_id``: the route
    proved membership of a workspace and this writes into that one, so a body
    naming another cannot move the row.

    ``position`` is assigned from the list's own order rather than accepted, so
    the ordering a transcript reads back in is the one it was saved in and no
    client can write two turns into one slot.
    """
    await _require_consent(db, user_id=user_id, kind="conversations")

    conversation = PlaygroundConversation(
        user_id=user_id,
        workspace_id=workspace_id,
        title=request.title,
        model=request.model,
    )
    db.add(conversation)
    await db.flush()
    for position, message in enumerate(request.messages):
        db.add(
            PlaygroundMessage(
                conversation_id=conversation.id,
                position=position,
                role=message.role,
                content=message.content,
                reasoning=message.reasoning,
            )
        )
    await _prune_oldest(
        db,
        model=PlaygroundConversation,
        user_id=user_id,
        workspace_id=workspace_id,
        keep=MAX_SAVED_CONVERSATIONS,
    )
    await db.commit()
    return PlaygroundConversationSummary(
        id=conversation.id,
        workspace_id=conversation.workspace_id,
        title=conversation.title,
        model=conversation.model,
        message_count=len(request.messages),
        created_at=conversation.created_at,
    )


async def read_conversation_messages(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> list[PlaygroundMessagePublic]:
    """One saved transcript's turns, in order, for its own owner.

    The owner predicate is on the *conversation*, joined rather than checked
    afterwards, so somebody else's conversation id returns nothing and the route
    turns that into the same 404 an unknown id gets.
    """
    rows = (
        (
            await db.execute(
                select(PlaygroundMessage)
                .join(
                    PlaygroundConversation,
                    col(PlaygroundConversation.id) == col(PlaygroundMessage.conversation_id),
                )
                .where(
                    col(PlaygroundMessage.conversation_id) == conversation_id,
                    col(PlaygroundConversation.user_id) == user_id,
                )
                .order_by(col(PlaygroundMessage.position))
            )
        )
        .scalars()
        .all()
    )
    return [
        PlaygroundMessagePublic(role=row.role, content=row.content, reasoning=row.reasoning)
        for row in rows
    ]


async def delete_conversation(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> bool:
    """Delete one of this identity's transcripts; False when it has no such row.

    The turns go with it through the foreign key's CASCADE rather than a second
    statement, so a partially deleted transcript is not a state this can reach.
    """
    result = await db.execute(
        delete(PlaygroundConversation).where(
            col(PlaygroundConversation.id) == conversation_id,
            col(PlaygroundConversation.user_id) == user_id,
        )
    )
    await db.commit()
    # getattr with a default: mypy sees .execute() as Result (no rowcount);
    # rowcount lives on CursorResult. Matches ``batch_service``.
    return bool(getattr(result, "rowcount", 0))


# ==============================================================================
# Saved comparisons
# ==============================================================================


async def list_comparisons(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> list[PlaygroundComparisonSummary]:
    """This identity's rated comparisons in one workspace, newest first."""
    rows = (
        (
            await db.execute(
                select(PlaygroundComparison)
                .where(
                    col(PlaygroundComparison.user_id) == user_id,
                    col(PlaygroundComparison.workspace_id) == workspace_id,
                )
                .order_by(col(PlaygroundComparison.created_at).desc())
                .limit(MAX_SAVED_COMPARISONS)
            )
        )
        .scalars()
        .all()
    )
    return [row.to_summary() for row in rows]


async def save_comparison(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    request: PlaygroundComparisonCreate,
) -> PlaygroundComparisonSummary:
    """Record one model preference, pruning the oldest beyond the cap."""
    await _require_consent(db, user_id=user_id, kind="comparisons")

    comparison = PlaygroundComparison(
        user_id=user_id,
        workspace_id=workspace_id,
        user_question=request.user_question,
        model_a=request.model_a,
        model_b=request.model_b,
        model_a_answer=request.model_a_answer,
        model_b_answer=request.model_b_answer,
        preference=request.preference,
    )
    db.add(comparison)
    await _prune_oldest(
        db,
        model=PlaygroundComparison,
        user_id=user_id,
        workspace_id=workspace_id,
        keep=MAX_SAVED_COMPARISONS,
    )
    await db.commit()
    return comparison.to_summary()


async def delete_comparison(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    comparison_id: uuid.UUID,
) -> bool:
    """Delete one of this identity's comparisons; False when it has no such row."""
    result = await db.execute(
        delete(PlaygroundComparison).where(
            col(PlaygroundComparison.id) == comparison_id,
            col(PlaygroundComparison.user_id) == user_id,
        )
    )
    await db.commit()
    # getattr with a default: mypy sees .execute() as Result (no rowcount);
    # rowcount lives on CursorResult. Matches ``batch_service``.
    return bool(getattr(result, "rowcount", 0))


async def _prune_oldest(
    db: AsyncSession,
    *,
    model: type[PlaygroundConversation] | type[PlaygroundComparison],
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    keep: int,
) -> None:
    """Drop this owner's oldest rows in this workspace past ``keep``.

    The new row is included in the count, whichever caller this is: autoflush
    runs on the ``SELECT`` below, so a row staged but not yet flushed is written
    before it is read. So ``keep`` is the total kept and not the total plus the
    one being added.

    What this must not do is prune somebody else's rows, which is why the owner
    predicate is on the select and not only on the delete.

    Deleted by id rather than with an ``ORDER BY ... LIMIT`` on the delete
    itself, which PostgreSQL does not accept and SQLite accepts only when
    compiled for it.
    """
    stale = (
        (
            await db.execute(
                select(col(model.id))
                .where(
                    col(model.user_id) == user_id,
                    col(model.workspace_id) == workspace_id,
                )
                .order_by(col(model.created_at).desc())
                .offset(keep)
            )
        )
        .scalars()
        .all()
    )
    if stale:
        await db.execute(delete(model).where(col(model.id).in_(stale)))


# ==============================================================================
# Pinned models
# ==============================================================================


async def list_favorite_models(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> list[str]:
    """This identity's pinned model keys in one workspace, in pinned order.

    Ordered by the stored ``position`` and not by ``created_at``: a replace that
    only reorders the list writes no new rows, so an insertion clock would read
    the list back in an order nobody chose.
    """
    return list(
        (
            await db.execute(
                select(col(PlaygroundFavoriteModel.model_key))
                .where(
                    col(PlaygroundFavoriteModel.user_id) == user_id,
                    col(PlaygroundFavoriteModel.workspace_id) == workspace_id,
                )
                .order_by(col(PlaygroundFavoriteModel.position))
            )
        )
        .scalars()
        .all()
    )


async def replace_favorite_models(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    model_keys: list[str],
) -> list[str]:
    """Replace this identity's pin list for one workspace with ``model_keys``.

    Delete-then-insert rather than a diff, because the list is short, the order
    is part of the value, and a diff would have to renumber the survivors anyway.
    Duplicates in the request are collapsed keeping first occurrence, which is
    the position the client showed them at; without that the unique constraint
    would refuse a list a client could plausibly send twice.

    No pruning branch here, unlike the two histories: the request schema caps the
    list, so a replace cannot leave more rows than the cap.
    """
    deduplicated = list(dict.fromkeys(model_keys))
    await db.execute(
        delete(PlaygroundFavoriteModel).where(
            col(PlaygroundFavoriteModel.user_id) == user_id,
            col(PlaygroundFavoriteModel.workspace_id) == workspace_id,
        )
    )
    for position, model_key in enumerate(deduplicated[:MAX_FAVORITE_MODELS]):
        db.add(
            PlaygroundFavoriteModel(
                user_id=user_id,
                workspace_id=workspace_id,
                model_key=model_key,
                position=position,
            )
        )
    await db.commit()
    return deduplicated[:MAX_FAVORITE_MODELS]
