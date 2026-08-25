"""Teaching the learned router: scored examples and warmth (standalone only).

A policy that names ``router: knn`` decides nothing until someone has told the
gateway which candidate is good enough for which kind of prompt. Two routes:

* ``POST /v1/routing/preferences/rank`` records the quality each candidate earned
  on a prompt, writing the routing-memory records the router later votes over. It
  takes a batch, because a pool needs ``router_seed_count`` examples (20 by
  default) before it routes at all, and twenty round trips is not a workflow.
* ``GET /v1/routing/status`` reports how warm each pool is, so an operator can see
  when the router will start routing and which policies depend on it.

There is deliberately no endpoint that fans a prompt out to the candidates for
you. Seeing what each candidate answers is what ``POST /v1/chat/completions``
already does, one call per candidate, and going through that path means the calls
are budget-checked and land in the usage log like all other provider spend. A
convenience endpoint that skipped both would be the only unmetered way to spend
money through this gateway.

Master-key gated, like ``/v1/routing/policies``, with ``user_id`` naming whose
memory is being taught rather than taking it from the calling key: which model
serves a caller is an operator decision, exactly as a policy's targets are.

Routing memory is **per user and per workspace**, even for a workspace-wide
policy: the records hold the prompts a user sends, so sharing them across users
would let one caller's traffic steer another's, and sharing them across
workspaces would do the same across tenants for a user who holds keys in both. A
workspace-wide learned policy therefore warms once per user in each workspace.
Both routes take an optional ``workspace_id``; omitting it means the deployment's
default workspace, which is where a master-key request bills and where every row
predating workspace scoping was backfilled.

Known gap, tracked on #187: there is no route that lists or deletes recorded
examples, so a mis-scored example can only be undone in the database. Until that
lands, ``rank`` validates hard (see :func:`_validated_scores`) rather than
accepting anything a typo can produce.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from any_llm.exceptions import AnyLLMError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.api.routes._helpers import resolve_managed_workspace_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import RouterPreference, RoutingMemory
from gateway.repositories.users_repository import get_active_user
from gateway.services.policy_store import effective_policies
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.routing import KNN_BACKEND, backend_pool_is_teachable, get_router_backend
from gateway.services.routing.knn import KnnRoutingMemory

router = APIRouter(prefix="/v1/routing", tags=["routing"])

# One request may not teach more than this. Each example costs an embedding call,
# so a batch is a loop with a bound rather than an unbounded fan-out.
MAX_EXAMPLES_PER_REQUEST = 100


class ScoredExample(BaseModel):
    """One prompt and how well each candidate answered it."""

    prompt: str = Field(min_length=1, description="The prompt that was tried.")

    @field_validator("prompt")
    @classmethod
    def _prompt_has_content(cls, value: str) -> str:
        """Refuse a blank prompt rather than counting it and storing nothing.

        ``record_preference`` returns 0 for a prompt with no content, so a
        whitespace-only one wrote an audit row, no routing-memory row, and a
        ``recorded`` count that silently disagreed with both.
        """
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value

    @field_validator("task_id")
    @classmethod
    def _normalize_task_id(cls, value: str | None) -> str | None:
        """Trim the partition label, and treat blank as absent.

        The request side already does this: ``Otari-Router-Task`` is trimmed and a
        blank header means the default pool. Storing `" support "` verbatim created a
        partition that ``Otari-Router-Task: support`` could never reach, and
        ``/status`` listed it as a real pool.
        """
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None
    scores: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        min_length=1,
        description=(
            "Selector -> quality in [0.0, 1.0], where 1.0 is a great answer. Ties are fine and "
            "meaningful: two models that both answered well is exactly the case where the router "
            "should take the cheaper one."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Partition this example belongs to, matching the Otari-Router-Task header requests send. "
            "Omit to file it in the user's default pool."
        ),
    )
    label_source: str = Field(default="human", description="Provenance of the scores: 'human' or 'judge'.")


class RankRequest(BaseModel):
    """Record how well each candidate did, for one or many prompts.

    A batch because of the arithmetic: a pool routes nothing until it holds
    ``router_seed_count`` examples (20 by default) and the vote reads the ``k``
    nearest (5 by default), so a useful first teaching pass is dozens of examples
    across the kinds of prompt you care about.
    """

    user_id: str = Field(description="Whose routing memory these examples belong to.")
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Which workspace's routing memory these examples belong to. Omit for the deployment's "
            "default workspace. Only requests billing to that workspace vote over them."
        ),
    )
    examples: list[ScoredExample] = Field(
        min_length=1,
        max_length=MAX_EXAMPLES_PER_REQUEST,
        description="The scored prompts to record.",
    )


class RecordedPool(BaseModel):
    """How warm one pool is after the write."""

    task_id: str | None
    records: int
    warm: bool


class RankResponse(BaseModel):
    recorded: int
    seed_count: int
    pools: list[RecordedPool]
    """Every pool this request wrote into, with its progress toward the seed count."""


class PoolStatus(BaseModel):
    records: int
    warm: bool


class TaskPool(BaseModel):
    task_id: str
    records: int
    warm: bool


class LearnedPolicy(BaseModel):
    """A policy whose selection depends on routing memory, for the status overview.

    Only a *learned* policy belongs here. A weighted policy names a router too, but
    its split is written in the policy document, so listing it under a warmth report
    would tie it to a pool it never reads.
    """

    name: str
    backend: str
    candidates: list[str]
    default_target: str


class RouterStatus(BaseModel):
    """How warm this user's routing memory is, and what depends on it.

    Routing memory has no single warmth: it is a set of independent pools.
    ``default_pool`` is what a request with no ``Otari-Router-Task`` header votes
    over (every record the user has in this workspace, labeled or not) and
    ``tasks`` lists each partition, which only requests carrying that label use.
    Each crosses ``seed_count`` on its own.
    """

    user_id: str
    workspace_id: uuid.UUID
    embedding_model: str
    seed_count: int
    granularity: str
    alpha: float
    k: int
    confidence_floor: float
    default_pool: PoolStatus
    tasks: list[TaskPool]
    policies: list[LearnedPolicy]


def _knn(config: GatewayConfig) -> KnnRoutingMemory:
    """The kNN backend, or a 500 if this build somehow cannot build one.

    There is no "router disabled" state to report: the backend exists whenever the
    gateway does, and whether it is *used* is a property of each policy. Teaching a
    pool before writing the policy that reads it is a legitimate order of
    operations, so these routes do not require a policy to exist.
    """
    backend = get_router_backend(config, KNN_BACKEND)
    if not isinstance(backend, KnnRoutingMemory):  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The kNN router backend is unavailable in this build.",
        )
    return backend


async def _require_user(db: AsyncSession, user_id: str) -> None:
    """404 unless ``user_id`` names a live user.

    Routing memory rows carry a foreign key to the user, so an unknown id would
    otherwise surface as an opaque 500 from the commit.
    """
    if await get_active_user(db, user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found")


def _canonical(
    config: GatewayConfig, selector: str, user_id: str | None = None, *, workspace_id: uuid.UUID | None = None
) -> str | None:
    """``instance:model`` for a selector, or ``None`` if it resolves to nothing."""
    try:
        resolved = resolve_provider_selector(config, selector, user_id, workspace_id=workspace_id)
    except (ValueError, AnyLLMError):
        return None
    return f"{resolved.instance}:{resolved.model}"


def _validated_scores(
    config: GatewayConfig, user_id: str, examples: list[ScoredExample], workspace_id: uuid.UUID
) -> dict[str, str]:
    """Refuse a score key no learned policy could ever ask about, and canonicalize the rest.

    The failure this prevents is the worst one the feature has. A mistyped selector
    is otherwise accepted with a 200, counts toward the seed count, and produces
    records the router can never match: ``/status`` reports the pool warm and every
    request declines with "no neighbor scored any candidate in this pool", which is
    visible only in a log line. No route can delete the bad records afterwards.

    Resolution alone is too weak a check: ``openai:gpt-4o-typo`` resolves happily,
    because the prefix names a configured instance and nothing here verifies model
    names. So keys are checked against the union of every learned policy's
    candidates (plus their default targets) for this user in this workspace,
    canonicalized so ``provider/model`` and ``instance:model`` spellings compare
    equal.

    Accepting those spellings is only safe if the stored key is the one the router
    looks up, so the returned map rewrites every accepted key to its canonical
    ``instance:model`` form, which is what :meth:`KnnRoutingMemory._score`
    canonicalizes the candidate pool to. Storing a *policy's* spelling instead
    would only move the mismatch: two learned policies for one user that spell the
    same model differently would agree with one and be invisible to the other.

    When no learned policy resolves for the user, only resolvability is enforced:
    teaching a pool before writing the policy that reads it is a legitimate order
    of operations, and refusing it would make the API demand a specific sequence.
    Those keys are canonicalized too, so the policy written afterwards matches them
    however it spells its candidates. A weighted policy does not count as one that
    resolves: it reads no examples, so letting its pool narrow what may be taught
    would refuse the very examples the learned policy being prepared needs.
    """
    known: dict[str, str] = {}
    for spec in effective_policies(config, user_id, workspace_id=workspace_id).values():
        if not backend_pool_is_teachable(spec.router_backend):
            continue
        for selector in [*spec.router_candidates, spec.default_target]:
            canonical = _canonical(config, selector, user_id, workspace_id=workspace_id)
            if canonical is not None:
                known.setdefault(canonical, selector)

    normalized: dict[str, str] = {}
    rejected: list[str] = []
    for example in examples:
        for selector in example.scores:
            if selector in rejected or selector in normalized:
                continue
            canonical = _canonical(config, selector, user_id, workspace_id=workspace_id)
            if canonical is None or (known and canonical not in known):
                rejected.append(selector)
                continue
            normalized[selector] = canonical
    if rejected:
        expected = (
            f" Candidates this user's learned policies can use: {', '.join(sorted(known.values()))}."
            if known
            else ""
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"These score keys do not name a model any learned policy can route to: "
                f"{', '.join(rejected)}. Records keyed on them would be unmatchable, so the pool would "
                f"report warm and never route, and nothing can delete them afterwards.{expected}"
            ),
        )

    # Two spellings of one candidate in one example would collapse onto a single
    # stored key, so one of the two scores would win silently. Refuse instead.
    for example in examples:
        seen: dict[str, str] = {}
        for selector in example.scores:
            target = normalized[selector]
            if target in seen and seen[target] != selector:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Score keys '{seen[target]}' and '{selector}' name the same model "
                        f"('{target}'), so one of their scores would be discarded. Score each "
                        "candidate once."
                    ),
                )
            seen[target] = selector
    return normalized


def _learned_policies(
    config: GatewayConfig, user_id: str | None, workspace_id: uuid.UUID
) -> list[LearnedPolicy]:
    policies: list[LearnedPolicy] = []
    for name, spec in effective_policies(config, user_id, workspace_id=workspace_id).items():
        backend = spec.router_backend
        if not backend_pool_is_teachable(backend) or backend is None:
            continue
        policies.append(
            LearnedPolicy(
                name=name,
                backend=backend,
                candidates=spec.router_candidates,
                default_target=spec.default_target,
            )
        )
    return sorted(policies, key=lambda policy: policy.name)


async def _pool_counts(
    db: AsyncSession, backend: KnnRoutingMemory, user_id: str, workspace_id: uuid.UUID
) -> tuple[int, list[tuple[str, int]]]:
    """This user's record counts in one workspace, for the current embedding model.

    Filtered by workspace because that is the partition the router reads
    (``KnnRoutingMemory._load_records``): a total spanning workspaces would report
    a pool warm that no single request ever votes over.
    """
    scope = (
        RoutingMemory.user_id == user_id,
        RoutingMemory.workspace_id == workspace_id,
        RoutingMemory.embedding_model == backend.embedding_model,
    )
    total = int((await db.execute(select(func.count()).select_from(RoutingMemory).where(*scope))).scalar_one())
    rows = (
        await db.execute(
            select(RoutingMemory.task_id, func.count())
            .where(*scope, RoutingMemory.task_id.is_not(None))
            .group_by(RoutingMemory.task_id)
            .order_by(func.count().desc(), RoutingMemory.task_id)
        )
    ).all()
    return total, [(str(task_id), int(count)) for task_id, count in rows]


@router.post("/preferences/rank", dependencies=[Depends(verify_master_key)])
async def rank_candidates(
    request: RankRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RankResponse:
    """Record scored examples: one routing-memory record each, plus an audit row.

    The routing-memory record is written before its audit row for each example,
    because it is the load-bearing one (the router votes over it) and embedding it
    can fail; writing the audit row only afterwards means a failed embedding never
    leaves an orphan audit row.

    A failed embedding is a 502 that names the model, not a 500. Every example in
    the batch is embedded, so this is the call an operator makes most often and the
    one most likely to meet a misconfigured ``router_embedding_model``.

    Score keys are stored canonically as ``instance:model`` (see
    :func:`_validated_scores`), which is the form the router canonicalizes its
    candidates to, so how a policy spells a candidate cannot decide whether it
    matches.
    """
    backend = _knn(config)
    await _require_user(db, request.user_id)
    workspace_id = await resolve_managed_workspace_id(db, request.workspace_id)
    # Committed before the loop, because `record_preference` writes its
    # routing-memory row in a session of its own. On a deployment that has never
    # provisioned tenancy, resolving the default workspace *creates* it, and an
    # uncommitted workspace is one the other session's foreign key cannot see. A
    # no-op when the workspace already existed, which is every other case.
    await db.commit()
    normalized = _validated_scores(config, request.user_id, request.examples, workspace_id)

    recorded = 0
    touched: set[str | None] = set()
    for example in request.examples:
        scores = {normalized[selector]: score for selector, score in example.scores.items()}
        try:
            written = await backend.record_preference(
                user_id=request.user_id,
                workspace_id=workspace_id,
                prompt=example.prompt,
                scores=scores,
                task_id=example.task_id,
                label_source=example.label_source,
            )
        except SQLAlchemyError as exc:
            logger.warning("Router example write failed after %d example(s)", recorded)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
            ) from exc
        except Exception as exc:
            # Broad on purpose, and matching the read path: `knn.rank` catches bare
            # `Exception` around the same call because an embedding provider can fail
            # in ways any-llm does not wrap. The read path degrades to the policy's
            # default target; the write path has nothing safe to do, so it says what
            # broke. Earlier examples in the batch are already committed, which the
            # message states rather than pretending the call was atomic.
            logger.warning(
                "Router example embedding failed after %d example(s): %s", recorded, type(exc).__name__
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Could not embed the example: the router embedding model "
                    f"'{backend.embedding_model}' is not reachable. Configure a provider for it, or set "
                    f"OTARI_ROUTER_EMBEDDING_MODEL to one you have. {recorded} example(s) were recorded "
                    "before this failure."
                ),
            ) from exc

        recorded += written
        touched.add(example.task_id)
        try:
            db.add(
                RouterPreference(
                    user_id=request.user_id,
                    workspace_id=workspace_id,
                    prompt=example.prompt,
                    task_id=example.task_id,
                    scores=example.scores,
                    label_source=example.label_source,
                )
            )
            await db.commit()
        except SQLAlchemyError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error"
            ) from None

    # Warmth of every pool this batch wrote into: each named partition, plus the
    # default pool when any example carried no task label.
    total, per_task = await _pool_counts(db, backend, request.user_id, workspace_id)
    counts = dict(per_task)
    pools = [
        RecordedPool(
            task_id=task_id,
            records=total if task_id is None else counts.get(task_id, 0),
            warm=(total if task_id is None else counts.get(task_id, 0)) >= backend.seed_count,
        )
        for task_id in sorted(touched, key=lambda value: (value is not None, value or ""))
    ]
    return RankResponse(recorded=recorded, seed_count=backend.seed_count, pools=pools)


@router.get("/status", dependencies=[Depends(verify_master_key)])
async def routing_memory_status(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user_id: Annotated[str, Query(description="Whose routing memory to report on.")],
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Which workspace's routing memory to report on. Omit for the default workspace."),
    ] = None,
) -> RouterStatus:
    """Report how warm one user's routing memory is in one workspace, per pool.

    ``user_id`` is required rather than optional because there is no aggregate
    answer: warmth is per user, and a total across users would describe a pool
    that no request ever votes over. The same holds across workspaces, which is
    why ``workspace_id`` narrows rather than aggregating; it merely defaults
    instead of being required, because a single-workspace deployment has one
    answer.
    """
    backend = _knn(config)
    await _require_user(db, user_id)
    target_workspace_id = await resolve_managed_workspace_id(db, workspace_id)
    total, per_task = await _pool_counts(db, backend, user_id, target_workspace_id)
    seed = backend.seed_count
    return RouterStatus(
        user_id=user_id,
        workspace_id=target_workspace_id,
        embedding_model=backend.embedding_model,
        seed_count=seed,
        granularity=backend.granularity,
        alpha=backend.alpha,
        k=backend.k,
        confidence_floor=backend.confidence_floor,
        default_pool=PoolStatus(records=total, warm=total >= seed),
        tasks=[
            TaskPool(task_id=task_id, records=count, warm=count >= seed) for task_id, count in per_task
        ],
        policies=_learned_policies(config, user_id, target_workspace_id),
    )


__all__ = ["router"]
