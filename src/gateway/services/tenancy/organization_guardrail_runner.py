"""The guardrails an organization defined, built once per worker and held ready.

The store next door (`organization_guardrail_definition_service`) says what each
check is. This builds it. A later step looks one up by id and runs it on the
request path; nothing here reads a request or changes what happens to one.

It lives in this package, and not beside `services/guardrails.py`, because
`org_provider_key_service` here already does this exact job for provider keys: a
process-wide cache, a load at startup and a background refresher. A
`services/guardrails/` package would collide with the module that serves the
remote path, which stays a supported backend and is untouched.

**Three facts about any-guardrail drive the whole design.**

*One: ``create`` and ``evaluate`` are plain synchronous calls.* Both end in a
vendor HTTP request, and `watsonx_guardian` reaches the network inside ``create``
rather than only inside a check. Run either on the event loop and every other
request on this worker stops, so both go to a thread.

*Two: a thread cannot be cancelled.* A build or a check that outruns its deadline
runs to the end and its answer is dropped. Nothing waits on it, which is what
keeps this simple, and it is also why the pool is bounded rather than trusted to
drain: the bound is the backstop a cancellation cannot be.

*Three: ``asyncio.to_thread`` would share the process-wide default executor* with
`services/file_extractors`, so a burst of checks would stall file extraction as
collateral damage. This pool is its own, and small.

**The cache is keyed on the definition, not on the profile an organization
mandates it under.** One definition mandated under three profiles is one entry
and one vendor client, which is the whole return on storing the two separately.

**A refresh is a diff, not a rebuild.** The provider overlay next door reloads
decrypted dictionaries, where this holds constructed vendor clients; rebuilding
all of them twice a minute per worker would do real I/O for nothing. Each entry
carries the row's ``updated_at``, which the column sets on every write, so a tick
builds only what moved or is missing and drops what is gone or disabled.

**There is no lazy build.** A definition this worker does not hold is disabled,
absent, or one that failed to build, and a request cannot tell those apart. A
build while a request waits is the thing the cache exists to prevent.

**A vendor exception's message never reaches a log line.** A vendor SDK
constructor raises whatever it likes and may echo the arguments it was handed,
which are the organization's credentials. What is logged is the guardrail class,
the definition id and the exception's type. Not the definition's name either: an
admin may have typed a credential into it by mistake. The one exception is
`ImportError`, whose text is upstream's own fixed install hint and the only
build failure an operator can act on.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from any_guardrail import AnyGuardrail, Guardrail, GuardrailName, GuardrailOutput

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.tenancy import OrganizationGuardrailDefinitionRepository
from gateway.services.guardrails import GuardrailsNotReachableError
from gateway.services.tenancy.organization_guardrail_definition_service import build_arguments

if TYPE_CHECKING:
    from datetime import datetime

    from gateway.models.guardrails import OrganizationGuardrailDefinition

# Same cadence as the provider key overlay, and for the same reason: a write
# refreshes the worker that served it, and this is what carries it to the
# siblings and the other replicas.
GUARDRAIL_RUNNER_REFRESH_SECONDS = 30.0

# Small on purpose. These are vendor HTTP calls, not model loads, and the bound
# is what stops a hung vendor from taking threads the rest of the process needs.
_MAX_THREADS = 4

# Well under the remote path's 30 seconds. A guardrail runs before a request is
# dispatched, so its deadline is part of the caller's latency rather than a
# backstop on a call already made.
_CHECK_TIMEOUT_SECONDS = 10.0

# Longer, because a build happens off the request path and `watsonx_guardian`
# validates its URL over the network inside the constructor.
_BUILD_TIMEOUT_SECONDS = 20.0

_T = TypeVar("_T")

# (organization_id, definition_id) -> what this worker holds for it.
_held: dict[tuple[uuid.UUID, uuid.UUID], "_Held"] = {}
_loaded_at: float | None = None
_threads: ThreadPoolExecutor | None = None


@dataclass(frozen=True)
class GuardrailCheck:
    """One in-process verdict, in the three fields the remote path also reports.

    Deliberately not any-guardrail's own `GuardrailOutput`: the caller is
    `services/guardrails`, which knows nothing about this package and must not
    have to import a vendor library to name the answer it already has a shape
    for.
    """

    valid: bool
    explanation: str | None
    score: float | None


@dataclass(frozen=True)
class _Held:
    """One definition this worker has tried to build.

    ``guardrail`` is None when the build failed. A failure is held rather than
    dropped so that a later read can say "saved, but not running" instead of
    leaving a mandated check silently unevaluated, and so the next tick does not
    retry a build that cannot work until the row changes.
    """

    fingerprint: datetime
    guardrail_name: str
    guardrail: Guardrail | None


class OrganizationGuardrailHandle:
    """A built guardrail, ready to run, with the deadline and the thread around it."""

    def __init__(self, definition_id: uuid.UUID, guardrail_name: GuardrailName, guardrail: Guardrail) -> None:
        self._definition_id = definition_id
        self._guardrail_name = guardrail_name
        self._guardrail = guardrail

    async def check(self, prompt: str) -> GuardrailCheck:
        """Run the guardrail over ``prompt``, in a thread, under a deadline.

        Every failure becomes `GuardrailsNotReachableError`, so an in-process
        guardrail is governed by the same fail-open and fail-closed handling as a
        remote one and the caller needs no second branch. The message names the
        definition and the exception's type and nothing else, because the
        caller's fail-open arm logs it; the public detail is generic, and a
        caller that knows the profile replaces it with one that names it.
        """
        try:
            output = await _in_a_thread(
                lambda: AnyGuardrail.evaluate(self._guardrail_name, self._guardrail, prompt),
                seconds=_CHECK_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - see the module docstring: the message is never logged
            raise GuardrailsNotReachableError(
                f"guardrail definition {self._definition_id} failed to run: {type(exc).__name__}",
                public_detail="guardrail could not be evaluated",
            ) from exc

        # `evaluate` returns whatever the guardrail's own `validate` returned, and
        # its type says that can be a list. No guardrail an organization may
        # define returns one for a single prompt, so a list is an answer this
        # code does not understand rather than one to take the first element of:
        # picking [0] would hide a per-message verdict set behind a verdict that
        # looks whole. Refusing it puts the request through the caller's
        # mode-aware handling, which is the right posture for "unreadable".
        if not isinstance(output, GuardrailOutput):
            raise GuardrailsNotReachableError(
                f"guardrail definition {self._definition_id} returned {type(output).__name__}, not one verdict",
                public_detail="guardrail could not be evaluated",
            )
        return GuardrailCheck(valid=output.valid, explanation=output.explanation, score=output.score)


def handle(organization_id: uuid.UUID, definition_id: uuid.UUID) -> OrganizationGuardrailHandle | None:
    """The built guardrail for this definition, or None when this worker holds none.

    A plain dictionary lookup. None covers every reason at once, which is the
    right shape: disabled, deleted and failed to build are one answer to a
    request, and the caller has to treat them alike.
    """
    entry = _held.get((organization_id, definition_id))
    if entry is None or entry.guardrail is None:
        return None
    return OrganizationGuardrailHandle(definition_id, GuardrailName(entry.guardrail_name), entry.guardrail)


def build_state(organization_id: uuid.UUID, definition_id: uuid.UUID) -> str | None:
    """Whether this worker built the definition, failed to, or has not seen it.

    Separate from :func:`handle` because the two audiences differ: a request only
    needs to know whether a check can run, while an admin reading their own row
    needs "saved, but not running" told apart from "not saved here yet".
    """
    entry = _held.get((organization_id, definition_id))
    if entry is None:
        return None
    return "built" if entry.guardrail is not None else "failed"


def runner_is_stale(ttl: float = GUARDRAIL_RUNNER_REFRESH_SECONDS) -> bool:
    """Whether the cache has never loaded or has outlived ``ttl``."""
    return _loaded_at is None or (time.monotonic() - _loaded_at) >= ttl


def reset_guardrail_runner() -> None:
    """Drop everything held and give the threads back (shutdown, tests).

    The pool is rebuilt on the next call rather than kept, because this also runs
    between two lifespans of the same app object in the test suite, and a pool
    that outlived the shutdown that dropped its guardrails would be holding
    threads for work nothing can ask for any more.
    """
    global _loaded_at, _threads  # noqa: PLW0603

    _held.clear()
    _loaded_at = None
    if _threads is not None:
        _threads.shutdown(wait=False, cancel_futures=True)
        _threads = None


async def load_guardrail_runner_at_startup() -> None:
    """Build what this worker will serve, before it serves anything.

    A failure is logged rather than raised, the posture
    `load_org_provider_keys_at_startup` takes: an organization's guardrails are
    additive to the deployment's own, and a gateway that boots holding none is
    better than one that refuses to boot because a vendor's constructor did.
    """
    reset_guardrail_runner()
    try:
        await _refresh_on_a_session_of_its_own()
    except Exception:
        logger.exception("Failed to build organization guardrail definitions; continuing with none held")
        return
    if _held:
        logger.info(
            "Built %d of %d organization guardrail definition(s)",
            sum(1 for entry in _held.values() if entry.guardrail is not None),
            len(_held),
        )


async def run_guardrail_runner_refresher(interval: float = GUARDRAIL_RUNNER_REFRESH_SECONDS) -> None:
    """Rebuild what moved, forever, so another worker's write arrives here too.

    Every error is swallowed and retried on the next tick, so a database blip
    cannot kill the refresher and freeze what this worker holds. Cancelled at
    shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Organization guardrail rebuild failed; retrying in %ss", interval, exc_info=True)


async def _refresh_on_a_session_of_its_own() -> None:
    """What the startup load and the refresher call: a refresh with nothing shared.

    Split from the refresh itself so the read can be pointed at a caller's Unit
    of Work, which is what a test does and what a write path that rebuilds one
    entry will want.
    """
    async with create_unit_of_work() as uow:
        await refresh_guardrail_runner(uow)


async def refresh_guardrail_runner(uow: UnitOfWork) -> None:
    """Read every enabled definition and build the ones this worker does not hold.

    The read is one query and its block closes before the first build, so no
    vendor call is ever made with a database transaction open. The rows outlive
    it because the session factory sets ``expire_on_commit=False`` and this
    table declares no relationship, so nothing here loads lazily.
    """
    global _loaded_at  # noqa: PLW0603

    async with uow:
        rows = await OrganizationGuardrailDefinitionRepository(uow).list_enabled_in_every_organization()

    rebuilt: dict[tuple[uuid.UUID, uuid.UUID], _Held] = {}
    for row in rows:
        key = (row.organization_id, row.id)
        entry = _held.get(key)
        rebuilt[key] = entry if entry is not None and entry.fingerprint == row.updated_at else await _build(row)

    _held.clear()
    _held.update(rebuilt)
    _loaded_at = time.monotonic()


async def _build(definition: OrganizationGuardrailDefinition) -> _Held:
    """Construct one definition's guardrail, recording a failure rather than raising.

    One row this deployment cannot build must not cost the others theirs, and a
    build failure is a state a later read reports rather than an error anyone is
    waiting on.
    """
    try:
        arguments = build_arguments(definition)
        guardrail = await _in_a_thread(
            lambda: AnyGuardrail.create(GuardrailName(definition.guardrail_name), **arguments),
            seconds=_BUILD_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - see the module docstring: the message is never logged
        logger.warning(
            "Could not build guardrail %s for definition %s: %s",
            definition.guardrail_name,
            definition.id,
            _safe_reason(exc),
        )
        return _Held(fingerprint=definition.updated_at, guardrail_name=definition.guardrail_name, guardrail=None)
    return _Held(fingerprint=definition.updated_at, guardrail_name=definition.guardrail_name, guardrail=guardrail)


def _safe_reason(exc: BaseException) -> str:
    """What may be said about a failure, which is its type and almost never its text.

    `ImportError` is the exception. Its message is upstream's own constant install
    hint, it names the extra a deployment is missing rather than anything the
    organization supplied, and it is the one build failure an operator can fix.
    Today it is what `azure_content_safety` raises, because
    ``azure-ai-contentsafety`` is not among this gateway's dependencies.
    """
    return f"{type(exc).__name__}: {exc}" if isinstance(exc, ImportError) else type(exc).__name__


async def _in_a_thread(call: Callable[[], _T], *, seconds: float) -> _T:
    """Run a blocking any-guardrail call on this module's own threads, under a deadline.

    A deadline that passes abandons the answer, it does not stop the work: the
    thread runs to the end and its result is dropped. That is what the bound on
    the pool is for, and why nothing here treats a timeout as having freed a
    thread.
    """
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(_thread_pool(), call), seconds)


def _thread_pool() -> ThreadPoolExecutor:
    """This module's executor, built on first use.

    Not the process-wide default one, which `services/file_extractors` shares
    through ``asyncio.to_thread``: a burst of guardrail checks saturating it
    would stall an unrelated upload's text extraction.
    """
    global _threads  # noqa: PLW0603

    if _threads is None:
        _threads = ThreadPoolExecutor(max_workers=_MAX_THREADS, thread_name_prefix="otari-guardrail")
    return _threads


__all__ = [
    "GUARDRAIL_RUNNER_REFRESH_SECONDS",
    "GuardrailCheck",
    "OrganizationGuardrailHandle",
    "build_state",
    "handle",
    "load_guardrail_runner_at_startup",
    "refresh_guardrail_runner",
    "reset_guardrail_runner",
    "run_guardrail_runner_refresher",
    "runner_is_stale",
]
