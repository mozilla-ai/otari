"""The guardrails this deployment has defined, built and ready to run.

`services/guardrails.py` sends a profile to an operator-run container over
``POST /validate``. That container holds the guardrails, built from its own YAML
at boot, which is why a profile there is a name this repository cannot describe.
A stored definition can be described, so this module builds one here and calls it
here.

It holds them rather than making them. Building happens in two places, neither of
them a request: the startup pass in ``services/guardrail_loader.py``, and the
write that changed a definition. So a check is a lookup and a call, a profile
nobody built is simply not available, and none of the machinery a lazy cache needs
is here: no key over the arguments, no in-flight table, no shield around a build a
request is waiting on.

What that leaves is shaped by three facts about ``any_guardrail``:

* ``create`` and ``validate`` are both plain synchronous ``def``. Both are
  offloaded to the process-wide default executor, shared with file extraction and
  OCR (``services/file_extractors.py``). Running either on the event loop would
  freeze every concurrent request for as long as it took.
* A thread cannot be cancelled. A build that outlives its deadline runs to
  completion and its result is dropped. Nothing is waiting on it, which is exactly
  why this can be that simple.
* Every failure becomes :class:`GuardrailsNotReachableError`, so the fail-open and
  fail-closed handling in ``run_input_guardrails`` governs a guardrail built here
  exactly as it governs a remote one, and the caller is told only the profile name.

Only a guardrail :func:`runs_in_process` accepts is built, which means every one
of them is a vendor client and every check is an HTTP request. That is what lets
concurrent checks share one built object with no lock of their own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from any_guardrail import AnyGuardrail, EvaluateArgumentError, Guardrail
from any_guardrail.base import GuardrailName

from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrail_catalog import runs_in_process
from gateway.services.guardrails import (
    GUARDRAIL_TIMEOUT_S,
    GuardrailResult,
    GuardrailsNotReachableError,
    unevaluated_detail,
)
from gateway.types.guardrail_definition import GuardrailDefinition

# Sentinel, because a guardrail reporting no verdict at all and one reporting an
# explicit `None` are different failures: the first is malformed, the second is a
# legitimate inconclusive result that must not block.
_NO_VERDICT = object()


@dataclass(frozen=True)
class _Ready:
    """One built guardrail, with the per-call arguments its definition carried."""

    name: GuardrailName
    guardrail: Guardrail
    validate_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class LoadOutcome:
    """How a whole pass went, for the one line its caller logs."""

    built: int
    failed: int


def _unavailable(profile: str, message: str) -> GuardrailsNotReachableError:
    """The one error this module raises, with the log half and the public half apart.

    ``message`` names the profile, the guardrail class and a failure's type. It
    never carries a third-party exception's text, because a vendor SDK echoes the
    arguments it was handed and those hold the API key.
    """
    return GuardrailsNotReachableError(message, public_detail=unevaluated_detail(profile))


def _resolve(profile: str, definition: GuardrailDefinition) -> GuardrailName:
    """The upstream class to build, refusing one this gateway must not run itself.

    Checked here as well as at the store, because a row written before that rule
    existed is still in the database, and building it is what the rule prevents.
    """
    if not runs_in_process(definition.guardrail_name):
        raise _unavailable(
            profile,
            f"guardrail profile {profile!r} names {definition.guardrail_name!r}, "
            "which this gateway does not run in its own process",
        )
    return GuardrailName(definition.guardrail_name)


def _verdict(output: object, cfg: GuardrailConfig) -> GuardrailResult:
    """Map an ``any_guardrail`` output onto the result the request path reads.

    Typed as ``object`` rather than ``GuardrailOutput`` because the shape checks
    are real: this is a third-party return value under a ``>=0.7.7,<0.8.0`` floor,
    and the same checks the HTTP path makes on a response body apply to it.
    ``categories``, ``spans`` and ``usage`` are dropped; ``GuardrailResult`` has no
    home for them.
    """
    if isinstance(output, list):
        if not output:
            raise _unavailable(cfg.profile, f"guardrail profile {cfg.profile!r} returned an empty result list")
        output = output[0]

    valid = getattr(output, "valid", _NO_VERDICT)
    if valid is _NO_VERDICT:
        raise _unavailable(cfg.profile, f"guardrail profile {cfg.profile!r} returned no verdict")
    if valid is not None and not isinstance(valid, bool):
        raise _unavailable(cfg.profile, f"guardrail profile {cfg.profile!r} returned a non-boolean verdict")

    return GuardrailResult(
        profile=cfg.profile,
        mode=cfg.mode,
        valid=valid,
        explanation=getattr(output, "explanation", None),
        score=getattr(output, "score", None),
    )


class GuardrailRunner:
    """Holds one built guardrail per profile, and runs any of them against text.

    One instance serves the process. Create it from inside a running event loop,
    not at import: it holds an ``asyncio.Lock``, which binds to the loop that first
    uses it, so an instance built at import would break under a second loop.
    """

    def __init__(self, *, timeout_s: float = GUARDRAIL_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._ready: dict[str, _Ready] = {}
        # Held across a whole build, not just the assignment that ends it. The
        # assignments are already atomic between awaits; what needs serializing is
        # the seconds of building in front of them, or a write that lands mid-pass
        # is undone by the swap of definitions that pass read before it.
        self._lock = asyncio.Lock()
        # Profiles deleted while a pass was building. A delete cannot wait for the
        # lock, because it answers a request, so the pass reconciles instead.
        self._dropped: set[str] = set()
        self._pass_in_flight = False

    def knows(self, profile: str) -> bool:
        """Whether ``profile`` is built and can be checked against."""
        return profile in self._ready

    def profiles(self) -> frozenset[str]:
        """Every profile this runner holds."""
        return frozenset(self._ready)

    async def load(self, definitions: Mapping[str, GuardrailDefinition]) -> LoadOutcome:
        """Build this whole set and make it what the runner holds.

        Replaces rather than merges, so a profile the deployment no longer defines
        stops answering. Built into a fresh map and swapped in one step, because
        filling the live one would leave a window where a lookup misses a profile
        that exists.

        A definition that will not build is counted and skipped; the rest are still
        built, and nothing is raised. The caller logs the outcome.

        A delete that lands while this is building wins over it, because the
        definitions this read are older than that deletion. A write that lands
        waits for the lock and then applies on top, for the same reason.
        """
        built: dict[str, _Ready] = {}
        failed = 0
        async with self._lock:
            self._dropped.clear()
            self._pass_in_flight = True
            try:
                for profile, definition in sorted(definitions.items()):
                    try:
                        built[profile] = await self._build(profile, definition)
                    except GuardrailsNotReachableError as exc:
                        logger.warning("Guardrail %r was not built: %s", profile, exc)
                        failed += 1
            finally:
                self._pass_in_flight = False
            for profile in self._dropped:
                built.pop(profile, None)
            self._dropped.clear()
            self._ready = built
        return LoadOutcome(built=len(built), failed=failed)

    async def load_one(self, profile: str, definition: GuardrailDefinition) -> None:
        """Build one definition and make it what ``profile`` means from now on.

        A failure drops whatever was held instead of leaving it in place. The old
        definition is gone from the store by the time this runs, so serving from
        the object built out of it would enforce a rule the operator deleted.
        Unavailable is the safer half of that choice, and ``on_unavailable`` is
        already the knob for deciding what unavailable costs.

        Raises :class:`GuardrailsNotReachableError`, so a caller warming many
        profiles decides what one failure means for the rest.

        Waits for a pass in flight rather than racing it, so the definition this
        was given, which is newer, is not replaced by the set that pass read.
        """
        async with self._lock:
            try:
                ready = await self._build(profile, definition)
            except GuardrailsNotReachableError:
                self._ready.pop(profile, None)
                raise
            self._ready[profile] = ready

    def drop(self, profile: str) -> None:
        """Forget ``profile``, so a check against it reports unavailable.

        Synchronous and lockless, because it answers a delete and must not wait on
        a build. Recorded when a pass is in flight so that pass drops it too rather
        than restoring it from the definitions it read before the deletion.
        """
        self._ready.pop(profile, None)
        if self._pass_in_flight:
            self._dropped.add(profile)

    async def check(self, *, cfg: GuardrailConfig, input_text: str) -> GuardrailResult:
        """Run ``input_text`` past the guardrail ``cfg.profile`` names.

        No building here and no fallback to one. A profile this runner does not
        hold was never defined, was defined as something this gateway does not run,
        or failed to build, and all three are the same answer to a request.
        """
        ready = self._ready.get(cfg.profile)
        if ready is None:
            raise _unavailable(cfg.profile, f"guardrail profile {cfg.profile!r} is not built on this gateway")
        return await self._run(ready, cfg, input_text)

    async def probe(
        self, *, definition: GuardrailDefinition, cfg: GuardrailConfig, input_text: str
    ) -> GuardrailResult:
        """Build ``definition`` and check ``input_text`` against it, registering nothing.

        What answers "does this definition work" without making it the answer to
        anything else. A definition that is disabled, or that the last pass could
        not build, is as testable as any other, and finding out must not put either
        one in front of live traffic.
        """
        ready = await self._build(cfg.profile, definition)
        return await self._run(ready, cfg, input_text)

    async def _run(self, ready: _Ready, cfg: GuardrailConfig, input_text: str) -> GuardrailResult:
        """Check ``input_text`` against one built guardrail, however it was obtained."""
        # The caller's arguments win, matching the sidecar's documented contract.
        # A mandated profile's entry already carries the operator's, because
        # `_overlay_mandate` replaced the caller's before the request got here.
        kwargs = {**ready.validate_kwargs, **cfg.validate_kwargs}
        return _verdict(await self._evaluate(ready, cfg, input_text, kwargs), cfg)

    async def _build(self, profile: str, definition: GuardrailDefinition) -> _Ready:
        """Construct the guardrail on a worker thread, bounded by the same deadline a check gets.

        Offloaded because ``create`` imports the vendor SDK. On a timeout the
        thread runs on and what it returns is dropped, which costs nothing: no
        request is waiting for it, and the next pass or write will build again.
        """
        name = _resolve(profile, definition)
        try:
            guardrail = await asyncio.wait_for(
                asyncio.to_thread(AnyGuardrail.create, name, **definition.create_kwargs),
                self._timeout_s,
            )
        except TimeoutError as exc:
            raise _unavailable(
                profile, f"guardrail profile {profile!r} ({name.value}) did not build within {self._timeout_s}s"
            ) from exc
        except ImportError as exc:
            raise self._missing_package(profile, name, exc) from exc
        except Exception as exc:
            # Broad on purpose, against the usual rule: a vendor SDK's constructor
            # raises whatever it likes, and none of it may reach a caller as a 500.
            raise _unavailable(
                profile,
                f"guardrail profile {profile!r} ({name.value}) failed to build: {type(exc).__name__}",
            ) from exc

        return _Ready(name=name, guardrail=guardrail, validate_kwargs=dict(definition.validate_kwargs))

    async def _evaluate(
        self, ready: _Ready, cfg: GuardrailConfig, input_text: str, kwargs: dict[str, Any]
    ) -> object:
        """Call the guardrail on a worker thread, and turn any failure into an unavailable verdict.

        ``AnyGuardrail.evaluate`` rather than ``guardrail.validate``: the classes do
        not share one signature, and upstream ships a per-guardrail builder table
        for exactly that. Concurrent checks share the one built object without a
        lock, because every guardrail here is a vendor client making an HTTP call.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(AnyGuardrail.evaluate, ready.name, ready.guardrail, input_text, **kwargs),
                self._timeout_s,
            )
        except TimeoutError as exc:
            raise _unavailable(
                cfg.profile,
                f"guardrail profile {cfg.profile!r} ({ready.name.value}) did not answer within {self._timeout_s}s",
            ) from exc
        except EvaluateArgumentError as exc:
            # The one third-party message carried whole. Upstream builds it from
            # argument *names* and never their values, and it is the only text that
            # tells an operator which field they left out.
            raise _unavailable(
                cfg.profile, f"guardrail profile {cfg.profile!r} ({ready.name.value}) was called wrongly: {exc}"
            ) from exc
        except Exception as exc:
            raise _unavailable(
                cfg.profile,
                f"guardrail profile {cfg.profile!r} ({ready.name.value}) failed: {type(exc).__name__}",
            ) from exc

    def _missing_package(self, profile: str, name: GuardrailName, exc: ImportError) -> GuardrailsNotReachableError:
        """Tell an operator which package a guardrail wanted, when that is the failure.

        Upstream re-raises a gated import as ``raise ImportError(msg) from e`` and
        treats a chained cause as its missing-package signal, so an uncaused one is
        a real bug rather than an uninstalled package and is not reported as one.

        Its message is the second exception to the no-third-party-text rule, and
        for the same reason as the first: it is a fixed template over a package
        name, with no argument of the operator's in it. Otari declares no extra
        that would supply these, so upstream's text is the only actionable string
        there is.
        """
        if exc.__cause__ is None:
            return _unavailable(profile, f"guardrail profile {profile!r} ({name.value}) could not be imported")
        return _unavailable(
            profile, f"guardrail profile {profile!r} ({name.value}) is missing a package: {exc}"
        )


# The one runner the process uses, and the one a store write must reach to rebuild
# what it changed. Created on first use rather than at import, because the class
# holds an `asyncio.Lock` that binds to the loop first touching it: an instance
# built at import would outlive a lifespan restart and fail from inside asyncio
# under the next loop. The pooled search client has the same shape for the same
# reason.
_runner: GuardrailRunner | None = None


def get_guardrail_runner() -> GuardrailRunner:
    """The process-wide runner, built on the first call from a running loop."""
    global _runner  # noqa: PLW0603

    if _runner is None:
        _runner = GuardrailRunner()
    return _runner


def reset_guardrail_runner() -> None:
    """Drop the runner and everything it has built (shutdown, tests).

    Unconditional at shutdown rather than gated on the startup pass having run: a
    store write builds a runner in a deployment that had no definitions at boot.
    A no-op when nothing ever built one.
    """
    global _runner  # noqa: PLW0603

    _runner = None
