"""One request's guardrails, run at each point of its lifecycle.

A guardrail entry says what it reads (``on``) and what it does when it fires
(``mode``). The points it can run at, and what each mode does there:

* the request's input and the tool results it carries, before the model is
  called (:meth:`RequestGuardrails.admit`);
* each call to a tool the gateway runs, and its result, inside the tool loop
  (:meth:`RequestGuardrails.before_tool_call`, :meth:`RequestGuardrails.after_tool_call`);
* the model's answer and the tool calls it hands the client, once it is complete
  (:meth:`RequestGuardrails.finish_answer`).

``monitor`` records what a guardrail found. ``block`` refuses input or an answer,
and withholds a tool result or skips a tool call so an agent can recover.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Response, status

from gateway.api.routes._guardrail_targets import AnswerPart, RequestTarget
from gateway.api.routes._helpers import GUARDRAILS_RESULT_HEADER
from gateway.core.env import otari_env
from gateway.core.error_codes import GUARDRAIL_VIOLATION, error_headers
from gateway.exceptions.guardrails_exceptions import AnswerRefused
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig, GuardrailDirection
from gateway.services.guardrails import (
    GuardrailResult,
    GuardrailsNotReachableError,
    GuardrailTarget,
    GuardrailVerdict,
    InProcessGuardrail,
    cached_result,
    check_guardrail_urls,
    check_identity,
    remember_result,
    run_input_guardrails,
    withheld_tool_result,
)
from gateway.services.tool_usage import TOOL_ERROR_SENTINEL
from gateway.services.url_safety import UnsafeURLError

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.services.traces import RequestTrace


# How many pieces of one request are checked at a time.
_CONCURRENT_CHECKS = 8


def _refusal(message: str, results: Sequence[GuardrailResult]) -> dict[str, Any]:
    return {
        "message": message,
        "code": GUARDRAIL_VIOLATION,
        "guardrails": [{"profile": r.profile, "explanation": r.explanation, "score": r.score} for r in results],
    }


def _skipped_call(profile: str, *, unchecked: bool = False) -> str:
    """What the model reads in place of a result, for a call a guardrail stopped before it ran."""
    why = "could not be checked by" if unchecked else "was flagged by"
    return (
        f"{TOOL_ERROR_SENTINEL} Otari did not run this tool call: it {why} the {profile!r} guardrail. "
        "Continue without it, or tell the user the call was refused."
    )


@dataclass
class _Content:
    """One target the guardrails read, and what happened to it."""

    target: GuardrailTarget
    new: bool
    reads: GuardrailDirection
    withhold: Callable[[str], None] | None = None
    withheld: bool = False
    refusals: list[GuardrailResult] = field(default_factory=list)

    @property
    def reported(self) -> bool:
        """Whether its verdicts belong on the header and the trace: new content, or content withheld now."""
        return self.new or self.withheld


class RequestGuardrails:
    """One request's guardrails, run over each piece of content at each point of its lifecycle.

    ``guardrails`` is the effective list: the caller's own, merged with any the
    caller's organization mandates and any a routing policy mandates (see
    :func:`gateway.api.routes._pipeline.merge_guardrail_layers`, which is where
    the merge happens so every completion endpoint enforces a mandate alike).
    ``credentials`` carries the bearer credential for the endpoint an entry
    names, keyed by profile: an organization's stored one, or the caller's own
    for a profile no layer mandates. ``mandated`` names the profiles a layer above the caller supplied,
    which is what decides whether a URL that fails its safety check is the
    caller's malformed request (a 400 naming their own URL) or a stored entry
    being unevaluable (governed by its ``mode`` / ``on_unavailable``, with the
    endpoint kept out of the response). ``in_process`` carries the guardrails
    this worker holds for the organization's own definitions, keyed by profile;
    one of those is answered here rather than sent to any service, and a profile
    whose guardrail this worker does not hold arrives as ``None`` and is
    unevaluable.

    A ``block`` entry that cannot be evaluated and fails closed
    (see :func:`gateway.services.guardrails.run_input_guardrails`) refuses
    input or an answer with a ``502`` or a ``403``, and withholds a tool result or
    skips a tool call. Every verdict is summarized per profile on the
    :data:`GUARDRAILS_RESULT_HEADER` response header and recorded on the
    request's trace. The header is set on the injected ``response``, so what is
    found after a streamed response started is on the trace only.
    """

    def __init__(
        self,
        guardrails: list[GuardrailConfig] | None,
        *,
        response: Response,
        config: GatewayConfig | None = None,
        credentials: Mapping[str, str] | None = None,
        mandated: Collection[str] | None = None,
        in_process: Mapping[str, InProcessGuardrail | None] | None = None,
        trace: RequestTrace | None = None,
    ) -> None:
        self._guardrails = guardrails or []
        self._response = response
        self._credentials = credentials or {}
        self._mandated = frozenset(mandated or ())
        self._in_process = in_process or {}
        self._trace = trace
        # Effective guardrails URL: dashboard override / config / env, falling back to
        # the env var when no config is threaded in (e.g. unit tests). A dashboard
        # override mutates config, so it hot-applies on the next request.
        self._default_url = (
            (config.guardrails_url if config is not None else None) or otari_env("GUARDRAILS_URL") or None
        )
        self._unsafe: Mapping[str, UnsafeURLError] | None = None
        self._reported: list[GuardrailResult] = []
        self._withheld: dict[str, int] = {}

    # -- what applies ----------------------------------------------------------

    def _entries(self, reads: GuardrailDirection) -> list[GuardrailConfig]:
        return [g for g in self._guardrails if reads in g.on]

    def _reads(self, *directions: GuardrailDirection) -> bool:
        return any(direction in g.on for g in self._guardrails for direction in directions)

    @property
    def reads_request(self) -> bool:
        """Whether any guardrail reads what the request carries, which is when its targets are worth building."""
        return self._reads("input", "tool_results")

    @property
    def guards_tools(self) -> bool:
        """Whether a gateway-run tool's calls or results are read."""
        return self._reads("tool_calls", "tool_results")

    @property
    def reads_answer(self) -> bool:
        """Whether a guardrail reads the answer whole, so a stream is held until it ends."""
        return self._reads("output", "tool_calls")

    # -- before the model is called --------------------------------------------

    async def admit(self, targets: Sequence[RequestTarget]) -> None:
        """Run the guardrails over the request's input and tool results, before the provider is called.

        Raises:
            HTTPException: ``400`` when a *caller-supplied* guardrail's ``url``
                fails the SSRF/scheme safety check (a mandated entry's failure is a
                502 or a recorded inconclusive instead, per its own settings);
                ``403`` when a ``block`` guardrail flags the input; ``502`` when a
                ``block`` guardrail that fails closed can't evaluate
                the input. The 502 body names the profile and not the endpoint,
                which goes to the log instead.
        """
        if not self._guardrails:
            return
        try:
            self._unsafe = await check_guardrail_urls(
                self._guardrails, credentials=self._credentials, mandated=self._mandated
            )
        except UnsafeURLError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        contents = [
            _Content(
                target=t.target,
                new=t.new,
                reads="tool_results" if t.target.event == "after_tool" else "input",
                withhold=t.withhold,
            )
            for t in targets
        ]
        await self._run(contents)
        refused = [r for c in contents if c.target.event == "before_model" for r in c.refusals]
        if refused:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_refusal("Request blocked by guardrail policy.", refused),
                headers=error_headers(GUARDRAIL_VIOLATION),
            )
        self._write_header()

    # -- inside the tool loop --------------------------------------------------

    async def before_tool_call(self, tool: str, arguments: dict[str, Any]) -> str | None:
        """What the model reads instead of a gateway-run tool call a guardrail refused, or ``None`` to run it.

        Never raises for a check that cannot run: one that fails closed refuses the call.
        """
        if not self._reads("tool_calls"):
            return None
        target = GuardrailTarget("before_tool", json.dumps(arguments), tool=tool, executor="gateway")
        content = _Content(target=target, new=True, reads="tool_calls")
        try:
            await self._run([content])
        except _Unchecked as unchecked:
            self._count_withheld(unchecked.profile)
            self._write_header()
            return _skipped_call(unchecked.profile, unchecked=True)
        if content.refusals:
            self._count_withheld(content.refusals[0].profile)
        self._write_header()
        return _skipped_call(content.refusals[0].profile) if content.refusals else None

    async def after_tool_call(self, tool: str, text: str) -> str:
        """``text``, or what the model reads in its place when a guardrail withholds it.

        For a tool the gateway runs, inside its tool loop. Never raises for a check
        that cannot run: one that fails closed withholds the result instead. A
        ``[tool error]`` result is read too, since an MCP server's own error text
        follows that prefix.
        """
        if not self._reads("tool_results") or not text:
            return text
        kept: list[str] = []
        content = _Content(
            target=GuardrailTarget("after_tool", text, tool=tool, executor="gateway"),
            new=True,
            reads="tool_results",
            withhold=kept.append,
        )
        await self._run([content])
        self._write_header()
        return kept[-1] if kept else text

    # -- once the answer is complete -------------------------------------------

    async def finish_answer(self, parts: Sequence[AnswerPart]) -> None:
        """Run the guardrails on the answer's text and the tool calls it hands the client.

        Raises:
            AnswerRefused: a ``block`` guardrail flagged the answer or one of its
                tool calls, or one that fails closed could not read them.
        """
        contents = [
            _Content(
                target=GuardrailTarget("before_tool" if p.kind == "tool_call" else "after_model", p.text, tool=p.tool),
                new=True,
                reads="tool_calls" if p.kind == "tool_call" else "output",
            )
            for p in parts
        ]
        try:
            await self._run(contents)
        except _Unchecked as unchecked:
            self._write_header()
            raise AnswerRefused(
                _refusal(
                    f"The answer could not be checked by the {unchecked.profile!r} guardrail.",
                    [GuardrailResult(profile=unchecked.profile, mode="block", valid=None)],
                )
            ) from unchecked.__cause__
        self._write_header()
        refused = [r for c in contents for r in c.refusals]
        if refused:
            raise AnswerRefused(_refusal("Answer blocked by guardrail policy.", refused))

    # -- running the guardrails ------------------------------------------------

    async def _run(self, contents: list[_Content]) -> None:
        """Every entry that reads each piece of content, over it.

        A failure that fails closed withholds a tool result (when it can be
        withheld) and raises :class:`_Unchecked` for anything else, which the
        caller answers in its own terms. :meth:`admit` answers it with a 502.
        """
        reading = [c for c in contents if self._entries(c.reads)]
        # A long history on a worker that has kept no verdicts yet would otherwise check every piece at once.
        slots = asyncio.Semaphore(_CONCURRENT_CHECKS)

        async def verdict(content: _Content) -> GuardrailVerdict:
            async with slots:
                return await self._verdict(content.target.text, self._entries(content.reads))

        started = datetime.now(UTC)
        outcomes = await asyncio.gather(*(verdict(c) for c in reading), return_exceptions=True)
        ended = datetime.now(UTC)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(outcome, GuardrailsNotReachableError):
                raise outcome
        for content, outcome in zip(reading, outcomes, strict=True):
            if isinstance(outcome, GuardrailsNotReachableError):
                self._unchecked(content, outcome, started, ended)
                continue
            assert isinstance(outcome, GuardrailVerdict)
            blocking = [r for r in outcome.results if r.flagged and r.mode == "block"]
            if blocking and content.withhold is not None:
                self._withhold(content, content.withhold, blocking[0].profile)
            elif blocking:
                content.refusals += blocking
            if content.reported:
                self._report(outcome.results, content, started, ended)

    def _unchecked(
        self, content: _Content, exc: GuardrailsNotReachableError, started: datetime, ended: datetime
    ) -> None:
        """A guardrail that fails closed could not read ``content``: withhold it, or stop."""
        profile = exc.profile or "guardrail"
        mode = next((g.mode for g in self._guardrails if g.profile == profile), "block")
        self._reported.append(GuardrailResult(profile=profile, mode=mode, valid=None, unavailable=exc.reason))
        self._record_span(profile, mode, None, content, started, ended, withheld=content.withhold is not None)
        if content.withhold is not None:
            logger.warning("guardrail check on a tool result could not be evaluated; withholding it: %s", exc)
            self._withhold(content, content.withhold, profile, unchecked=True)
            return
        # The full reason, endpoint included, goes to the log; the caller gets
        # the error's `public_detail`, which names the profile and nothing else.
        # An organization's guardrail endpoint is not the caller's to see
        # (otari#654), and it is not theirs to fix either.
        logger.warning("guardrail check could not be evaluated: %s", exc)
        if content.target.event == "before_model":
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.public_detail) from exc
        raise _Unchecked(profile) from exc

    async def _verdict(self, text: str, entries: list[GuardrailConfig]) -> GuardrailVerdict:
        """Every entry's verdict on ``text``, from this worker's kept verdicts where it has them."""
        if not entries:
            return GuardrailVerdict()
        kept: list[GuardrailResult] = []
        ask: list[GuardrailConfig] = []
        identities: dict[str, str] = {}
        for cfg in entries:
            identity = check_identity(
                cfg,
                base_url=(cfg.url or self._default_url or "").rstrip("/"),
                credential=self._credentials.get(cfg.profile),
                in_process=self._in_process,
            )
            hit = cached_result(identity, text, cfg) if identity is not None else None
            if hit is not None:
                kept.append(hit)
                continue
            ask.append(cfg)
            if identity is not None:
                identities[cfg.profile] = identity
        fresh = GuardrailVerdict()
        if ask:
            fresh = await run_input_guardrails(
                ask,
                text,
                default_url=self._default_url,
                credentials=self._credentials,
                mandated=self._mandated,
                in_process=self._in_process,
                unsafe=self._unsafe,
            )
        for result in fresh.results:
            if (identity := identities.get(result.profile)) is not None:
                remember_result(identity, text, result)
        order = {cfg.profile: i for i, cfg in enumerate(entries)}
        return GuardrailVerdict(results=sorted([*kept, *fresh.results], key=lambda r: order.get(r.profile, len(order))))

    # -- what is reported ------------------------------------------------------

    def _withhold(
        self, content: _Content, withhold: Callable[[str], None], profile: str, *, unchecked: bool = False
    ) -> None:
        """Put the notice ``profile`` leaves in place of ``content``, and count it on the header."""
        withhold(withheld_tool_result(profile, unchecked=unchecked))
        content.withheld = True
        self._count_withheld(profile)

    def _count_withheld(self, profile: str) -> None:
        self._withheld[profile] = self._withheld.get(profile, 0) + 1

    def _report(
        self, results: Sequence[GuardrailResult], content: _Content, started: datetime, ended: datetime
    ) -> None:
        # The checks on one piece of content run concurrently, so each span covers them all.
        for result in results:
            self._reported.append(result)
            self._record_span(
                result.profile,
                result.mode,
                result.valid,
                content,
                started,
                ended,
                withheld=content.withheld and result.flagged,
            )

    def _record_span(
        self,
        profile: str,
        mode: str,
        valid: bool | None,
        content: _Content,
        started: datetime,
        ended: datetime,
        *,
        withheld: bool = False,
    ) -> None:
        if self._trace is not None:
            self._trace.record_guardrail(
                profile=profile,
                mode=mode,
                valid=valid,
                started=started,
                ended=ended,
                event=content.target.event,
                tool=content.target.tool,
                withheld=withheld,
            )

    def _write_header(self) -> None:
        if not self._reported:
            return
        # One entry per profile, kept compact and free of the freeform `explanation`
        # to avoid oversized / non-ASCII headers. `status` and `reason` are fixed
        # words, so a `valid: null` still says whether the check ran.
        self._response.headers[GUARDRAILS_RESULT_HEADER] = json.dumps(
            [
                _summary(result, withheld=self._withheld.get(result.profile, 0))
                for result in _per_profile(self._reported)
            ],
            separators=(",", ":"),
        )


class _Unchecked(Exception):
    """A guardrail that fails closed could not read content that can be neither refused here nor withheld."""

    def __init__(self, profile: str) -> None:
        super().__init__(profile)
        self.profile = profile


def _severity(result: GuardrailResult) -> int:
    """How much a result says, for picking the one a profile reports: a flag, then a check that did not run."""
    if result.flagged:
        return 3
    if result.status == "unavailable":
        return 2
    return 1 if result.status == "inconclusive" else 0


def _per_profile(results: list[GuardrailResult]) -> list[GuardrailResult]:
    """Each profile's most telling result, in the order the profiles first reported."""
    chosen: dict[str, GuardrailResult] = {}
    for result in results:
        current = chosen.get(result.profile)
        if current is None or _severity(result) > _severity(current):
            chosen[result.profile] = result
    return list(chosen.values())


def _summary(result: GuardrailResult, *, withheld: int = 0) -> dict[str, object]:
    """One guardrail's entry in the :data:`GUARDRAILS_RESULT_HEADER` value."""
    summary: dict[str, object] = {
        "profile": result.profile,
        "mode": result.mode,
        "valid": result.valid,
        "score": result.score,
        "status": result.status,
    }
    if result.unavailable is not None:
        summary["reason"] = result.unavailable
    if withheld:
        summary["withheld"] = withheld
    return summary
