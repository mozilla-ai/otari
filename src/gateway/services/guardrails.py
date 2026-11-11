"""Run a request's guardrails, against the guardrails service or in this process.

A profile is served one of two ways. Most are POSTed to the guardrails service
described below. A profile an organization mandated through a definition of its
own is answered by a guardrail this worker already built and holds, handed in as
``in_process`` and declared here as :class:`InProcessGuardrail`; nothing about it
leaves the process, and nothing in this module knows how one is built.

The guardrails service is any HTTP service the operator points
``guardrails_url`` at. The only endpoint we call is::

    POST /validate  {profile, input_text, validate_kwargs}
        → {profile, result: {valid, explanation, score}}

``result.valid is False`` means the guardrail flagged the input.

A request's guardrails check each :class:`GuardrailTarget`: the caller's new
input before the model is called, and each tool result before the model reads
it, whether the client ran the tool or the gateway did inside its tool loop.
:func:`run_input_guardrails` checks one text; the routes fan it out over a
request's targets.

The service URL is operator-controlled (``OTARI_GUARDRAILS_URL``); callers may
override it per-guardrail via :attr:`GuardrailConfig.url`, which is SSRF-checked
here in :func:`run_input_guardrails` (not at parse time: the check does a DNS
lookup that must be awaited, and Pydantic validators can't await — see
:mod:`gateway.models.guardrails`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from gateway.models.guardrails import GuardrailConfig
from gateway.services.tool_usage import TOOL_ERROR_SENTINEL
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0

GuardrailEvent = Literal["before_model", "after_tool", "before_tool", "after_model"]
"""Where a guardrail runs: on what the model is about to read, a tool result, a tool call, or the answer."""

ToolExecutor = Literal["client", "gateway"]


@dataclass(frozen=True)
class GuardrailTarget:
    """One piece of content a request's guardrails check, and where it came from.

    A ``block`` flag on ``before_model`` content refuses the request. A flag on
    an ``after_tool`` result withholds that result instead, so the agent can
    carry on without it. ``executor`` says who ran the tool: the client, whose
    results arrive in the request, or the gateway, inside its tool loop.
    """

    event: GuardrailEvent
    text: str
    tool: str | None = None
    executor: ToolExecutor = "client"


def withheld_tool_result(profile: str, *, unchecked: bool = False) -> str:
    """What the model reads in place of a tool result a guardrail withheld.

    The same words for the same profile every time: a client resends the
    original on its next request, and an identical replacement keeps the
    provider's prompt cache matching. It reads as a tool error, so a gateway
    tool's activity reports the call as failed.
    """
    why = "could not be checked by" if unchecked else "was flagged by"
    return (
        f"{TOOL_ERROR_SENTINEL} This tool result was withheld by Otari: it {why} the {profile!r} guardrail. "
        "Continue without it, or tell the user it was withheld."
    )


UnavailableReason = Literal["not_configured", "evaluation_failed"]
"""Why a guardrail could not run, in words safe to send back to the caller.

``not_configured`` means no guardrails service was named for the profile.
``evaluation_failed`` covers every other failure, and the log holds the detail.
"""

GuardrailStatus = Literal["evaluated", "inconclusive", "unavailable"]


def _unevaluated_detail(profile: str) -> str:
    """What a caller is told when a guardrail could not run.

    Names the profile, which the caller either asked for or is subject to, and
    nothing about where it would have run or what the endpoint said back.
    """
    return f"guardrail profile {profile!r} could not be evaluated"


class GuardrailsNotReachableError(RuntimeError):
    """Raised when the guardrails service can't be reached or returns malformed data.

    Carries two messages, because the two audiences are different. ``str(exc)``
    is for the log and names the endpoint and the underlying failure;
    :attr:`public_detail` is what reaches the caller in the 502 body and
    deliberately does not. Since otari#654 that endpoint may be one an
    organization configured, which the caller was never told about and cannot
    act on, and the root ``AGENTS.md`` rule against leaking internals in a
    public error response covers exactly that. The one message that stays whole
    is the no-URL-configured case: it names an environment variable rather than
    an address, and it is the only one a reader can actually act on.

    Neither message quotes what the endpoint sent back, because an answer can
    echo the prompt. A bad answer is named by its HTTP status, its exception
    class, or the type of the bad value.
    """

    def __init__(
        self,
        message: str,
        *,
        public_detail: str | None = None,
        reason: UnavailableReason = "evaluation_failed",
    ) -> None:
        super().__init__(message)
        self.public_detail = public_detail if public_detail is not None else message
        self.reason: UnavailableReason = reason
        # The profile that failed closed, set by `run_input_guardrails` as it raises.
        self.profile: str | None = None


@dataclass
class GuardrailResult:
    """Outcome of one guardrail check."""

    profile: str
    mode: str
    valid: bool | None
    explanation: object | None = None
    score: object | None = None
    unavailable: UnavailableReason | None = None
    """Set when the check could not run and the request was served anyway."""

    @property
    def status(self) -> GuardrailStatus:
        """Whether the check ran, ran without a verdict, or could not run."""
        if self.unavailable is not None:
            return "unavailable"
        return "inconclusive" if self.valid is None else "evaluated"

    @property
    def flagged(self) -> bool:
        """A guardrail flags the input when it explicitly reports ``valid=False``.

        ``valid is None`` is no verdict rather than a flag; ``on_unavailable``
        decides whether the request may go on without one.
        """
        return self.valid is False


@dataclass
class GuardrailVerdict:
    """Aggregate of all input-direction guardrail results for a request."""

    results: list[GuardrailResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True when any ``mode="block"`` guardrail flagged the input."""
        return any(r.flagged and r.mode == "block" for r in self.results)

    @property
    def flagged(self) -> list[GuardrailResult]:
        """All flagged results, regardless of mode (for monitor annotation/logging)."""
        return [r for r in self.results if r.flagged]


class InProcessVerdict(Protocol):
    """What a guardrail this process runs answers with.

    The same three fields the guardrails service reports, so both backends reach
    :class:`GuardrailResult` the same way. Read-only properties rather than
    attributes, so a frozen value type satisfies this by shape.
    """

    @property
    def valid(self) -> bool: ...

    @property
    def explanation(self) -> str | None: ...

    @property
    def score(self) -> float | None: ...


class InProcessGuardrail(Protocol):
    """A guardrail this worker holds, built from an organization's own definition.

    Declared here by shape rather than imported, so this module keeps knowing
    nothing about the one that builds and holds them
    (`services/tenancy/organization_guardrail_runner.py`) and the dependency
    points one way.

    The contract is narrow deliberately: `check` answers a verdict or raises
    :class:`GuardrailsNotReachableError`. That is what lets an in-process
    guardrail go through the same ``mode`` and ``on_unavailable`` handling as a
    remote one instead of needing failure arms of its own.
    """

    async def check(self, prompt: str, **validate_kwargs: Any) -> InProcessVerdict: ...


async def _validate_one(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    cfg: GuardrailConfig,
    input_text: str,
    credential: str | None = None,
) -> GuardrailResult:
    payload: dict[str, object] = {"profile": cfg.profile, "input_text": input_text}
    if cfg.validate_kwargs:
        payload["validate_kwargs"] = cfg.validate_kwargs
    # An organization guardrail, or a caller's entry with its own `url`, may
    # carry a credential for the endpoint it names. It goes in the header rather
    # than the body: `/validate` forbids
    # unknown body fields, and a guardrail's *vendor* key is not this in any case;
    # the guardrails service constructs its guardrails from the operator's own
    # config and holds those itself. Never logged, here or by the caller.
    headers = {"Authorization": f"Bearer {credential}"} if credential else None
    try:
        response = await client.post(f"{base_url}/validate", json=payload, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} failed against {base_url}: HTTP {exc.response.status_code}",
            public_detail=_unevaluated_detail(cfg.profile),
        ) from exc
    except (httpx.ProtocolError, httpx.DecodingError) as exc:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} failed against {base_url}: {type(exc).__name__}",
            public_detail=_unevaluated_detail(cfg.profile),
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} failed against {base_url}: {exc}",
            public_detail=_unevaluated_detail(cfg.profile),
        ) from exc
    try:
        result = response.json()["result"]
    except (KeyError, TypeError, ValueError) as exc:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned no readable 'result': {type(exc).__name__}",
            public_detail=_unevaluated_detail(cfg.profile),
        ) from exc

    # `result` may be a list when the service runs the guardrail over a list of
    # inputs; we only ever send a single string, so unwrap the common case.
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned an unexpected result shape: {type(result).__name__}",
            public_detail=_unevaluated_detail(cfg.profile),
        )

    # Treat a missing or non-boolean `valid` as malformed and raise, so the
    # mode-aware handling in run_input_guardrails applies (block fails closed,
    # monitor fails open) rather than silently passing a `block` guardrail. An
    # explicit `valid: null` is a legitimate inconclusive verdict (not flagged).
    if "valid" not in result:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned no 'valid' field",
            public_detail=_unevaluated_detail(cfg.profile),
        )
    valid = result["valid"]
    if valid is not None and not isinstance(valid, bool):
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned a non-boolean 'valid': {type(valid).__name__}",
            public_detail=_unevaluated_detail(cfg.profile),
        )

    return GuardrailResult(
        profile=cfg.profile,
        mode=cfg.mode,
        valid=valid,
        explanation=result.get("explanation"),
        score=result.get("score"),
    )


async def _check_in_process(
    guardrail: InProcessGuardrail | None,
    *,
    cfg: GuardrailConfig,
    input_text: str,
) -> GuardrailResult:
    """Run a guardrail this worker holds, instead of posting the check anywhere.

    ``None`` is a profile whose definition this worker holds nothing for:
    disabled, deleted, or one that failed to build. It is unevaluable and says
    so, which matters more than it looks. Falling through to the remote path
    instead would send the profile to the deployment's guardrails service, which
    has never heard of it, so a failed build would arrive as somebody else's
    error and a fail-open entry would serve the request unchecked.

    The validate kwargs are the organization's own. `_overlay_mandate` gives a
    mandated profile's whole config to the mandating entry, so a caller cannot
    add an argument to a check they are subject to.
    """
    if guardrail is None:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} names a definition this worker does not hold",
            public_detail=_unevaluated_detail(cfg.profile),
        )
    try:
        verdict = await guardrail.check(input_text, **cfg.validate_kwargs)
    except GuardrailsNotReachableError as exc:
        # Its message names the definition and the type of what went wrong, and
        # its public detail names neither, because the runner does not know the
        # profile. Naming it is this end's job, and it is all the caller is told.
        raise GuardrailsNotReachableError(str(exc), public_detail=_unevaluated_detail(cfg.profile)) from exc
    return GuardrailResult(
        profile=cfg.profile,
        mode=cfg.mode,
        valid=verdict.valid,
        explanation=verdict.explanation,
        score=verdict.score,
    )


async def check_guardrail_urls(
    guardrails: list[GuardrailConfig],
    *,
    credentials: Mapping[str, str] | None = None,
    mandated: Collection[str] | None = None,
) -> dict[str, UnsafeURLError]:
    """SSRF-check every entry's ``url``, returning the mandated profiles whose URL failed.

    A caller's own URL that fails raises :class:`UnsafeURLError`, which is a
    malformed request. A mandated entry's failure is returned instead, for
    :func:`run_input_guardrails` to treat as that profile being unevaluable.
    Separate so a request checked over several targets resolves each URL once.

    Raises:
        UnsafeURLError: a caller-supplied ``url`` failed the check.
    """
    credentials = credentials or {}
    mandated = frozenset(mandated or ())
    unsafe: dict[str, UnsafeURLError] = {}
    # Paired with its URL rather than filtered in place, so what reaches
    # `validate_mcp_url` is a `str` and not a `str | None` narrowed by
    # inspection.
    checked = [(g.profile, g.url) for g in guardrails if g.url is not None]
    outcomes = await asyncio.gather(
        *(validate_mcp_url(url, has_authorization_token=bool(credentials.get(profile))) for profile, url in checked),
        return_exceptions=True,
    )
    for (profile, _), outcome in zip(checked, outcomes, strict=True):
        if not isinstance(outcome, BaseException):
            continue
        if not isinstance(outcome, UnsafeURLError) or profile not in mandated:
            raise outcome
        unsafe[profile] = outcome
    return unsafe


async def run_input_guardrails(
    guardrails: list[GuardrailConfig],
    input_text: str,
    *,
    default_url: str | None,
    credentials: Mapping[str, str] | None = None,
    mandated: Collection[str] | None = None,
    in_process: Mapping[str, InProcessGuardrail | None] | None = None,
    allow_no_verdict: bool = False,
    unsafe: Mapping[str, UnsafeURLError] | None = None,
) -> GuardrailVerdict:
    """Run every guardrail in ``guardrails`` over ``input_text`` and return the aggregate verdict.

    Which entries apply to a piece of content (its ``on``) is the caller's
    choice, made before this is called; every entry handed in is evaluated.

    ``mandated`` names the profiles that came from a layer above the caller (an
    organization entry or a routing policy) rather than from the request body.
    It decides what an unsafe or unresolvable URL *means*, which is not the same
    question for the two sources: a caller who sent a bad URL sent a malformed
    request and is told so, while a mandated entry's URL is stored
    configuration the caller cannot see or fix, so failing to validate it is the
    guardrail being unevaluable and is governed by ``mode`` and
    ``on_unavailable`` like any other evaluation failure. Without this a
    transient DNS failure on an organization's endpoint refused every request
    from every workspace it was scoped to, including entries explicitly
    configured to fail open.

    ``credentials`` maps a profile name to the bearer credential its entry
    carries: an organization's stored one
    (`services/tenancy/organization_guardrail_service.py`), or the caller's own
    for a profile no layer mandates. It is a separate argument rather than read
    off :class:`GuardrailConfig` because the merge decides which credential
    belongs to the URL that survived it, and no stored secret may follow a
    caller's entry.

    ``in_process`` maps a profile to a guardrail this worker already holds,
    built from the organization's own definition
    (`services/tenancy/organization_guardrail_runner.py`). A profile in it is
    answered here and never sent anywhere, so its entry has no URL and reaches
    neither the safety check above nor the ``default_url`` fallback below. A
    value of ``None`` is a profile whose definition this worker holds nothing
    for, and is unevaluable rather than a reason to try the remote path.

    Each guardrail's URL is ``cfg.url or default_url``, and is SSRF-checked up
    front (see :func:`check_guardrail_urls`) unless ``unsafe`` says it already
    was. A ``mandated`` entry's failure is not an outright rejection: it is held
    until the per-entry loop below reaches that profile.

    Failure handling depends on the guardrail's ``mode`` and its
    ``on_unavailable``:

    * ``block`` guardrails with ``on_unavailable="block"`` (the
      default) **fail closed** — if one can't be evaluated (service unreachable, no URL
      configured, malformed response) we raise
      :class:`GuardrailsNotReachableError` (the caller surfaces it as a 502)
      rather than let an unchecked request through.
    * ``block`` guardrails with ``on_unavailable="monitor"`` trade enforcement for
      availability: the check is recorded as inconclusive and the request
      proceeds, the same as a ``monitor`` guardrail. Only reachability is affected;
      a guardrail that runs and flags still blocks.
    * ``monitor`` guardrails **fail open** — they're observe-only, so an
      evaluation error is logged and recorded as an inconclusive result
      (``valid=None``) and the request proceeds; we don't 502 a request the
      caller explicitly asked us not to enforce.

    A check that runs and answers no verdict (``valid=None``) was not evaluated
    either, so it takes the same paths. ``allow_no_verdict`` returns that answer
    instead, for a caller that runs a check only to show what the service says.

    ``unsafe`` is :func:`check_guardrail_urls`'s answer when the caller already
    ran it, which skips the check here.

    Returns:
        A :class:`GuardrailVerdict` aggregating every input guardrail's result.

    Raises:
        GuardrailsNotReachableError: if a ``block``-mode guardrail with
            ``on_unavailable="block"`` can't be evaluated.
    """
    # A `url` override is SSRF-checked here rather than at request-body-parse
    # time (the check does a DNS lookup that must be awaited).
    #
    # What a failure *means* then depends on where the URL came from.
    # `return_exceptions=True` is what lets that be decided per entry rather
    # than by whichever check happened to fail first:
    #
    # * A caller's own URL is a malformed request, so it rejects the request
    #   mode-independently and the reason travels back to them. That is the
    #   pre-existing behavior and the message names a host they supplied.
    # * A *mandated* entry's URL is stored configuration the caller can neither
    #   see nor fix, so the same failure is the guardrail being unevaluable and
    #   goes through the `mode` / `on_unavailable` handling below. Its message
    #   names an endpoint an organization configured, so it stays out of the
    #   response for the same reason the 502 path keeps it out (otari#654).
    #
    # Either way the unsafe endpoint is never actually called.
    credentials = credentials or {}
    in_process = in_process or {}
    if unsafe is None:
        unsafe = await check_guardrail_urls(guardrails, credentials=credentials, mandated=mandated)

    if not guardrails:
        return GuardrailVerdict()

    results: list[GuardrailResult] = []
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S) as client:
        for cfg in guardrails:
            fails_closed = cfg.mode == "block" and cfg.on_unavailable == "block"
            base_url = (cfg.url or default_url or "").rstrip("/")
            try:
                if (unsafe_url := unsafe.get(cfg.profile)) is not None:
                    raise GuardrailsNotReachableError(
                        f"guardrail profile {cfg.profile!r} names an endpoint that failed the "
                        f"safety check: {unsafe_url}",
                        public_detail=_unevaluated_detail(cfg.profile),
                    )
                if cfg.profile in in_process:
                    result = await _check_in_process(in_process[cfg.profile], cfg=cfg, input_text=input_text)
                elif not base_url:
                    raise GuardrailsNotReachableError(
                        f"guardrail profile {cfg.profile!r} requested but no guardrails service is "
                        "configured. Set OTARI_GUARDRAILS_URL on the gateway or pass `url` on the "
                        "guardrail entry.",
                        reason="not_configured",
                    )
                else:
                    result = await _validate_one(
                        client,
                        base_url=base_url,
                        cfg=cfg,
                        input_text=input_text,
                        credential=credentials.get(cfg.profile),
                    )
            except GuardrailsNotReachableError as exc:
                if fails_closed:
                    exc.profile = cfg.profile
                    raise  # fail closed: an enforcing guardrail must not be skipped
                # The reason belongs here and nowhere else: the fail-closed arm
                # above hands its message to `RequestGuardrails`, which logs
                # it, but this arm serves the request, so this line is the only
                # record that a check an organization mandated did not run. It
                # names the endpoint for the same reason that one does, and for
                # the same audience.
                logger.warning(
                    "guardrail %r could not be evaluated (mode=%s on_unavailable=%s); failing open: %s",
                    cfg.profile,
                    cfg.mode,
                    cfg.on_unavailable,
                    exc,
                )
                results.append(
                    GuardrailResult(
                        profile=cfg.profile,
                        mode=cfg.mode,
                        valid=None,
                        explanation="guardrail service unavailable",
                        unavailable=exc.reason,
                    )
                )
                continue
            if result.valid is None and fails_closed and not allow_no_verdict:
                no_verdict = GuardrailsNotReachableError(
                    f"guardrail profile {cfg.profile!r} returned no verdict",
                    public_detail=_unevaluated_detail(cfg.profile),
                )
                no_verdict.profile = cfg.profile
                raise no_verdict
            if result.flagged:
                logger.info(
                    "guardrail flagged input: profile=%s mode=%s score=%s",
                    result.profile,
                    result.mode,
                    result.score,
                )
            results.append(result)

    return GuardrailVerdict(results=results)


# A client resends every tool result it holds on each request of an agent loop,
# and a withheld one has to be withheld again every time, so each request checks
# all of them. Keeping the verdicts makes a repeat cost a lookup rather than a
# vendor call. Per worker: another worker checks the same text once itself.
_VERDICT_CACHE_SIZE = 4096
_VERDICT_CACHE_TTL_S = 3600.0


@dataclass(frozen=True)
class _CachedVerdict:
    valid: bool | None
    explanation: object | None
    score: object | None
    expires_at: float


_verdicts: OrderedDict[tuple[str, str], _CachedVerdict] = OrderedDict()


def check_identity(
    cfg: GuardrailConfig,
    *,
    base_url: str,
    credential: str | None,
    in_process: Mapping[str, InProcessGuardrail | None],
) -> str | None:
    """What decides one check's verdict on a given text, or ``None`` when it cannot be told apart.

    A remote check is its profile, endpoint, credential and arguments, so two
    organizations reach the same verdict only by asking the same service the
    same question. An in-process check is the version of the definition it was
    built from and its arguments, and ``None`` for a guardrail that does not say.
    """
    if cfg.profile in in_process:
        identity = getattr(in_process[cfg.profile], "identity", None)
        return None if identity is None else json.dumps([identity, cfg.validate_kwargs], sort_keys=True, default=str)
    if not base_url:
        return None
    secret = hashlib.sha256((credential or "").encode()).hexdigest()
    return json.dumps([cfg.profile, base_url, secret, cfg.validate_kwargs], sort_keys=True, default=str)


def cached_result(identity: str, text: str, cfg: GuardrailConfig) -> GuardrailResult | None:
    """The verdict this worker already holds for ``identity`` on ``text``, under ``cfg``'s mode."""
    key = (identity, hashlib.sha256(text.encode()).hexdigest())
    entry = _verdicts.get(key)
    if entry is None:
        return None
    if entry.expires_at <= time.monotonic():
        del _verdicts[key]
        return None
    _verdicts.move_to_end(key)
    return GuardrailResult(
        profile=cfg.profile,
        mode=cfg.mode,
        valid=entry.valid,
        explanation=entry.explanation,
        score=entry.score,
    )


def remember_result(identity: str, text: str, result: GuardrailResult) -> None:
    """Keep an evaluated verdict. One without a verdict is asked again next time."""
    if result.status != "evaluated":
        return
    key = (identity, hashlib.sha256(text.encode()).hexdigest())
    _verdicts[key] = _CachedVerdict(
        valid=result.valid,
        explanation=result.explanation,
        score=result.score,
        expires_at=time.monotonic() + _VERDICT_CACHE_TTL_S,
    )
    _verdicts.move_to_end(key)
    while len(_verdicts) > _VERDICT_CACHE_SIZE:
        _verdicts.popitem(last=False)


def clear_verdict_cache() -> None:
    """Drop every kept verdict."""
    _verdicts.clear()
