"""Run caller-requested guardrails against the guardrails service.

The guardrails service (``otari-anyguardrails-container``) wraps
`any-guardrail <https://github.com/mozilla-ai/any-guardrail>`_ behind a small
HTTP API. The only endpoint we call is::

    POST /validate  {profile, input_text, validate_kwargs}
        → {profile, result: {valid, explanation, score}}

``result.valid is False`` means the guardrail flagged the input (e.g. the
``prompt-injection`` profile — Deepset in-process, or DuoGuard via an
encoderfile — detected a prompt injection).

Unlike the sandbox / web-search backends, this does **not** duck-type the MCP
tool-loop ``pool`` protocol: guardrails never enter the tool loop. It is a flat
pre-provider interceptor — see :func:`run_input_guardrails`, which the three
route handlers call right after auth and before dispatching to the provider.

The service URL is operator-controlled (``OTARI_GUARDRAILS_URL``); callers may
override it per-guardrail via :attr:`GuardrailConfig.url`, which is SSRF-checked
at parse time (see :mod:`gateway.models.guardrails`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from gateway.models.guardrails import GuardrailConfig

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0


class GuardrailsNotReachableError(RuntimeError):
    """Raised when the guardrails service can't be reached or returns malformed data."""


@dataclass
class GuardrailResult:
    """Outcome of one guardrail check."""

    profile: str
    mode: str
    valid: bool | None
    explanation: object | None = None
    score: object | None = None

    @property
    def flagged(self) -> bool:
        """A guardrail flags the input when it explicitly reports ``valid=False``.

        ``valid is None`` (guardrail returned no verdict) is treated as *not
        flagged* — we don't block on an inconclusive result.
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


async def _validate_one(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    cfg: GuardrailConfig,
    input_text: str,
) -> GuardrailResult:
    payload: dict[str, object] = {"profile": cfg.profile, "input_text": input_text}
    if cfg.validate_kwargs:
        payload["validate_kwargs"] = cfg.validate_kwargs
    try:
        response = await client.post(f"{base_url}/validate", json=payload)
        response.raise_for_status()
        body = response.json()
        result = body["result"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} failed against {base_url}: {exc}"
        ) from exc

    # `result` may be a list when the service runs the guardrail over a list of
    # inputs; we only ever send a single string, so unwrap the common case.
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned an unexpected result shape: {result!r}"
        )

    # Treat a missing or non-boolean `valid` as malformed and raise, so the
    # mode-aware handling in run_input_guardrails applies (block fails closed,
    # monitor fails open) rather than silently passing a `block` guardrail. An
    # explicit `valid: null` is a legitimate inconclusive verdict (not flagged).
    if "valid" not in result:
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned no 'valid' field: {result!r}"
        )
    valid = result["valid"]
    if valid is not None and not isinstance(valid, bool):
        raise GuardrailsNotReachableError(
            f"guardrail profile {cfg.profile!r} returned a non-boolean 'valid': {valid!r}"
        )

    return GuardrailResult(
        profile=cfg.profile,
        mode=cfg.mode,
        valid=valid,
        explanation=result.get("explanation"),
        score=result.get("score"),
    )


async def run_input_guardrails(
    guardrails: list[GuardrailConfig],
    input_text: str,
    *,
    default_url: str | None,
) -> GuardrailVerdict:
    """Run every input-direction guardrail and return the aggregate verdict.

    Only guardrails with ``"input"`` in :attr:`GuardrailConfig.on` run here
    (``"output"`` is accepted but not yet enforced — see the model docstring).
    Each guardrail's URL is ``cfg.url or default_url``.

    Failure handling depends on the guardrail's ``mode``:

    * ``block`` guardrails **fail closed** — if one can't be evaluated (service
      unreachable, no URL configured, malformed response) we raise
      :class:`GuardrailsNotReachableError` (the caller surfaces it as a 502)
      rather than let an unchecked request through.
    * ``monitor`` guardrails **fail open** — they're observe-only, so an
      evaluation error is logged and recorded as an inconclusive result
      (``valid=None``) and the request proceeds; we don't 502 a request the
      caller explicitly asked us not to enforce.

    Returns:
        A :class:`GuardrailVerdict` aggregating every input guardrail's result.

    Raises:
        GuardrailsNotReachableError: if a ``block``-mode guardrail can't be
            evaluated.
    """
    input_guardrails = [g for g in guardrails if "input" in g.on]
    if not input_guardrails:
        return GuardrailVerdict()

    results: list[GuardrailResult] = []
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S) as client:
        for cfg in input_guardrails:
            base_url = (cfg.url or default_url or "").rstrip("/")
            try:
                if not base_url:
                    raise GuardrailsNotReachableError(
                        f"guardrail profile {cfg.profile!r} requested but no guardrails service is "
                        "configured. Set OTARI_GUARDRAILS_URL on the gateway or pass `url` on the "
                        "guardrail entry."
                    )
                result = await _validate_one(client, base_url=base_url, cfg=cfg, input_text=input_text)
            except GuardrailsNotReachableError:
                if cfg.mode == "block":
                    raise  # fail closed: an enforcing guardrail must not be skipped
                logger.warning("monitor guardrail %r could not be evaluated; failing open", cfg.profile)
                results.append(
                    GuardrailResult(
                        profile=cfg.profile,
                        mode=cfg.mode,
                        valid=None,
                        explanation="guardrail service unavailable",
                    )
                )
                continue
            if result.flagged:
                logger.info(
                    "guardrail flagged input: profile=%s mode=%s score=%s",
                    result.profile,
                    result.mode,
                    result.score,
                )
            results.append(result)

    return GuardrailVerdict(results=results)
