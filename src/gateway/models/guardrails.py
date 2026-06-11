"""Request-body model for the gateway-managed ``guardrails`` field.

Guardrails are *not* a model-callable tool. Unlike ``otari_code_execution`` /
``otari_web_search`` (which the model decides to invoke inside the tool-use
loop), a guardrail is a request-level policy the **caller** opts into: it runs
on the request regardless of what the model decides, and the model never sees
it. So it lives in its own top-level ``guardrails`` field — modelled like
``mcp_servers`` (see :mod:`gateway.models.mcp`) — rather than inside ``tools``.

The gateway extracts this field, runs the configured checks against the
operator-controlled guardrails service (``otari-anyguardrails-container``,
which exposes ``POST /validate``), and strips the field before forwarding the
request upstream. Omit the field entirely → no guardrail runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

GuardrailDirection = Literal["input", "output"]


def _default_directions() -> list[GuardrailDirection]:
    return ["input"]


class GuardrailConfig(BaseModel):
    """A single guardrail check the caller wants the gateway to enforce.

    URL safety: when ``url`` is supplied it is validated at parse time with the
    same SSRF guard used for MCP server URLs (loopback allowed by default for
    same-host sidecars; gated by ``OTARI_MCP_ALLOW_LOOPBACK`` /
    ``OTARI_MCP_ALLOW_PRIVATE_HOSTS``). Most deployments omit ``url`` and rely
    on the operator-set ``OTARI_GUARDRAILS_URL`` instead.
    """

    profile: str = Field(min_length=1, max_length=128)
    """Profile name configured on the guardrails service (e.g. ``"alinia"``)."""

    url: str | None = Field(default=None, min_length=1)
    """Optional per-request override of the operator-set ``OTARI_GUARDRAILS_URL``."""

    on: list[GuardrailDirection] = Field(default_factory=_default_directions)
    """Which directions to check. v1 enforces ``input`` only; ``output`` is
    accepted but not yet enforced (the response-direction check is a planned
    follow-up that needs streaming handling)."""

    mode: Literal["block", "monitor"] = "monitor"
    """``monitor`` (default) → forward the request anyway and annotate the
    response with the verdict (shadow mode); good for observing without
    disrupting workflows on false positives. ``block`` → reject the request
    with a 403 and never call the provider when the guardrail flags it."""

    validate_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs forwarded to the guardrails service ``/validate`` call,
    merged on top of the profile's own ``validate_kwargs`` server-side."""

    @model_validator(mode="after")
    def _check_url_safety(self) -> "GuardrailConfig":
        if self.url is not None:
            try:
                validate_mcp_url(self.url, has_authorization_token=False)
            except UnsafeURLError as exc:
                raise ValueError(str(exc)) from exc
        return self
