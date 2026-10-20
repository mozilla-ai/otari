"""Structured decisions: typed questions over a state, answered with probabilities.

TypeSafe's ``/v1/systemone`` defined the shape, and OpenRouter's alpha
``/api/alpha/decisions`` and llama-server's ``/v1/systemone`` adopted it, so one
request body reaches any of them. None is an any-llm provider, so this module is
the client for all three: it resolves a
``<name>:<model>`` selector against ``decision_providers``, sends the body to
that upstream, and returns its answer unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx

from gateway.core.metered_pricing import quantize_cost
from gateway.services.provider_kwargs import split_selector

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.schemas.inference import DecisionRequest, DecisionResponse

# Each provider's default root and the path under it. ``api_base`` replaces the
# root, so it points a provider at a proxy or regional host without the path.
# A self-hosted server has no default root; config validation requires one.
_ENDPOINTS: dict[str, tuple[str | None, str]] = {
    "typesafe": ("https://api.typesafe.ai", "/v1/systemone"),
    "openrouter": ("https://openrouter.ai/api", "/alpha/decisions"),
    "llamacpp": (None, "/v1/systemone"),
}

_DEFAULT_TIMEOUT_S = 30.0

# See search_backend._client: the same lazily built, process-wide pool.
_client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class DecisionProvider:
    """A configured ``decision_providers`` entry, ready to dispatch against."""

    name: str
    """The selector prefix, and the instance pricing, budgets and usage key on."""
    provider: str
    """One of ``DECISION_PROVIDERS``."""
    api_key: str | None
    """``None`` for a keyless self-hosted server."""
    url: str
    timeout_s: float


class UnknownDecisionProviderError(ValueError):
    """The selector names no configured decision provider."""


class DecisionProviderError(RuntimeError):
    """The upstream refused the request, could not be reached, or answered unreadably."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        """The upstream's HTTP status, or ``None`` when no response arrived."""


def get_decision_client() -> httpx.AsyncClient:
    """The process-wide pooled client decisions are dispatched on."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient()
    return _client


async def close_decision_client() -> None:
    """Close the pooled client. A no-op when no decision was ever dispatched."""
    global _client
    client, _client = _client, None
    if client is not None and not client.is_closed:
        await client.aclose()


def resolve_decision_provider(config: GatewayConfig, selector: str) -> tuple[DecisionProvider, str]:
    """Split ``selector`` into its configured provider and the bare model name.

    Raises:
        UnknownDecisionProviderError: If the prefix is not a ``decision_providers`` key.

    """
    split = split_selector(selector)
    if split is None or split[0] not in config.decision_providers:
        configured = ", ".join(sorted(config.decision_providers)) or "none"
        msg = (
            f"Model '{selector}' does not name a configured decision provider. "
            f"Use '<provider>:<model>' with one of: {configured}."
        )
        raise UnknownDecisionProviderError(msg)
    name, model = split
    entry = config.decision_providers[name]
    provider = str(entry.get("provider") or name)
    default_root, path = _ENDPOINTS[provider]
    root = str(entry.get("api_base") or default_root).rstrip("/")
    return (
        DecisionProvider(
            name=name,
            provider=provider,
            api_key=str(entry["api_key"]) if entry.get("api_key") else None,
            url=f"{root}{path}",
            timeout_s=float(entry.get("timeout") or _DEFAULT_TIMEOUT_S),
        ),
        model,
    )


def decision_body(request: DecisionRequest, model: str) -> dict[str, Any]:
    """The body ``request`` is sent upstream as, naming the bare ``model`` rather than its selector."""
    questions = {name: question.model_dump(exclude_none=True) for name, question in request.questions.items()}
    body: dict[str, Any] = {"model": model, "state": request.state, "questions": questions}
    if request.images:
        body["images"] = request.images
    return body


def reported_charge(response: DecisionResponse) -> Decimal | None:
    """The charge the provider stated for the call, or ``None`` where it stated none."""
    usage = response.usage
    return quantize_cost(Decimal(str(usage.cost))) if usage is not None and usage.cost is not None else None


async def request_decision(
    provider: DecisionProvider,
    body: dict[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send ``body`` to the provider and return its decoded answer.

    The error message names the provider and the upstream status only. A
    provider's error body can echo the state it was sent, which is caller
    content the security guidance keeps out of logs.

    Raises:
        DecisionProviderError: On a transport failure, an error status, or a
            body that is not a JSON object.

    """
    http = client or get_decision_client()
    try:
        response = await http.post(
            provider.url,
            json=body,
            headers={"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {},
            timeout=provider.timeout_s,
            # A redirect would carry the provider's key to wherever it points.
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        msg = f"{provider.provider} decisions could not be reached: {type(exc).__name__}"
        raise DecisionProviderError(msg) from exc

    if response.status_code >= httpx.codes.MULTIPLE_CHOICES:
        msg = f"{provider.provider} decisions returned HTTP {response.status_code}"
        raise DecisionProviderError(msg, status_code=response.status_code)

    try:
        answer = response.json()
    except ValueError as exc:
        msg = f"{provider.provider} decisions returned a body that is not JSON"
        raise DecisionProviderError(msg, status_code=response.status_code) from exc
    if not isinstance(answer, dict):
        msg = f"{provider.provider} decisions returned a {type(answer).__name__} body, expected an object"
        raise DecisionProviderError(msg, status_code=response.status_code)
    return answer
