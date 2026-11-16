"""Structured decisions: typed questions over a state, answered with probabilities.

TypeSafe's ``/v1/systemone`` defined the shape, and OpenRouter's alpha
``/api/alpha/decisions`` and llama-server's ``/v1/systemone`` adopted it, so one
request body reaches any of them. Gemini's ``/v1beta/decisions`` asks the same
three question types in its own words, so its body and answer are translated
here. None is an any-llm provider, so this module is the client for all of them:
it resolves a ``<name>:<model>`` selector against ``decision_providers``, sends
the body to that upstream, and returns its answer in TypeSafe's shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

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
    "gemini": ("https://generativelanguage.googleapis.com", "/v1beta/decisions"),
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
    return (
        decision_provider_for(
            name,
            str(entry.get("provider") or name),
            api_key=str(entry["api_key"]) if entry.get("api_key") else None,
            api_base=str(entry["api_base"]) if entry.get("api_base") else None,
            timeout_s=float(entry.get("timeout") or _DEFAULT_TIMEOUT_S),
        ),
        model,
    )


def decision_provider_for(
    name: str, provider: str, *, api_key: str | None, api_base: str | None, timeout_s: float
) -> DecisionProvider:
    """A provider to dispatch against, from the facts a ``decision_providers`` entry or a stored row holds.

    ``provider`` is one of ``DECISION_PROVIDERS``, and ``api_base`` replaces its
    default root, which a self-hosted server has none of.
    """
    default_root, path = _ENDPOINTS[provider]
    root = str(api_base or default_root).rstrip("/")
    return DecisionProvider(name=name, provider=provider, api_key=api_key, url=f"{root}{path}", timeout_s=timeout_s)


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


def _gemini_text(content: object) -> str:
    """Gemini takes text where TypeSafe also takes structured JSON, which its guide says to serialize."""
    return content if isinstance(content, str) else json.dumps(content)


def _gemini_image(data_url: str) -> dict[str, str]:
    """An inline image block from a ``data:image/...;base64,...`` URL, the only image form a request admits."""
    header, _, data = data_url.partition(",")
    return {"type": "image", "mime_type": header.removeprefix("data:").split(";", 1)[0], "data": data}


def _gemini_question(question: dict[str, Any]) -> dict[str, Any]:
    """One question in Gemini's words. Fields TypeSafe allows beyond these have no Gemini counterpart."""
    kind = question["type"]
    out: dict[str, Any] = {"type": kind, "instructions": _gemini_text(question["instructions"])}
    criteria = question.get("criteria")
    if kind == "noul" and isinstance(criteria, dict):
        named = {"true_description": criteria.get("true"), "false_description": criteria.get("false")}
        described = {key: _gemini_text(value) for key, value in named.items() if value is not None}
        if described:
            out["criteria"] = described
    elif kind == "choice" and isinstance(criteria, dict):
        out["criteria"] = {option: "" if text is None else _gemini_text(text) for option, text in criteria.items()}
    elif kind == "score" and isinstance(criteria, list):
        out["criteria"] = [_gemini_text(level) for level in criteria]
    return out


def _gemini_body(body: dict[str, Any]) -> dict[str, Any]:
    """``body``, in TypeSafe's shape, as Gemini's Decisions API takes it.

    The state becomes ``input``: text, followed by an image block per inline image.
    """
    text = _gemini_text(body["state"])
    images = cast("list[str]", body.get("images") or [])
    payload: str | list[dict[str, str]] = (
        [{"type": "text", "text": text}, *(_gemini_image(image) for image in images)] if images else text
    )
    questions = cast("dict[str, dict[str, Any]]", body["questions"])
    return {
        "model": body["model"],
        "input": payload,
        "questions": {name: _gemini_question(question) for name, question in questions.items()},
    }


def _gemini_answer(answer: dict[str, Any], model: str, *, status_code: int) -> dict[str, Any]:
    """Gemini's answer in TypeSafe's shape: the model named, and the token counts under TypeSafe's keys.

    Gemini's own usage fields stay alongside, as extra fields the response schema keeps.
    Gemini generates no output tokens, so ``total_tokens`` stands in for a missing
    input count; an answer with neither is refused rather than settled as free.

    Raises:
        DecisionProviderError: If the answer reports no input token count.

    """
    usage = answer.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("total_input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("total_tokens")
    if input_tokens is None:
        msg = "gemini decisions returned an answer with no input token count"
        raise DecisionProviderError(msg, status_code=status_code)
    output_tokens = usage.get("total_output_tokens")
    return {
        "model": model,
        **answer,
        "usage": {
            **usage,
            "input_tokens": input_tokens,
            "output_tokens": 0 if output_tokens is None else output_tokens,
        },
    }


def _auth_headers(provider: DecisionProvider) -> dict[str, str]:
    if not provider.api_key:
        return {}
    if provider.provider == "gemini":
        return {"x-goog-api-key": provider.api_key}
    return {"Authorization": f"Bearer {provider.api_key}"}


async def request_decision(
    provider: DecisionProvider,
    body: dict[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send ``body`` to the provider and return its decoded answer.

    ``body`` and the answer are in TypeSafe's shape whichever provider is asked;
    a provider that speaks another is translated on the way out and back.

    The error message names the provider and the upstream status only. A
    provider's error body can echo the state it was sent, which is caller
    content the security guidance keeps out of logs.

    Raises:
        DecisionProviderError: On a transport failure, an error status, a
            body that is not a JSON object, or a Gemini answer with no input
            token count.

    """
    http = client or get_decision_client()
    gemini = provider.provider == "gemini"
    try:
        response = await http.post(
            provider.url,
            json=_gemini_body(body) if gemini else body,
            headers=_auth_headers(provider),
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
    if gemini:
        return _gemini_answer(answer, str(body.get("model", "")), status_code=response.status_code)
    return answer
