"""End-to-end tests for the require_pricing gate (F3) and its precedence.

These build a client with ``require_pricing=True`` (the production default; the
shared ``client`` fixture turns it off for the legacy suite). They cover the 402
route branch — exercised nowhere else — and verify that user/blocked/budget
rejections (404/403) take precedence over the missing-pricing rejection (402),
i.e. budget is reserved before the pricing gate is enforced.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.pricing import API_ORIGIN, OPERATOR_ORIGIN, OrganizationModelPricing

from .conftest import build_test_client

_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def strict_pricing_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient for a gateway with require_pricing=True (fail-closed).

    Default pricing is disabled so these tests exercise the missing-pricing gate
    in isolation: otherwise models.dev would price well-known models (gpt-4o)
    and the 402 branch would never be reached. OpenAI and Cohere hold a credential,
    because an unpriced model whose provider has none is refused as unconfigured
    instead, and the chat provider call fails in place of reaching OpenAI. One
    unpriced ``fake`` search tool stands for the direct search endpoint.
    """
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=True,
        default_pricing=False,
        web_fetch_enabled=True,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-test"}, "cohere": {"api_key": "co-test"}},
        search_tools={"fake-search": {"provider": "fake"}},
    )
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=RuntimeError("provider down"))):
        yield from build_test_client(config)


def _chat(client: TestClient, *, model: str, user: str) -> int:
    resp = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": _MESSAGES, "user": user},
        headers=_MASTER_HEADER,
    )
    return int(resp.status_code)


def test_unpriced_model_rejected_with_402(strict_pricing_client: TestClient) -> None:
    """An unpriced model is rejected with 402 when require_pricing is on (F3)."""
    strict_pricing_client.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="priced-user") == 402


def test_missing_pricing_rejection_is_recorded_in_the_usage_log(strict_pricing_client: TestClient) -> None:
    """A 402 rejection is logged as an error row so an operator can see dropped traffic.

    Regression for #317: the gate refunded the reservation and raised without
    writing anything, so requests dropped for missing pricing were invisible to
    every admin view (the activity log, the error rate, the pricing alarm's
    count). Cost stays null: nothing was spent.
    """
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    assert _chat(c, model="openai:gpt-4o", user="priced-user") == 402

    rows = c.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert len(rows) == 1
    # The resolved target, not the request selector: the same form every success
    # row on this pipeline uses, so a model's failures and successes group
    # together under one key in the activity log, the model filter (an exact
    # match on this column), and usage-by-model. #449 logged the full selector
    # here, which split them; #465 unified it.
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["provider"] == "openai"
    assert rows[0]["endpoint"] == "/v1/chat/completions"
    assert rows[0]["user_id"] == "priced-user"
    assert rows[0]["status"] == "error"
    assert rows[0]["cost"] is None
    assert "pricing" in rows[0]["error_message"].lower()
    # The gateway's own rejection code, so this shows up as "pricing" in the error
    # taxonomy instead of being indistinguishable from a provider fault (#433).
    assert rows[0]["status_code"] == 402
    # Always an enforced row, never an imported-looking one: the dashboard treats
    # counts_toward_budget=False as imported and offers those to bulk delete and
    # set-price, which must never reach a row the gateway wrote itself.
    assert rows[0]["counts_toward_budget"] is True

    # The dashboard's live "N failed in the last hour" signal reads this count.
    count = c.get(f"{API_ROOT}/usage/count", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert count["total"] == 1
    # And it reads it scoped to gateway traffic, which these rows must satisfy or
    # the alarm would undercount its own rejections.
    scoped = c.get(
        f"{API_ROOT}/usage/count", params={"status": "error", "source": "gateway"}, headers=_MASTER_HEADER
    ).json()
    assert scoped["total"] == 1


def test_passthrough_missing_pricing_rejection_is_recorded_too(strict_pricing_client: TestClient) -> None:
    """The pass-through gate (embeddings, images, rerank) records its 402 as well,
    so the failure count covers every rejected request, not only chat."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(
        f"{API_ROOT}/embeddings",
        json={"model": "openai:text-embedding-3-small", "input": "hi", "user": "priced-user"},
        headers=_MASTER_HEADER,
    )
    assert resp.status_code == 402

    rows = c.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/v1/embeddings"
    assert rows[0]["cost"] is None
    assert rows[0]["counts_toward_budget"] is True
    assert rows[0]["status_code"] == 402
    # Bare model with the instance in `provider`, the one form both scaffolds now
    # use for every row they write (see the chat assertion above).
    assert rows[0]["model"] == "text-embedding-3-small"
    assert rows[0]["provider"] == "openai"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("chat/completions", {"model": "openai:gpt-4o", "messages": _MESSAGES}),
        ("embeddings", {"model": "openai:text-embedding-3-small", "input": "hi"}),
        ("rerank", {"model": "cohere:rerank-v3.5", "query": "q", "documents": ["a", "b"]}),
        ("images/generations", {"model": "openai:dall-e-3", "prompt": "a cat"}),
        ("search/fake-search", {"query": "q"}),
    ],
)
def test_missing_pricing_refusal_carries_its_code_on_every_route(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """Chat, the pass-through routes and direct search name the refusal the same way, in the body and the header."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(f"{API_ROOT}/{path}", json={**body, "user": "priced-user"}, headers=_MASTER_HEADER)
    assert resp.status_code == 402
    assert resp.json()["code"] == "pricing_required"
    assert resp.headers["Otari-Error-Code"] == "pricing_required"


_COMPLETION_BODIES = [
    ("chat/completions", {"messages": _MESSAGES}),
    # The Messages dialect names the user in its own metadata field.
    ("messages", {"messages": _MESSAGES, "max_tokens": 16, "metadata": {"user_id": "priced-user"}}),
    ("responses", {"input": "hi"}),
]


@pytest.mark.parametrize(("path", "body"), _COMPLETION_BODIES)
def test_known_unpriced_model_is_pricing_required_on_every_completion_route(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """A model that resolves but has no price is refused as unpriced."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(
        f"{API_ROOT}/{path}", json={**body, "model": "openai:gpt-4o", "user": "priced-user"}, headers=_MASTER_HEADER
    )
    assert resp.status_code == 402, resp.text
    assert resp.headers["Otari-Error-Code"] == "pricing_required"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        *_COMPLETION_BODIES,
        ("embeddings", {"input": "hi"}),
        ("rerank", {"query": "q", "documents": ["a", "b"]}),
        ("images/generations", {"prompt": "a cat"}),
    ],
)
@pytest.mark.parametrize("model", ["no-such-model", "nosuchprovider:some-model"])
def test_unresolvable_model_is_invalid_model_not_pricing_required(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any], model: str
) -> None:
    """A model nothing resolves is invalid_model even with require_pricing on.

    It has no price either, so the pricing gate would otherwise answer first
    with a 402 that sends the caller to price a model that does not exist. The
    refusal still leaves its row.
    """
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(f"{API_ROOT}/{path}", json={**body, "model": model, "user": "priced-user"}, headers=_MASTER_HEADER)
    assert resp.status_code == 400, resp.text
    assert resp.headers["Otari-Error-Code"] == "invalid_model"
    assert "Unknown or unsupported model" in resp.text

    rows = c.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
    assert len(rows) == 1
    assert rows[0]["model"] == model
    assert rows[0]["provider"] is None
    assert rows[0]["status_code"] == 400
    assert rows[0]["cost"] is None


def test_budget_exempt_key_writes_no_pricing_rejection_row(strict_pricing_client: TestClient) -> None:
    """A budget-exempt key skips the gate, so no missing-pricing row is written.

    The gate is guarded by ``not budget_exempt``: an exempt key is never debited,
    so there is no budget for the require_pricing safety gate to protect. Nothing
    else pins that, so pin it here. (Whether a rejection row could ever carry
    counts_toward_budget=False is pinned separately, per gate, in
    test_gateway_rejection_logging.py; the shared writer fixes the flag at True.)

    Structured as an A/B against an enforced key rather than as a bare "no row
    appeared": this fixture's provider call fails, so an exempt request logs no
    pricing rejection either way. Without the control, the
    negative would pass even if the gate had stopped writing rows entirely.
    """
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "gate-user"}, headers=_MASTER_HEADER)

    def issue_key(name: str, *, exempt: bool) -> str:
        body = {"key_name": name, "user_id": "gate-user", "exclude_from_budget": exempt}
        return str(c.post(f"{API_ROOT}/keys", json=body, headers=_MASTER_HEADER).json()["key"])

    def chat(key: str) -> int:
        resp = c.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-4o", "messages": _MESSAGES},
            headers={API_KEY_HEADER: f"Bearer {key}"},
        )
        return int(resp.status_code)

    def pricing_rejections() -> list[dict[str, Any]]:
        rows = c.get(f"{API_ROOT}/usage", params={"status": "error"}, headers=_MASTER_HEADER).json()
        return [r for r in rows if "No pricing is configured" in (r["error_message"] or "")]

    # Control: an enforced key hits the gate and leaves exactly one row, carrying
    # the enforced flag. This is what makes the negative below meaningful.
    assert chat(issue_key("enforced", exempt=False)) == 402
    control = pricing_rejections()
    assert len(control) == 1
    assert control[0]["counts_toward_budget"] is True

    # The exempt key skips the gate, so it adds no rejection row of its own.
    assert chat(issue_key("exempt", exempt=True)) != 402
    assert len(pricing_rejections()) == 1


def test_priced_model_passes_the_gate(strict_pricing_client: TestClient) -> None:
    """A priced model clears the pricing gate (no 402); any later failure is a provider error."""
    strict_pricing_client.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    strict_pricing_client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=_MASTER_HEADER,
    )
    assert _chat(strict_pricing_client, model="openai:gpt-4o", user="priced-user") != 402


# An unpriced model, and an unpriced search tool on the direct endpoint.
_UNPRICED_CALLS = [
    ("chat/completions", {"model": "openai:gpt-4o", "messages": _MESSAGES}),
    ("search/fake-search", {"query": "q"}),
]


@pytest.mark.parametrize(("path", "body"), _UNPRICED_CALLS)
def test_budget_exempt_key_skips_pricing_gate(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """A key exempt from budget clears the require_pricing gate: unpriced usage is
    allowed through, not 402, since there is no budget to protect. A chat call
    then fails as a provider error, never a 402; a search is served."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "exempt-user"}, headers=_MASTER_HEADER)
    key = c.post(
        f"{API_ROOT}/keys",
        json={"key_name": "exempt", "user_id": "exempt-user", "exclude_from_budget": True},
        headers=_MASTER_HEADER,
    ).json()["key"]
    resp = c.post(f"{API_ROOT}/{path}", json=body, headers={API_KEY_HEADER: f"Bearer {key}"})
    assert resp.status_code != 402


@pytest.mark.parametrize(("path", "body"), _UNPRICED_CALLS)
def test_blocked_user_takes_precedence_over_missing_pricing(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """A blocked user gets 403, not 402 — budget/state is checked before pricing."""
    strict_pricing_client.post(
        f"{API_ROOT}/users", json={"user_id": "blocked-user", "blocked": True}, headers=_MASTER_HEADER
    )
    resp = strict_pricing_client.post(
        f"{API_ROOT}/{path}", json={**body, "user": "blocked-user"}, headers=_MASTER_HEADER
    )
    assert resp.status_code == 403


@pytest.mark.parametrize(("path", "body"), _UNPRICED_CALLS)
def test_missing_user_takes_precedence_over_missing_pricing(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """A nonexistent user gets 404, not 402 — user existence is checked before pricing."""
    resp = strict_pricing_client.post(f"{API_ROOT}/{path}", json={**body, "user": "ghost-user"}, headers=_MASTER_HEADER)
    assert resp.status_code == 404


def test_missing_pricing_rejection_row_carries_the_request_tags(strict_pricing_client: TestClient) -> None:
    """A refused request's row is tagged, so spend reports by tag also count what was refused."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    resp = c.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "openai:gpt-4o", "messages": _MESSAGES, "user": "priced-user", "metadata": {"purpose": "chat"}},
        headers=_MASTER_HEADER,
    )
    assert resp.status_code == 402

    rows = c.get(f"{API_ROOT}/usage", params={"tag": "purpose:chat"}, headers=_MASTER_HEADER).json()
    assert [row["status_code"] for row in rows] == [402]


# --------------------------------------------------------------------------- #
# The gateway's own tools in the loop, checked against the rows that settle them
# --------------------------------------------------------------------------- #

# The fixture's one search instance is the in-loop default, so an in-loop search
# runs on it and settles under its own search price first, then ``otari:web_search``.
_SEARCH_INSTANCE = "fake-search"


def _price(client: TestClient, model_key: str, *, kind: str = "model") -> None:
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={
            "kind": kind,
            "model_key": model_key,
            "input_price_per_million": 10_000.0,
            "output_price_per_million": 0.0,
        },
        headers=_MASTER_HEADER,
    )
    assert resp.status_code in {200, 201}, resp.text


def _override(client: TestClient, db: Session, model_key: str, *, origin: str) -> None:
    """The organization's own rate for ``model_key``, written with ``origin``.

    The master key writes it as the deployment's operator, so an organization's
    own write is that row with its origin changed.
    """
    created = client.post(
        f"{API_ROOT}/organizations/me/pricing",
        json={"model_key": model_key, "input_price_per_million": 10_000.0, "output_price_per_million": 0.0},
        headers=_MASTER_HEADER,
    )
    assert created.status_code == 201, created.text
    db.execute(
        update(OrganizationModelPricing).where(OrganizationModelPricing.model_key == model_key).values(origin=origin)
    )
    db.commit()


def _in_loop(client: TestClient, tool: str, provider: AsyncMock) -> Any:
    """A chat request on a priced model that declares ``tool``, its tool loop's model calls answered by ``provider``."""
    client.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    _price(client, "openai:gpt-4o")
    with patch("gateway.services.mcp_loop.acompletion", new=provider):
        return client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-4o", "messages": _MESSAGES, "user": "priced-user", "tools": [{"type": tool}]},
            headers=_MASTER_HEADER,
        )


@pytest.mark.parametrize(
    ("tool", "instance", "kind"),
    [("otari_web_search", _SEARCH_INSTANCE, "search"), ("otari_web_fetch", "builtin_fetch", "fetch")],
)
def test_an_in_loop_tool_priced_only_under_its_instances_row_is_served(
    strict_pricing_client: TestClient, tool: str, instance: str, kind: str
) -> None:
    """The instance's row prices its calls at settlement, so the check at admission lets them through."""
    _price(strict_pricing_client, instance, kind=kind)
    provider = AsyncMock(side_effect=RuntimeError("provider down"))

    resp = _in_loop(strict_pricing_client, tool, provider)

    assert resp.status_code != 402, resp.text
    provider.assert_awaited()


@pytest.mark.parametrize(("path", "body"), _COMPLETION_BODIES)
def test_an_unpriced_in_loop_search_is_refused_naming_its_row_with_its_code(
    strict_pricing_client: TestClient, path: str, body: dict[str, Any]
) -> None:
    """Refused before any model call, in each dialect, the way an unpriced model is."""
    c = strict_pricing_client
    c.post(f"{API_ROOT}/users", json={"user_id": "priced-user"}, headers=_MASTER_HEADER)
    _price(c, "openai:gpt-4o")
    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock()) as chat_loop:
        resp = c.post(
            f"{API_ROOT}/{path}",
            json={**body, "model": "openai:gpt-4o", "user": "priced-user", "tools": [{"type": "otari_web_search"}]},
            headers=_MASTER_HEADER,
        )

    assert resp.status_code == 402, resp.text
    assert resp.headers["Otari-Error-Code"] == "pricing_required"
    assert "otari:web_search" in resp.text
    chat_loop.assert_not_called()


def test_the_in_loop_refusal_carries_its_code_in_the_body_too(strict_pricing_client: TestClient) -> None:
    resp = _in_loop(strict_pricing_client, "otari_web_fetch", AsyncMock())

    assert resp.status_code == 402, resp.text
    assert resp.json()["code"] == "pricing_required"
    assert "otari:web_fetch" in resp.json()["detail"]


@pytest.mark.parametrize(("origin", "served"), [(API_ORIGIN, False), (OPERATOR_ORIGIN, True)])
def test_only_the_operators_override_prices_an_in_loop_search_on_an_instance(
    strict_pricing_client: TestClient, test_db: Session, origin: str, served: bool
) -> None:
    """The deployment pays the instance's provider, so an organization's own rate does not price the call."""
    _override(strict_pricing_client, test_db, "otari:web_search", origin=origin)

    resp = _in_loop(strict_pricing_client, "otari_web_search", AsyncMock(side_effect=RuntimeError("provider down")))

    assert (resp.status_code != 402) is served, resp.text


@pytest.mark.parametrize(("origin", "served"), [(API_ORIGIN, False), (OPERATOR_ORIGIN, True)])
def test_only_the_operators_override_prices_a_search_on_the_legacy_settings(
    strict_pricing_client: TestClient, test_db: Session, origin: str, served: bool
) -> None:
    """The deployment's own search backend runs on its account, so an organization's own rate does not price it."""
    _override(strict_pricing_client, test_db, "otari:web_search", origin=origin)
    provider = AsyncMock(side_effect=RuntimeError("provider down"))

    with patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}):
        resp = _in_loop(strict_pricing_client, "otari_web_search", provider)

    assert (resp.status_code != 402) is served, resp.text
