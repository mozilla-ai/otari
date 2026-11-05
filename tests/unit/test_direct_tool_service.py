"""The direct tool service on its own: the order of its steps, its refusals, and what it holds, refunds and settles.

A search runs on any-search's ``fake`` provider and a fetch on any-fetch's; who pays, the ledger, the gates
and the workspace's policy are fakes that record what the service asked of them.
"""

import asyncio
import uuid
from decimal import Decimal
from typing import Any

import httpx
import pytest

from gateway.core.config import GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, PRICING_REQUIRED
from gateway.exceptions.tools_exceptions import (
    DirectToolCallError,
    FundsCheckFailedError,
    WebSearchPolicyResolutionFailedError,
    WebSearchPolicyResolutionFailure,
)
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.ports.billing_port import FUNDS_CHECK_UNAVAILABLE_DETAIL
from gateway.schemas.tools import FetchRequest, SearchRequest
from gateway.services.tools import DirectCaller, DirectToolCall, DirectToolService, DirectToolUsage
from gateway.services.upstream_redaction import PASSTHROUGH_PROVIDER_ERROR_DETAIL

WORKSPACE = uuid.uuid4()
RATE_LIMIT = {"X-RateLimit-Limit": "3"}


class _Accounts:
    def __init__(self, *, refusal: str | None = None) -> None:
        self.refusal = refusal
        self.kinds: list[str] = []

    def names_end_user(self, caller: DirectCaller) -> bool:
        return False

    def billed_user(self, caller: DirectCaller) -> str:
        return "billed"

    async def end_user(self, api_key: Any, external_id: str, *, requested_budget_id: str | None) -> tuple[str, None]:
        raise AssertionError("no end user is named")

    async def tool_refusal(self, api_key: Any, *, kind: str, provider: str, tool: str) -> str | None:
        self.kinds.append(kind)
        return self.refusal


class _Ledger:
    def __init__(self, *, rate: Decimal | None = Decimal("0.01"), funding: Exception | None = None) -> None:
        self.rate = rate
        self.funding = funding
        self.committed = True
        self.steps: list[Any] = []

    async def workspace_of(self, api_key_id: str | None) -> uuid.UUID:
        return WORKSPACE

    async def organization_of(self, api_key_id: str | None) -> None:
        return None

    async def unit_rate(self, call: DirectToolCall[Any], *, organization_id: uuid.UUID | None) -> Decimal | None:
        return self.rate

    async def hold(self, user_id: str, amount: Decimal, **_: Any) -> str:
        self.steps.append(("hold", user_id, amount))
        return "hold"

    async def refund(self, hold: object) -> None:
        self.steps.append(("refund", hold))

    async def settle(self, hold: object, cost: Decimal) -> None:
        self.steps.append(("settle", hold, cost))

    async def admit_funding(self, **_: Any) -> str:
        if self.funding is not None:
            raise self.funding
        self.steps.append(("admit_funding",))
        return "funding"

    async def settle_funding(self, funding: object, cost: Decimal, *, model: str, description: str) -> None:
        self.steps.append(("settle_funding", funding, cost, description))

    async def release(self) -> bool:
        self.steps.append(("release",))
        return self.committed


class _Gates:
    def __init__(self, *, failing_writer: Exception | None = None) -> None:
        self.failing_writer = failing_writer
        self.rows: list[DirectToolUsage] = []
        self.tracked: list[str] = []

    async def check_rate_limit(self, user_id: str) -> dict[str, str]:
        return dict(RATE_LIMIT)

    async def admit_rules(self, **_: Any) -> None:
        return None

    def track(self, *, model: str, **_: Any) -> None:
        self.tracked.append(model)

    async def record(self, usage: DirectToolUsage) -> None:
        if self.failing_writer is not None:
            raise self.failing_writer
        self.rows.append(usage)


class _Policies:
    def __init__(self, answer: ResolvedWebSearchConfig | None = None, *, unreadable: bool = False) -> None:
        self.answer = answer
        self.unreadable = unreadable

    async def resolve(self, workspace_id: uuid.UUID, requested_tools: Any) -> ResolvedWebSearchConfig | None:
        if self.unreadable:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.STORED_POLICY_INVALID)
        return self.answer


def _config(**updates: Any) -> GatewayConfig:
    options = {"hits": [{"url": "https://example.com/otari"}], "cost": 0.007}
    page = {"text": "The canned page.", "cost": 0.002}
    return GatewayConfig(
        require_pricing=updates.pop("require_pricing", False),
        web_fetch_enabled=updates.pop("web_fetch_enabled", True),
        search_tools={"fake-search": {"provider": "fake", "options": options}, "fake-free": {"provider": "fake"}},
        fetch_tools={"fake-fetch": {"provider": "fake", "options": page}},
        **updates,
    )


async def _search(
    *,
    tool: str | None = "fake-search",
    accounts: _Accounts | None = None,
    ledger: _Ledger | None = None,
    gates: _Gates | None = None,
    policies: _Policies | None = None,
    request: dict[str, Any] | None = None,
    **config: Any,
) -> Any:
    async with httpx.AsyncClient() as client:
        service = DirectToolService(
            config=_config(**config),
            accounts=accounts or _Accounts(),
            ledger=ledger or _Ledger(),
            gates=gates or _Gates(),
            policies=policies or _Policies(),
            web_client=client,
        )
        caller = DirectCaller(api_key=None, is_master_key=True, user="billed", end_user_budget_id=None, request_id="r1")
        return await service.search(
            SearchRequest(query="otari", **(request or {})), caller=caller, tool_name=tool, endpoint="/v1/search"
        )


@pytest.mark.asyncio
async def test_a_served_call_settles_its_hold_at_the_reported_cost() -> None:
    ledger, gates = _Ledger(), _Gates()

    outcome = await _search(ledger=ledger, gates=gates)

    assert outcome.response.search_tool == "fake-search"
    assert outcome.headers == RATE_LIMIT
    assert ledger.steps == [
        ("hold", "billed", Decimal("0.01")),
        ("admit_funding",),
        ("release",),
        ("settle_funding", "funding", Decimal("0.007"), "fake:fake-search"),
        ("settle", "hold", Decimal("0.007")),
    ]
    assert gates.tracked == ["fake-search"]
    (row,) = gates.rows
    assert (row.status, row.cost, row.total_tokens, row.request_group_id) == ("success", Decimal("0.007"), 0, "r1")
    assert (row.workspace_id, row.model, row.provider, row.endpoint) == (WORKSPACE, "fake-search", "fake", "/v1/search")


@pytest.mark.asyncio
async def test_a_call_whose_provider_reports_no_cost_settles_at_the_rate() -> None:
    ledger = _Ledger()

    await _search(tool="fake-free", ledger=ledger)

    assert ledger.steps[-1] == ("settle", "hold", Decimal("0.01"))


@pytest.mark.asyncio
async def test_an_unknown_tool_is_refused_before_anything_is_held_and_logged_under_its_name() -> None:
    ledger, gates = _Ledger(), _Gates()

    with pytest.raises(DirectToolCallError) as refused:
        await _search(tool="nope", ledger=ledger, gates=gates)

    assert refused.value.status_code == 400
    assert ledger.steps == []
    (row,) = gates.rows
    assert (row.model, row.provider, row.status_code, row.counts_toward_budget) == ("nope", None, 400, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("the writer is gone"), TimeoutError()], ids=["writer", "database"])
async def test_a_refusal_whose_row_cannot_be_written_answers_all_the_same(error: Exception) -> None:
    with pytest.raises(DirectToolCallError) as refused:
        await _search(tool="nope", gates=_Gates(failing_writer=error))

    assert refused.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policies", "status"),
    [
        pytest.param(_Policies(unreadable=True), 503, id="unreadable"),
        pytest.param(
            _Policies(
                ResolvedWebSearchConfig(
                    enabled=False,
                    max_results=None,
                    purpose_hint=None,
                    allowed_domains=None,
                    blocked_domains=None,
                    provider_options=None,
                    authorized_tools=None,
                )
            ),
            403,
            id="switched-off",
        ),
    ],
)
async def test_the_workspaces_policy_refuses_before_the_hold(policies: _Policies, status: int) -> None:
    ledger = _Ledger()

    with pytest.raises(DirectToolCallError) as refused:
        await _search(ledger=ledger, policies=policies)

    assert refused.value.status_code == status
    assert ledger.steps == []


@pytest.mark.asyncio
async def test_a_key_without_the_tool_is_refused_before_the_hold() -> None:
    ledger = _Ledger()

    with pytest.raises(DirectToolCallError) as refused:
        await _search(ledger=ledger, accounts=_Accounts(refusal="Search tool 'fake-search' is not permitted."))

    assert (refused.value.status_code, refused.value.message) == (403, "Search tool 'fake-search' is not permitted.")
    assert ledger.steps == []


@pytest.mark.asyncio
async def test_an_unpriced_call_under_require_pricing_is_refused_after_the_hold_and_refunded() -> None:
    ledger, gates = _Ledger(rate=None), _Gates()

    with pytest.raises(DirectToolCallError) as refused:
        await _search(ledger=ledger, gates=gates, require_pricing=True)

    assert refused.value.status_code == 402
    assert refused.value.headers is not None and refused.value.headers[ERROR_CODE_HEADER] == PRICING_REQUIRED
    assert ledger.steps == [("hold", "billed", Decimal(0)), ("refund", "hold")]
    assert [row.status_code for row in gates.rows] == [402]


@pytest.mark.asyncio
async def test_a_funds_check_that_fails_is_a_503_after_a_refund() -> None:
    ledger = _Ledger(funding=FundsCheckFailedError("RuntimeError"))

    with pytest.raises(DirectToolCallError) as refused:
        await _search(ledger=ledger)

    assert (refused.value.status_code, refused.value.message) == (503, FUNDS_CHECK_UNAVAILABLE_DETAIL)
    assert ledger.steps[-1] == ("refund", "hold")


@pytest.mark.asyncio
async def test_a_deployment_paid_call_whose_admission_does_not_commit_is_not_served() -> None:
    ledger, gates = _Ledger(), _Gates()
    ledger.committed = False

    with pytest.raises(DirectToolCallError) as refused:
        await _search(ledger=ledger, gates=gates)

    assert refused.value.status_code == 503
    assert ledger.steps[-1] == ("refund", "hold")
    assert [row.status for row in gates.rows] == ["error"]


@pytest.mark.asyncio
async def test_a_provider_failure_is_a_502_with_its_safe_fields_and_a_refund() -> None:
    ledger, gates = _Ledger(), _Gates()
    failing = {"provider_options": {"error": "http_error", "error_status": 503}}

    with pytest.raises(DirectToolCallError) as failed:
        await _search(ledger=ledger, gates=gates, request=failing)

    assert (failed.value.status_code, failed.value.message) == (502, PASSTHROUGH_PROVIDER_ERROR_DETAIL)
    assert failed.value.headers is not None and failed.value.headers["Otari-Error-Status"] == "503"
    assert ledger.steps[-1] == ("refund", "hold")
    (row,) = gates.rows
    assert (row.status, row.status_code, row.cost, row.counts_toward_budget) == ("error", 503, None, True)


@pytest.mark.asyncio
async def test_a_canceled_call_is_refunded() -> None:
    ledger = _Ledger()

    async def canceled(*_: Any, **__: Any) -> Any:
        raise asyncio.CancelledError

    with (
        pytest.MonkeyPatch.context() as patch,
        pytest.raises(asyncio.CancelledError),
    ):
        patch.setattr("gateway.services.tools._direct_tool_call.run_direct_search", canceled)
        await _search(ledger=ledger)

    assert ledger.steps[-1] == ("refund", "hold")


async def _fetch(
    url: str = "https://example.com/article",
    *,
    accounts: _Accounts | None = None,
    ledger: _Ledger | None = None,
    gates: _Gates | None = None,
    policies: _Policies | None = None,
    options: dict[str, Any] | None = None,
    **config: Any,
) -> Any:
    async with httpx.AsyncClient() as client:
        service = DirectToolService(
            config=_config(**config),
            accounts=accounts or _Accounts(),
            ledger=ledger or _Ledger(),
            gates=gates or _Gates(),
            policies=policies or _Policies(),
            web_client=client,
        )
        caller = DirectCaller(api_key=None, is_master_key=True, user="billed", end_user_budget_id=None, request_id="r2")
        request = FetchRequest(url=url, provider_options=options)
        return await service.fetch(request, caller=caller, tool_name="fake-fetch", endpoint="/v1/fetch")


@pytest.mark.asyncio
async def test_a_served_fetch_settles_at_its_reported_cost_and_asks_for_the_fetch_tool() -> None:
    accounts, ledger, gates = _Accounts(), _Ledger(), _Gates()

    outcome = await _fetch(accounts=accounts, ledger=ledger, gates=gates)

    assert (outcome.response.fetch_tool, outcome.response.text) == ("fake-fetch", "The canned page.")
    assert accounts.kinds == ["fetch"]
    assert ledger.steps[-1] == ("settle", "hold", Decimal("0.002"))
    (row,) = gates.rows
    assert (row.status, row.endpoint, row.request_group_id) == ("success", "/v1/fetch", "r2")


@pytest.mark.asyncio
async def test_a_fetch_with_web_fetch_off_is_refused_before_the_hold() -> None:
    ledger = _Ledger()

    with pytest.raises(DirectToolCallError) as refused:
        await _fetch(ledger=ledger, web_fetch_enabled=False)

    assert refused.value.status_code == 400
    assert ledger.steps == []


@pytest.mark.asyncio
async def test_a_page_found_at_a_disallowed_url_is_a_refusal_after_the_hold() -> None:
    """The provider's call refuses where it found the page, so the hold is refunded and the row is a refusal."""
    ledger, gates = _Ledger(), _Gates()
    policy = ResolvedWebSearchConfig(
        enabled=True,
        max_results=None,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=("example.org",),
        provider_options=None,
        authorized_tools=None,
    )

    with pytest.raises(DirectToolCallError) as refused:
        await _fetch(
            ledger=ledger, gates=gates, policies=_Policies(policy), options={"final_url": "https://example.org/x"}
        )

    assert refused.value.status_code == 403
    assert ledger.steps[-1] == ("refund", "hold")
    (row,) = gates.rows
    assert (row.status, row.status_code, row.cost) == ("error", 403, None)
