"""A direct tool call: a gateway-run web tool a caller asks for by path, with no model.

:class:`DirectToolService` runs one such call from its billed user to its settled
cost: the tool, the workspace's web access policy, the key's tool access, the rate
limits, the budget hold, ``require_pricing``, the organization's funds, the provider
call, the usage row, and the settlement or refund. A call that ends without a
result raises :class:`DirectToolCallError`, which carries what the caller is
answered with; a refusal the gateway makes itself writes a usage row too, so
refused traffic shows in Activity and counts as a failure.

The budget, pricing, funding, rate limit and usage code these steps use runs on the
request's session or on the request itself, which a service never holds, and the
budgets and api-keys packages reach this one through tenancy. So the service names
what it needs as interfaces (:class:`DirectToolAccounts`, :class:`DirectToolLedger`,
:class:`DirectToolGates`, :class:`WorkspaceWebPolicies`), and the builder in
``gateway.api.deps`` implements them.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from gateway.core.config import END_USER_BUDGET_HEADER
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.error_codes import INSUFFICIENT_FUNDS, PRICING_REQUIRED, error_headers
from gateway.core.metered_pricing import quantize_cost
from gateway.exceptions import TenancyError
from gateway.exceptions.tools_exceptions import (
    DirectToolCallError,
    DirectToolFailedError,
    FundsCheckFailedError,
    WebSearchPolicyResolutionFailedError,
)
from gateway.log_config import logger
from gateway.models.tools import WebTool
from gateway.ports.billing_port import FUNDS_CHECK_UNAVAILABLE_DETAIL, InsufficientFundsError
from gateway.services.tools._web_access import web_access_disabled_refusal
from gateway.services.tools._web_search import (
    pick_direct_fetch_instance,
    pick_direct_search_instance,
    resolve_direct_fetch,
    resolve_direct_search,
    run_direct_fetch,
    run_direct_search,
    unpriced_direct_fetch_detail,
    unpriced_direct_search_detail,
)
from gateway.services.upstream_redaction import PASSTHROUGH_PROVIDER_ERROR_DETAIL, redacted_upstream_detail

if TYPE_CHECKING:
    import httpx

    from gateway.core.config import GatewayConfig
    from gateway.core.settings.tools import ToolInstance, ToolKind
    from gateway.models.api_keys import APIKey
    from gateway.models.tools import ResolvedWebSearchConfig
    from gateway.schemas.tools import FetchRequest, FetchResponse, SearchRequest, SearchResponse
    from gateway.types.tool_charges import RateKey

ResultT = TypeVar("ResultT")
ResultT_co = TypeVar("ResultT_co", covariant=True)

# The usage row's model when the request named no tool that could be resolved; the column is not nullable.
_UNRESOLVED_TOOL = "unknown"


@dataclass(frozen=True)
class DirectCaller:
    """Who asks for a direct tool call, as the request authenticated it.

    ``user`` is the request's ``user`` field. ``end_user_budget_id`` is the budget
    the ``Otari-End-User-Budget`` header asks a new end user to start on.
    ``request_id`` is the id the call is answered under and its usage row grouped by.
    """

    api_key: APIKey | None
    is_master_key: bool
    user: str | None
    end_user_budget_id: str | None
    request_id: str


@dataclass(frozen=True)
class DirectToolOutcome(Generic[ResultT]):
    """A served call's response, and the headers it is answered with."""

    response: ResultT
    headers: dict[str, str]


class DirectToolResult(Protocol[ResultT_co]):
    """What a direct tool call answered, and what the provider said it cost; ``None`` leaves the cost to the rate."""

    @property
    def response(self) -> ResultT_co: ...

    @property
    def cost(self) -> Decimal | None: ...


@dataclass(frozen=True)
class DirectToolCall(Generic[ResultT]):
    """One direct tool call, resolved: what the service bills, may refuse and runs.

    ``provider`` serves the call, which an organization key can make another
    than the instance's. ``label`` and ``rate_keys`` are what billing records,
    and ``deployment_paid`` says the deployment's account runs the call, so only
    the operator's prices count and the organization's funds are charged.
    ``unpriced_detail`` is the 402's message when ``require_pricing`` finds no
    row for the call: the row to add, and where.
    """

    provider: str
    label: str | None
    rate_keys: tuple[RateKey, ...]
    deployment_paid: bool
    unpriced_detail: str
    run: Callable[[], Awaitable[DirectToolResult[ResultT]]]


@dataclass(frozen=True)
class DirectToolUsage:
    """One usage row of a direct tool call. A served call reports zero tokens; the others report none."""

    workspace_id: uuid.UUID
    api_key_id: str | None
    user_id: str
    model: str
    provider: str | None
    endpoint: str
    status: str
    latency_ms: int
    counts_toward_budget: bool
    request_group_id: str
    error_message: str | None = None
    status_code: int | None = None
    cost: Decimal | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class DirectToolAccounts(Protocol):
    """Who a direct tool call bills, and whether its key may run the tool."""

    def names_end_user(self, caller: DirectCaller) -> bool:
        """Whether the caller names an end user of its service key: anyone but the key's own user."""
        ...

    def billed_user(self, caller: DirectCaller) -> str:
        """The user the call bills: the key's own, or the one the master key names.

        Raises :class:`DirectToolCallError` when the request cannot bill one.
        """
        ...

    async def end_user(
        self, api_key: APIKey, external_id: str, *, requested_budget_id: str | None
    ) -> tuple[str, str | None]:
        """The end user a service key names, created on first use, and the budget it is on."""
        ...

    async def tool_refusal(self, api_key: APIKey | None, *, kind: ToolKind, provider: str, tool: str) -> str | None:
        """Why the key may not run the tool of this kind, or ``None`` when it may."""
        ...


class DirectToolLedger(Protocol):
    """The budget, price and funds steps of a direct tool call.

    ``hold`` returns the handle ``refund`` and ``settle`` take, and
    ``admit_funding`` the one ``settle_funding`` takes. ``release`` ends
    the request's transaction before the provider call and says whether it
    committed. ``admit_funding`` raises ``InsufficientFundsError`` when the
    organization's funds cannot cover the call, and :class:`FundsCheckFailedError`
    when they could not be read.
    """

    async def workspace_of(self, api_key_id: str | None) -> uuid.UUID:
        """The workspace a usage row of the key belongs to."""
        ...

    async def organization_of(self, api_key_id: str | None) -> uuid.UUID | None:
        """The organization whose prices and funds apply to the key's calls."""
        ...

    async def unit_rate(self, call: DirectToolCall[Any], *, organization_id: uuid.UUID | None) -> Decimal | None:
        """The rate a call is held at and falls back to, or ``None`` when no row prices it."""
        ...

    async def hold(
        self, user_id: str, amount: Decimal, *, api_key: APIKey | None, provider: str, counts_toward_budget: bool
    ) -> object:
        """Hold ``amount`` on the user's budget and the ceilings that bind the call."""
        ...

    async def refund(self, hold: object) -> None:
        """Release a hold, charging nothing."""
        ...

    async def settle(self, hold: object, cost: Decimal) -> None:
        """Settle a hold at the call's cost."""
        ...

    async def admit_funding(
        self, *, organization_id: uuid.UUID | None, api_key_id: str | None, user_id: str, request_id: str
    ) -> object | None:
        """Admit a call the deployment's account runs on the organization's funds, returning what settles it."""
        ...

    async def settle_funding(self, funding: object, cost: Decimal, *, model: str, description: str) -> None:
        """Charge the organization's funds what a settled ``cost`` owes."""
        ...

    async def release(self) -> bool:
        """Commit the request's transaction and free its connection; ``False`` when the commit failed."""
        ...


class DirectToolGates(Protocol):
    """The request-scoped steps of a direct tool call: its rate limits, its in-flight entry and its usage rows.

    ``check_rate_limit`` and ``admit_rules`` raise the 429 a caller over a limit gets.
    """

    async def check_rate_limit(self, user_id: str) -> dict[str, str]:
        """Count the call against the user's rpm, returning the headers a served call is answered with."""
        ...

    async def admit_rules(
        self, *, api_key_id: str | None, user_id: str, budget_exempt: bool, provider: str, tool: str
    ) -> None:
        """Count the call against the ``rate_limits`` rules and, unless exempt, the user's per-minute budget limits."""
        ...

    def track(self, *, endpoint: str, model: str, provider: str, user_id: str, api_key_id: str | None) -> None:
        """Mark the request in flight until its response has been sent."""
        ...

    async def record(self, usage: DirectToolUsage) -> None:
        """Write a usage row."""
        ...


class WorkspaceWebPolicies(Protocol):
    """A workspace's web access policy, as the in-loop tools read it."""

    async def resolve(
        self, workspace_id: uuid.UUID, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        """The policy that governs ``requested_tools``, or ``None`` where the workspace has none."""
        ...


def _refusal_detail(exc: TenancyError) -> str:
    """What the caller is told of a domain error: a 5xx names no credential or deployment state."""
    return PASSTHROUGH_PROVIDER_ERROR_DETAIL if exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR else exc.message


def _elapsed_ms(started_at: float) -> int:
    return round((time.monotonic() - started_at) * 1000)


class DirectToolService:
    """Runs direct tool calls: :meth:`search` on a configured search tool, :meth:`fetch` on a fetch tool."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        accounts: DirectToolAccounts,
        ledger: DirectToolLedger,
        gates: DirectToolGates,
        policies: WorkspaceWebPolicies,
        web_client: httpx.AsyncClient,
    ) -> None:
        self._config = config
        self._accounts = accounts
        self._ledger = ledger
        self._gates = gates
        self._policies = policies
        self._web_client = web_client

    async def search(
        self, request: SearchRequest, *, caller: DirectCaller, tool_name: str | None, endpoint: str
    ) -> DirectToolOutcome[SearchResponse]:
        """Run a search on the tool named, else the deployment's default; ``endpoint`` labels its usage row."""
        config = self._config
        client = self._web_client

        def resolve(tool: ToolInstance, policy: ResolvedWebSearchConfig | None) -> DirectToolCall[SearchResponse]:
            # A workspace whose organization brought its own search key searches with it on a
            # call that names no tool, so its own account pays. The tool is still what the
            # request is allowlisted, rate-limited and timed out as.
            search = resolve_direct_search(config, tool, policy, request, named=tool_name is not None)
            return DirectToolCall(
                provider=search.resolved.provider,
                label=search.resolved.label,
                rate_keys=search.resolved.rate_keys,
                deployment_paid=not search.resolved.on_credential,
                unpriced_detail=unpriced_direct_search_detail(search),
                run=lambda: run_direct_search(search, client),
            )

        return await self._run(
            endpoint=endpoint,
            kind="search",
            caller=caller,
            tool_name=tool_name,
            pick=lambda: pick_direct_search_instance(config, tool_name),
            policy_tools=(WebTool.SEARCH,),
            resolve=resolve,
        )

    async def fetch(
        self, request: FetchRequest, *, caller: DirectCaller, tool_name: str | None, endpoint: str
    ) -> DirectToolOutcome[FetchResponse]:
        """Read one page through the fetch tool named, else the deployment's; ``endpoint`` labels its usage row."""
        config = self._config
        client = self._web_client

        def resolve(tool: ToolInstance, policy: ResolvedWebSearchConfig | None) -> DirectToolCall[FetchResponse]:
            # A workspace whose organization brought its own fetch key reads pages with it
            # on a call that names no tool, so its own account pays. The tool is still what
            # the request is allowlisted, rate-limited and timed out as.
            fetch = resolve_direct_fetch(config, tool, policy, request, named=tool_name is not None)
            return DirectToolCall(
                provider=fetch.resolved.provider,
                label=fetch.resolved.label,
                rate_keys=fetch.resolved.rate_keys,
                deployment_paid=not fetch.resolved.on_credential,
                unpriced_detail=unpriced_direct_fetch_detail(fetch),
                run=lambda: run_direct_fetch(fetch, client, trust_env_proxy=config.web_retrieval_trust_env_proxy),
            )

        return await self._run(
            endpoint=endpoint,
            kind="fetch",
            caller=caller,
            tool_name=tool_name,
            pick=lambda: pick_direct_fetch_instance(config, tool_name),
            policy_tools=(WebTool.FETCH,),
            resolve=resolve,
        )

    async def _run(
        self,
        *,
        endpoint: str,
        kind: ToolKind,
        caller: DirectCaller,
        tool_name: str | None,
        pick: Callable[[], ToolInstance],
        policy_tools: Sequence[WebTool],
        resolve: Callable[[ToolInstance, ResolvedWebSearchConfig | None], DirectToolCall[ResultT]],
    ) -> DirectToolOutcome[ResultT]:
        """Pick, gate, hold, call, log and settle one direct tool call.

        ``pick`` returns the instance the call runs on, or raises a domain error
        whose status answers the caller; ``tool_name`` is what the request named,
        for the row of a pick that resolved nothing. ``policy_tools`` are the web
        tools the workspace's policy is read for. ``resolve`` turns the instance
        and that policy into the call, or raises a domain error. ``kind`` is the
        tool's kind, which the key's tool list is checked for and the log lines name.
        """
        started_at = time.monotonic()
        request_id = caller.request_id
        api_key = caller.api_key
        api_key_id = api_key.id if api_key else None
        # A key flagged exclude_from_budget logs cost but is never reserved or folded into users.spend.
        budget_exempt = api_key is not None and api_key.exclude_from_budget
        headers: dict[str, str] = {}

        if api_key is not None and caller.user and self._accounts.names_end_user(caller):
            # The owner's own rpm is checked before an end user is created for it.
            rate_limit_headers = await self._gates.check_rate_limit(str(api_key.user_id))
            user_id, end_user_budget_id = await self._accounts.end_user(
                api_key, caller.user, requested_budget_id=caller.end_user_budget_id
            )
            if end_user_budget_id is not None:
                headers[END_USER_BUDGET_HEADER] = end_user_budget_id
        else:
            user_id = self._accounts.billed_user(caller)
            rate_limit_headers = await self._gates.check_rate_limit(user_id)

        async def refused(
            detail: str,
            status_code: int,
            *,
            model: str,
            provider: str | None,
            workspace_id: uuid.UUID | None = None,
            answer: str | None = None,
            answer_headers: dict[str, str] | None = None,
        ) -> DirectToolCallError:
            """Record a call the gateway itself refused, and return the error that answers it.

            The row keeps the status the refusal returns and no cost. Writing it is
            best-effort: a sick log writer must not turn a clean 4xx into a 500.
            """
            try:
                await self._gates.record(
                    DirectToolUsage(
                        workspace_id=workspace_id or await self._ledger.workspace_of(api_key_id),
                        api_key_id=api_key_id,
                        user_id=user_id,
                        model=model,
                        provider=provider,
                        endpoint=endpoint,
                        status="error",
                        latency_ms=_elapsed_ms(started_at),
                        # A refusal has no cost for the flag to gate, and a row with it off reads as imported usage.
                        counts_toward_budget=True,
                        request_group_id=request_id,
                        error_message=redacted_upstream_detail(detail, PASSTHROUGH_PROVIDER_ERROR_DETAIL),
                        status_code=status_code,
                    )
                )
            except DATABASE_ERRORS as exc:
                # The error's own text can carry the row's values, so the log names the failure only.
                logger.error(
                    "Failed to record gateway rejection for %s on %s: %s",
                    user_id,
                    endpoint,
                    describe_database_error(exc),
                )
            except Exception:
                logger.exception("Failed to record gateway rejection for %s on %s", user_id, endpoint)
            return DirectToolCallError(answer or detail, status_code=status_code, headers=answer_headers)

        # Picked before any hold is taken, so an unknown tool costs the caller
        # nothing to be told about. The refusal is still logged: a request the
        # gateway turned away is dropped traffic, and it belongs in Activity and in
        # the failure count, where the operator can see it.
        try:
            tool = pick()
        except TenancyError as exc:
            # No tool was resolved, so there is no provider to attribute the row to
            # and the name is whatever the caller asked for.
            raise await refused(
                exc.message,
                exc.status_code,
                model=tool_name or _UNRESOLVED_TOOL,
                provider=None,
                answer=_refusal_detail(exc),
            ) from exc

        # Resolved before the gates below, because the first of them is a question
        # about the workspace.
        workspace_id = await self._ledger.workspace_of(api_key_id)

        # The workspace's web access policy, read as the in-loop tool reads it. A
        # workspace that has turned web access off has turned it off: leaving this
        # endpoint open would leave the switch bypassable by any key in that
        # workspace, which is a policy that fails open.
        try:
            policy = await self._policies.resolve(workspace_id, policy_tools)
        except WebSearchPolicyResolutionFailedError as exc:
            raise await refused(
                exc.message,
                HTTPStatus.SERVICE_UNAVAILABLE,
                model=tool.name,
                provider=tool.provider,
                workspace_id=workspace_id,
            ) from exc
        if policy is not None and not policy.enabled:
            refusal = web_access_disabled_refusal(policy_tools)
            raise await refused(
                refusal.message,
                HTTPStatus.FORBIDDEN,
                model=tool.name,
                provider=tool.provider,
                workspace_id=workspace_id,
            )

        try:
            call = resolve(tool, policy)
        except TenancyError as exc:
            raise await refused(
                exc.message,
                exc.status_code,
                model=tool.name,
                provider=tool.provider,
                workspace_id=workspace_id,
                answer=_refusal_detail(exc),
            ) from exc

        # Tool access control (per-key), by the tool's name. A key with no list of its
        # own inherits its user's, and a key restricted to models must be granted the
        # tool, which is the fail-closed side to be on for a spend surface.
        not_allowed = await self._accounts.tool_refusal(api_key, kind=kind, provider=tool.provider, tool=tool.name)
        if not_allowed is not None:
            raise await refused(
                not_allowed, HTTPStatus.FORBIDDEN, model=tool.name, provider=tool.provider, workspace_id=workspace_id
            )

        # The rate the hold takes and the settlement falls back to. Whose account
        # runs the call decides whose overrides count, and whether the
        # organization's funds pay for it.
        organization_id = await self._ledger.organization_of(api_key_id)
        rate = await self._ledger.unit_rate(call, organization_id=organization_id)
        unit_rate = rate if rate is not None else Decimal(0)
        # After the gates above, so a call they refuse is counted by no rule, and
        # before the hold, so one the rules refuse holds no budget to refund.
        await self._gates.admit_rules(
            api_key_id=api_key_id, user_id=user_id, budget_exempt=budget_exempt, provider=tool.provider, tool=tool.name
        )
        hold = await self._ledger.hold(
            user_id, unit_rate, api_key=api_key, provider=tool.provider, counts_toward_budget=not budget_exempt
        )

        async def refused_after_hold(
            detail: str, status_code: int, answer_headers: dict[str, str] | None = None
        ) -> DirectToolCallError:
            """Refund the hold, then record and return the refusal."""
            await self._ledger.refund(hold)
            return await refused(
                detail,
                status_code,
                model=tool.name,
                provider=tool.provider,
                workspace_id=workspace_id,
                answer_headers=answer_headers,
            )

        # After the budget checks, so a missing or exhausted budget answers first. The
        # call is refused when require_pricing is on and no row prices it, even if the
        # provider reports its own cost; a zero rate is a price. Two calls pass
        # unpriced: one on a key exempt from budgets, which is never debited, and one
        # with no rate keys, which this gateway does not settle.
        if rate is None and self._config.require_pricing and not budget_exempt and call.rate_keys:
            raise await refused_after_hold(
                call.unpriced_detail, HTTPStatus.PAYMENT_REQUIRED, error_headers(PRICING_REQUIRED)
            )
        # Last, so a refusal above answers before the organization's funds are read.
        funding: object | None = None
        if call.deployment_paid:
            try:
                funding = await self._ledger.admit_funding(
                    organization_id=organization_id, api_key_id=api_key_id, user_id=user_id, request_id=request_id
                )
            except InsufficientFundsError as exc:
                raise await refused_after_hold(
                    exc.public_message, HTTPStatus.PAYMENT_REQUIRED, error_headers(INSUFFICIENT_FUNDS)
                ) from exc
            except FundsCheckFailedError as exc:
                logger.error(
                    "BillingPort failed to admit a deployment-paid %s: organization=%s request_id=%s: %s",
                    kind,
                    organization_id,
                    request_id,
                    exc.description,
                )
                raise await refused_after_hold(FUNDS_CHECK_UNAVAILABLE_DETAIL, HTTPStatus.SERVICE_UNAVAILABLE) from exc

        def usage(**fields: Any) -> DirectToolUsage:
            """The row of a call that reached the provider; it carries a cost, so the key's exemption applies."""
            return DirectToolUsage(
                workspace_id=workspace_id,
                api_key_id=api_key_id,
                user_id=user_id,
                model=tool.name,
                provider=tool.provider,
                endpoint=endpoint,
                latency_ms=_elapsed_ms(started_at),
                counts_toward_budget=not budget_exempt,
                request_group_id=request_id,
                **fields,
            )

        # Every gate has passed and the provider is about to be called, so the request
        # is in flight from here until its response has been sent. The usage row
        # carries the same tool name and provider.
        self._gates.track(
            endpoint=endpoint, model=tool.name, provider=tool.provider, user_id=user_id, api_key_id=api_key_id
        )
        # Frees the request's one connection before the provider call; settlement
        # checks one back out. The commit also lands any credit the funding gate
        # granted, so a call whose admission did not commit is not served on it.
        if not await self._ledger.release() and funding is not None:
            logger.error(
                "Could not commit a deployment-paid %s admission: organization=%s request_id=%s",
                kind,
                organization_id,
                request_id,
            )
            raise await refused_after_hold(FUNDS_CHECK_UNAVAILABLE_DETAIL, HTTPStatus.SERVICE_UNAVAILABLE)

        try:
            result = await call.run()
        except Exception as exc:
            if isinstance(exc, TenancyError) and not isinstance(exc, DirectToolFailedError) and exc.status_code < 500:
                # The call's own checks refused it after the hold, as a fetch refuses the
                # URL a provider found the page at: a refusal, not a provider failure.
                raise await refused_after_hold(exc.message, exc.status_code) from exc
            if isinstance(exc, DirectToolFailedError):
                # The failure's safe fields ride as headers (decision 67); the detail stays generic.
                failed, row_status, failure_headers = (
                    exc.message,
                    exc.upstream_status or HTTPStatus.BAD_GATEWAY,
                    exc.headers,
                )
            else:
                # Raised by something other than the provider, and named by its type alone,
                # since an adapter's exception text could carry the query or the URL.
                failed, row_status, failure_headers = type(exc).__name__, HTTPStatus.BAD_GATEWAY, None
            await self._gates.record(usage(status="error", error_message=failed, status_code=row_status))
            await self._ledger.refund(hold)
            logger.error("%s failed for tool '%s' (%s): %s", kind.capitalize(), tool.name, call.provider, failed)
            raise DirectToolCallError(
                PASSTHROUGH_PROVIDER_ERROR_DETAIL, status_code=HTTPStatus.BAD_GATEWAY, headers=failure_headers
            ) from exc
        except BaseException:
            # A request its client dropped is refunded all the same.
            await self._ledger.refund(hold)
            raise

        # The provider's own charge is the true cost; the rate is the fallback for a
        # provider that reports none. Rounded once, here, so the row and the
        # settled spend agree.
        cost = quantize_cost(result.cost if result.cost is not None else unit_rate)
        # The call bills per request; there are no tokens to report, and a zero is
        # truthful where a null would read as "unknown".
        await self._gates.record(
            usage(status="success", prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=cost)
        )
        if funding is not None:
            await self._ledger.settle_funding(
                funding, cost, model=tool.name, description=f"{tool.provider}:{tool.name}"
            )
        await self._ledger.settle(hold, cost)
        return DirectToolOutcome(result.response, {**headers, **rate_limit_headers})
