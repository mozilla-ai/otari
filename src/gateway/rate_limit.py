"""Sliding-window rate limiting: in-process limiters, and the per-user limit and rules counted in a shared store."""

import asyncio
import itertools
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

from gateway.core.error_codes import RATE_LIMITED, error_headers
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.ports.rate_limit_store_port import RateLimitStorePort, RateLimitWindow

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

    from gateway.core.config import GatewayConfig, RateLimitRule

RATE_LIMIT_HITS = Counter(
    "gateway_rate_limit_hits",
    "Total number of rate limit hits",
    registry=REGISTRY,
)
# Bounded by the models the per: model rules name, which is config, not traffic.
RATE_LIMIT_MODEL_FULL = Counter(
    "gateway_rate_limit_model_full",
    "Attempts a per: model rate limit refused, so routing moved on or the request was refused",
    ["rule", "model"],
    registry=REGISTRY,
)


@dataclass
class RateLimitInfo:
    """Rate limit status returned by a successful check."""

    limit: int
    remaining: int
    reset: float


@dataclass
class _Window:
    """One key's admitted entries, oldest first, and what each one counts for."""

    entries: deque[tuple[float, str]] = field(default_factory=deque)
    costs: dict[str, int] = field(default_factory=dict)
    total: int = 0


class SlidingWindowLog:
    """Admitted entries per key, held in this process.

    Exact: an entry costing ``cost`` is admitted when the entries admitted in
    the last ``window_sec`` cost at most ``limit`` with it. Keys idle for longer
    than the widest window seen are dropped every ``_CLEANUP_INTERVAL`` hits, so
    the map is bounded by the keys that are active.
    """

    _CLEANUP_INTERVAL = 1000

    def __init__(self) -> None:
        self._requests: dict[str, _Window] = defaultdict(_Window)
        self._call_count = 0
        self._widest_window = 0.0
        self._handles = itertools.count()

    def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> RateLimitWindow:
        """Count an entry costing ``cost`` against ``key`` if it fits under ``limit``."""
        now = time.monotonic()
        cutoff = now - window_sec
        self._widest_window = max(self._widest_window, window_sec)

        self._call_count += 1
        if self._call_count >= self._CLEANUP_INTERVAL:
            self._cleanup(now - self._widest_window)
            self._call_count = 0

        window = self._requests[key]
        while window.entries and window.entries[0][0] <= cutoff:
            _, expired = window.entries.popleft()
            window.total -= window.costs.pop(expired)

        handle = None
        if window.total + cost <= limit:
            handle = str(next(self._handles))
            window.entries.append((now, handle))
            window.costs[handle] = cost
            window.total += cost
        reset_after = window.entries[0][0] + window_sec - now if window.entries else window_sec
        return RateLimitWindow(allowed=handle is not None, count=window.total, reset_after=reset_after, handle=handle)

    def settle(self, key: str, handle: str, cost: int) -> None:
        """Make an admitted entry count for ``cost`` instead; a no-op once it has left the window."""
        window = self._requests.get(key)
        if window is None or handle not in window.costs:
            return
        window.total += cost - window.costs[handle]
        window.costs[handle] = cost

    def _cleanup(self, cutoff: float) -> None:
        """Remove entries for keys with no recent requests."""
        stale = [key for key, w in self._requests.items() if not w.entries or w.entries[-1][0] <= cutoff]
        for key in stale:
            del self._requests[key]


def _info_or_raise(window: RateLimitWindow, limit: int) -> RateLimitInfo:
    """The headers' view of an admitted request, or the 429 for a refused one."""
    if not window.allowed:
        RATE_LIMIT_HITS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(math.ceil(window.reset_after)), **error_headers(RATE_LIMITED)},
        )
    # Wall-clock time for the externally facing reset header.
    return RateLimitInfo(limit=limit, remaining=limit - window.count, reset=time.time() + window.reset_after)


class RateLimiter:
    """A sliding-window limit counted in this process.

    For the limits that guard one process's own surfaces (sign-in, feedback,
    the public catalog). ``window_sec`` widens the window for a limit counted
    over longer than a minute; ``rpm`` is then the allowance per window.
    """

    def __init__(self, rpm: int, *, window_sec: float = 60.0) -> None:
        self._rpm = rpm
        self._window_sec = window_sec
        self._log = SlidingWindowLog()

    def check(self, user_id: str) -> RateLimitInfo:
        """Check whether a request is allowed for the given key.

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        return _info_or_raise(self._log.hit(user_id, self._rpm, self._window_sec), self._rpm)


class UserRateLimiter:
    """The per-user limit (``rate_limit_rpm``), counted in the store this build bound.

    With a shared store the limit holds for the deployment, however many
    replicas serve it.
    """

    def __init__(self, store: RateLimitStorePort, rpm: int, *, window_sec: float = 60.0) -> None:
        self._store = store
        self._rpm = rpm
        self._window_sec = window_sec

    async def check(self, user_id: str) -> RateLimitInfo:
        """Count one request for ``user_id``.

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        window = await self._store.hit(f"user:{user_id}", self._rpm, self._window_sec)
        return _info_or_raise(window, self._rpm)


async def check_rate_limit(request: Request, user_id: str) -> RateLimitInfo | None:
    """Check rate limit for a user, returning info for header injection.

    Returns RateLimitInfo when rate limiting is active, None when disabled.
    """
    rate_limiter: UserRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return None
    return await rate_limiter.check(user_id)


_RULE_WINDOW_SEC = 60.0
# Where a request's grant waits for RateLimitGrantMiddleware, in the ASGI scope's state.
_GRANT_STATE = "rate_limit_grant"
# A full max_concurrent frees as soon as any request ends, which no window predicts.
_CONCURRENCY_RETRY_AFTER_SEC = 1.0


class _Hold:
    """What one admission counted: its window entries, the token estimates among them, and its slots."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.estimates: list[tuple[str, str]] = []
        self.leases: list[tuple[str, str]] = []


class ModelHold:
    """What one attempt holds under the ``per: model`` rules naming its model."""

    def __init__(self, grant: "RateLimitGrant", hold: _Hold) -> None:
        self._grant = grant
        self._hold = hold

    async def drop(self, sent: bool = True) -> None:
        """Charge the attempt no tokens and give back its slots, for an attempt the walk moves past.

        Its requests stay counted when ``sent``, because the provider was sent them;
        otherwise nothing it took stays counted.
        """
        if self._hold in self._grant._holds:
            self._grant._holds.remove(self._hold)
        if not sent:
            await _undo(self._grant._store, self._hold)
            return
        estimates, self._hold.estimates = self._hold.estimates, []
        for key, handle in estimates:
            await self._grant._store.settle(key, handle, 0)
        leases, self._hold.leases = self._hold.leases, []
        for key, lease in leases:
            await self._grant._store.release(key, lease)


class RateLimitGrant:
    """What one request holds under the ``rate_limits`` rules: its token estimates and concurrency slots.

    Estimates are settled on what the request used. Slots are given back by
    :class:`RateLimitGrantMiddleware` when the response ends, however it ends.
    """

    def __init__(
        self, store: RateLimitStorePort, config: "GatewayConfig | None" = None, estimated_tokens: int = 0
    ) -> None:
        self._store = store
        self._config = config
        self._estimated_tokens = estimated_tokens
        self._holds: list[_Hold] = []
        self.handed_over = False

    def hand_over(self) -> None:
        """Mark the request dispatched, so its own paths settle the estimates from here."""
        self.handed_over = True

    async def settle(self, tokens: int) -> None:
        """Charge every token estimate ``tokens`` instead. Only the first call counts."""
        for hold in self._holds:
            estimates, hold.estimates = hold.estimates, []
            for key, handle in estimates:
                await self._store.settle(key, handle, max(tokens, 0))

    async def release(self) -> None:
        """Give back every concurrency slot. Only the first call counts."""
        for hold in self._holds:
            leases, hold.leases = hold.leases, []
            for key, lease in leases:
                await self._store.release(key, lease)

    async def admit_model(self, instance: str, model: str, *, name_model: bool = True) -> ModelHold | None:
        """Count an attempt on ``instance:model`` against every ``per: model`` rule naming it.

        ``None`` when no rule names it. A refused attempt is counted by no rule.
        ``name_model`` puts the model in the refusal; a routing policy passes
        ``False``, because its targets are not the caller's to see.

        Raises:
            HTTPException: 429 naming the first rule the attempt does not fit.

        """
        if self._config is None:
            return None
        name = f"{instance}:{model}"
        rules = [
            rule for rule in tuple(self._config.rate_limits) if rule.per == "model" and name in (rule.models or ())
        ]
        if not rules:
            return None
        hold = _Hold()
        label = f" for {name}" if name_model else ""
        for rule in rules:
            try:
                await _count_rule(self._store, rule, name, self._estimated_tokens, hold, label=label)
            except BaseException as exc:
                # The hold is not on the grant yet, so nothing else would give back what it took.
                if isinstance(exc, HTTPException):
                    RATE_LIMIT_MODEL_FULL.labels(rule=rule.name, model=name).inc()
                await _undo(self._store, hold)
                raise
        self._holds.append(hold)
        return ModelHold(self, hold)


def _count(n: int, noun: str) -> str:
    """``n`` of ``noun``, as a person would write it: "1 request", "2,000 tokens"."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _refused(detail: str, retry_after: float | None, rule: str) -> HTTPException:
    """A 429 with ``detail``, without ``Retry-After`` when no wait would let the request in."""
    RATE_LIMIT_HITS.inc()
    headers = error_headers(RATE_LIMITED, rule=rule)
    if retry_after is not None:
        headers["Retry-After"] = str(max(math.ceil(retry_after), 1))
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail, headers=headers)


async def _count_rule(
    store: RateLimitStorePort,
    rule: "RateLimitRule",
    subject: str,
    estimated_tokens: int,
    hold: _Hold,
    *,
    label: str = "",
) -> None:
    """Count one request against ``rule`` for ``subject``, adding what it took to ``hold``.

    ``label`` follows the rule's name in a refusal, to say which model it was counted for.

    Raises:
        HTTPException: 429 when the request does not fit; what it took so far stays in ``hold``.

    """
    base = f"rule:{rule.name}:{subject}"
    if rule.rpm is not None:
        window = await store.hit(f"{base}:rpm", rule.rpm, _RULE_WINDOW_SEC)
        if window.handle is None:
            raise _refused(
                f"Rate limit '{rule.name}'{label} exceeded: {_count(rule.rpm, 'request')} per minute",
                window.reset_after,
                rule.name,
            )
        hold.entries.append((f"{base}:rpm", window.handle))
    if rule.tpm is not None:
        # At least one token, so a request estimating none is still refused by a full window.
        cost = max(estimated_tokens, 1)
        if cost > rule.tpm:
            msg = (
                f"Request needs an estimated {_count(cost, 'token')}; "
                f"rate limit '{rule.name}'{label} allows {rule.tpm:,} per minute"
            )
            raise _refused(msg, None, rule.name)
        window = await store.hit(f"{base}:tpm", rule.tpm, _RULE_WINDOW_SEC, cost=cost)
        if window.handle is None:
            raise _refused(
                f"Rate limit '{rule.name}'{label} exceeded: {_count(rule.tpm, 'token')} per minute",
                window.reset_after,
                rule.name,
            )
        hold.entries.append((f"{base}:tpm", window.handle))
        hold.estimates.append((f"{base}:tpm", window.handle))
    if rule.max_concurrent is not None:
        lease = await store.acquire(f"{base}:concurrent", rule.max_concurrent, rule.lease_sec)
        if lease is None:
            raise _refused(
                f"Rate limit '{rule.name}'{label} exceeded: {_count(rule.max_concurrent, 'request')} in flight",
                _CONCURRENCY_RETRY_AFTER_SEC,
                rule.name,
            )
        hold.leases.append((f"{base}:concurrent", lease))


async def _undo(store: RateLimitStorePort, hold: _Hold) -> None:
    """Uncount everything ``hold`` took, for a request that was refused."""
    entries, hold.entries, hold.estimates = hold.entries, [], []
    for key, handle in entries:
        await store.settle(key, handle, 0)
    leases, hold.leases = hold.leases, []
    for key, lease in leases:
        await store.release(key, lease)


class RateLimitRules:
    """The ``rate_limits`` rules, counted in one store.

    Read from the config on every request, so a rule the dashboard adds or edits
    applies from the next request without rebuilding anything.
    """

    def __init__(self, store: RateLimitStorePort, config: "GatewayConfig") -> None:
        self._store = store
        self._config = config

    @property
    def active(self) -> bool:
        """Whether any rule is in effect."""
        return bool(self._config.rate_limits)

    async def admit(
        self, request: Request, *, key_id: str | None, user_id: str | None, estimated_tokens: int
    ) -> RateLimitGrant:
        """Count a request against every rule that applies to it, or against none.

        A refused request is not counted anywhere, so retrying it does not use
        up the limits it did fit. ``per: model`` rules are counted later, by
        :meth:`RateLimitGrant.admit_model`, once an attempt names its model.

        Raises:
            HTTPException: 429 naming the first rule the request does not fit.

        """
        grant = RateLimitGrant(self._store, self._config, estimated_tokens)
        # Before any slot is taken, so a request that dies here still gives them back.
        setattr(request.state, _GRANT_STATE, grant)
        subjects = {"deployment": "all", "key": key_id, "user": user_id}
        hold = _Hold()
        grant._holds.append(hold)
        try:
            for rule in tuple(self._config.rate_limits):
                subject = subjects.get(rule.per)
                if subject is None:
                    continue
                await _count_rule(self._store, rule, subject, estimated_tokens, hold)
        except HTTPException:
            await _undo(self._store, hold)
            raise
        return grant


async def admit_rate_limit_rules(
    request: Request, *, key_id: str | None, user_id: str | None, estimated_tokens: int
) -> RateLimitGrant | None:
    """Count a request against the deployment's ``rate_limits``, or ``None`` when it has none."""
    rules: RateLimitRules | None = getattr(request.app.state, "rate_limit_rules", None)
    if rules is None or not rules.active:
        return None
    return await rules.admit(request, key_id=key_id, user_id=user_id, estimated_tokens=estimated_tokens)


async def _close_grant(grant: RateLimitGrant) -> None:
    if not grant.handed_over:
        await grant.settle(0)
    await grant.release()


class RateLimitGrantMiddleware:
    """Gives back a request's concurrency slots once its response has ended.

    The ASGI call returns only after a response's last byte is sent or its
    client has gone, so this one ``finally`` covers a streamed body that
    outlives its handler as well as every way a request can fail. A request
    refused before it was handed over (by its budget, say) is charged no tokens.
    """

    def __init__(self, app: "ASGIApp") -> None:
        self.app = app

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        try:
            await self.app(scope, receive, send)
        finally:
            grant: RateLimitGrant | None = scope.get("state", {}).get(_GRANT_STATE)
            if grant is not None:
                try:
                    # Shielded: this can run while a disconnect is cancelling the request.
                    await asyncio.shield(_close_grant(grant))
                except Exception:
                    logger.exception("Could not give back a request's concurrency slots; their leases will run out")
