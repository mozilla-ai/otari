"""Bounded, process-local feedback for automatic same-model provider selection."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from statistics import median
from typing import TYPE_CHECKING

from google.oauth2.service_account import Credentials
from starlette.exceptions import HTTPException

from gateway.core.settings.routing import AutomaticOffering, AutomaticRouting, Region
from gateway.services.inference import (
    is_provider_credential_rejection,
    upstream_error_message,
    upstream_exception_shape,
)

if TYPE_CHECKING:
    from gateway.services.routing.compiler import CompiledPlan
    from gateway.types.attempt import Attempt

# EU member countries, not a guess from language, time zone or an IP address.
_EU = frozenset("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
_TTL = 300.0
_LIMIT = 10000
_COOLDOWN = 15.0
_MAX_COOLDOWN = 600.0
_REFUSAL_TTL = 300.0
# A model still missing once its refusal lapses is refused for longer, up to the last step.
_MISSING_MODEL_STEPS = (3600.0, 3 * 3600.0, 12 * 3600.0)
# How a provider's message says the model is not served on this credential. Request text a provider
# echoes back could match too; the cost is one offering ranked last for a while, never one switched off.
_MISSING_MODEL = re.compile(
    r"model\b.{0,120}\b(does not exist|not found|not accessible)"
    r"|(do not|don't) have access to (the )?model|model not found",
    re.IGNORECASE | re.DOTALL,
)
# A 404 naming something the request referenced, such as a previous response or an uploaded file, is not the model's.
_REQUEST_RESOURCE = re.compile(
    r"\b(file|batch|response|conversation|thread|vector[ _]store|container)s?\b", re.IGNORECASE
)
_CREDENTIAL_FIELD = re.compile(r"key|token|secret|credential|password|auth", re.IGNORECASE)
_ACCOUNT_FIELDS = frozenset(
    {
        "api_base",
        "base_url",
        "endpoint_url",
        "project",
        "location",
        "region_name",
        "organization",
        "account_id",
        "default_headers",
    }
)


def _credential_value(value: object) -> object:
    """Normalize known credential identities; isolate unsupported objects with a nonce."""
    if isinstance(value, Credentials):
        return (
            value.service_account_email,
            value.signer.key_id or secrets.token_hex(16),
            value.quota_project_id,
            value.universe_domain,
            value.scopes,
        )
    if isinstance(value, Mapping):
        return {str(key): _credential_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_credential_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Unknown credential implementations cannot safely inherit a binding.
    return secrets.token_hex(16)


def _credential_fields(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Stable authentication and account inputs, excluding SDK transport objects."""
    fields: dict[str, object] = {}
    for name, value in kwargs.items():
        if name == "client_args" and isinstance(value, Mapping):
            fields[name] = _credential_fields(value)
        elif name in _ACCOUNT_FIELDS or _CREDENTIAL_FIELD.search(name):
            fields[name] = _credential_value(value)
    return fields


def preferred_region(settings: AutomaticRouting, headers: Mapping[str, str]) -> Region | None:
    """Explicit preference, trusted edge country, then deployment default."""
    explicit = headers.get("otari-region", "").lower()
    if explicit in ("us", "eu"):
        return "us" if explicit == "us" else "eu"
    country = headers.get(settings.country_header, "").upper() if settings.country_header else ""
    if country == "US":
        return "us"
    if country in _EU:
        return "eu"
    return settings.default_region


@dataclass
class _Health:
    active: int = 0
    latency: dict[str, tuple[float, float]] = field(default_factory=dict)
    cooldown: float = 0
    failures: int = 0


def _backoff(failures: int) -> float:
    """Cooldown after ``failures`` consecutive failures: 15 seconds, doubling up to 10 minutes."""
    # Capped before exponentiation, so a long outage's count cannot overflow the float.
    return min(_COOLDOWN * 2.0 ** min(failures - 1, 6), _MAX_COOLDOWN)


def _consecutive(failures: int, cooled_until: float, now: float) -> int:
    """The failure count including this one, restarted when the last cooldown ended long ago."""
    return 1 if now - cooled_until > _MAX_COOLDOWN else failures + 1


@dataclass
class AutomaticSelection:
    """Private request context shared with the dispatch observer."""

    router: AutomaticRouter
    affinity: str | None
    mode: str
    scope: str = ""
    tenant: str = ""
    observations: list[AutomaticObservation] = field(default_factory=list)
    origins: dict[str, Attempt] = field(default_factory=dict)

    def dispatched_as(self, planned: Attempt, dispatched: Attempt) -> None:
        """Record outcomes of ``dispatched``, served on a credential resolved at dispatch, as ``planned``.

        Ranking only sees the offering as planned, so health, cooldowns and session binding must be kept
        under that identity for them to reach the next ranking.
        """
        self.origins[self.router.candidate_key(dispatched)] = planned

    def _ranked(self, attempt: Attempt) -> Attempt:
        """The offering ``attempt`` was ranked as."""
        return self.origins.get(self.router.candidate_key(attempt), attempt)

    def forget(self, attempt: Attempt) -> None:
        """Invalidate only a binding to the refused candidate, not a newer one."""
        self.forget_key(self.router.candidate_key(self._ranked(attempt)))

    def refuse(self, attempt: Attempt) -> None:
        """Rank an offering that refused this caller after the others until the refusal expires."""
        self.forget(attempt)
        self.router.remember_refusal(self.scope, f"{attempt.instance}:{attempt.model}")

    def forget_key(self, key: str) -> None:
        """Remove the session binding only if it still points to this credential fingerprint."""
        previous = self.router.affinities.get(self.affinity) if self.affinity else None
        if self.affinity and previous and previous[0] == key:
            del self.router.affinities[self.affinity]

    def close(self) -> None:
        """Release every outstanding observation without recording success or failure."""
        for observation in self.observations:
            observation.finish()

    def start(self, attempt: Attempt) -> AutomaticObservation:
        """Track active load for the offering dispatched and return its observer."""
        attempt = self._ranked(attempt)
        selector = f"{attempt.instance}:{attempt.model}"
        key = self.router.candidate_key(attempt)
        health = self.router._health(key)
        if health is not None:
            health.active += 1
        account = self.router.account_key(attempt, self.tenant)
        observation = AutomaticObservation(self, key, account, selector, health, self.router.clock())
        self.observations.append(observation)
        return observation


@dataclass
class AutomaticObservation:
    """One dispatched call; finish is idempotent on every terminal path."""

    selection: AutomaticSelection
    key: str
    account: str
    selector: str
    health: _Health | None
    started: float
    first_at: float | None = None
    finished: bool = False

    def first_response(self) -> None:
        """Record the first output time once for streaming latency feedback."""
        if self.first_at is None:
            self.first_at = self.selection.router.clock()

    def finish(
        self, *, success: bool = False, unhealthy: bool = False, account: bool = False, missing: bool = False
    ) -> None:
        """Release load once, then update health and session affinity for the reported outcome.

        ``account`` marks a failure of the credential itself (exhausted quota or funds, a rejected key),
        which cools every model served on it rather than this one alone. It applies only with ``unhealthy``,
        so a gateway-side refusal that carries the same status never cools a shared credential.
        ``missing`` marks a model the credential does not serve, which ranks last for the workspace for a while
        (:meth:`AutomaticRouter.remember_missing`) instead of counting against the offering's health or credential.
        """
        if self.finished:
            return
        self.finished = True
        router = self.selection.router
        now = router.clock()
        if missing:
            unhealthy = account = False
            router.remember_missing(self.selection.tenant, self.selection.mode, self.key)
            self.selection.forget_key(self.key)
        # A failed credential is the account's, not the offering's: it cools the account alone.
        credential = account and unhealthy
        if self.health is not None:
            self.health.active = max(0, self.health.active - 1)
            if unhealthy and not credential:
                self.health.failures = _consecutive(self.health.failures, self.health.cooldown, now)
                self.health.cooldown = now + _backoff(self.health.failures)
            if success:
                # Streaming measures first output; non-streaming measures completion.
                elapsed = max(0.001, (self.first_at if self.first_at is not None else now) - self.started)
                old = self.health.latency.get(self.selection.mode)
                value = 0.8 * old[0] + 0.2 * elapsed if old and now - old[1] < _TTL else elapsed
                self.health.latency[self.selection.mode] = (value, now)
                self.health.cooldown = 0
                self.health.failures = 0
        if credential:
            router.cool_account(self.account, now)
        elif success:
            router.accounts.pop(self.account, None)
            router.missing.pop(router._digest(("missing", self.selection.tenant, self.selection.mode, self.key)), None)
        if unhealthy:
            self.selection.forget_key(self.key)
        if success and self.selection.affinity:
            router.affinities[self.selection.affinity] = (self.key, now)
            router.affinities.move_to_end(self.selection.affinity)
            while len(router.affinities) > _LIMIT:
                router.affinities.popitem(last=False)


class AutomaticRouter:
    """Ranks eligible offerings without changing credential or budget authority.

    No prompts or credentials are retained. State is per gateway worker and
    observations expire; admission remains the authoritative capacity check.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Initialize private feedback state with an injectable monotonic clock."""
        self.clock = clock
        self._salt = secrets.token_bytes(32)
        self.health: OrderedDict[str, _Health] = OrderedDict()
        self.affinities: OrderedDict[str, tuple[str, float]] = OrderedDict()
        # Credential-wide cooldowns as (until, consecutive failures), and per-caller refusals as expiry times.
        self.accounts: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self.refusals: OrderedDict[str, float] = OrderedDict()
        # Models a credential does not serve, as (until, step reached).
        self.missing: OrderedDict[str, tuple[float, int]] = OrderedDict()

    def _digest(self, value: object) -> str:
        """Create a private process-local fingerprint without retaining its input."""
        payload = json.dumps(value, sort_keys=True).encode()
        # Keyed, process-local fingerprints are never persisted as password verifiers.
        # codeql[py/weak-sensitive-data-hashing]
        return hmac.new(self._salt, payload, hashlib.sha256).hexdigest()

    def candidate_key(self, attempt: Attempt) -> str:
        """Identify the actual credential, including one resolved at dispatch."""
        return self._digest(
            (attempt.instance, attempt.provider.value, attempt.model, _credential_fields(attempt.kwargs))
        )

    @staticmethod
    def is_account_failure(exc: BaseException) -> bool:
        """Whether the credential itself failed, not one model: exhausted quota or funds (402), or a refused key."""
        return upstream_exception_shape(exc)[1] == 402 or is_provider_credential_rejection(exc)

    @staticmethod
    def is_missing_model(exc: BaseException) -> bool:
        """Whether the provider does not serve the model on this credential: a 404, or a 400 or 403 saying so.

        A 404 counts unless it names something the request referenced and not the model.
        A status the gateway raised itself is never the provider's.
        """
        if isinstance(exc, HTTPException):
            return False
        status_code = upstream_exception_shape(exc)[1]
        if status_code not in (400, 403, 404):
            return False
        message = upstream_error_message(exc)
        if _MISSING_MODEL.search(message):
            return True
        if status_code != 404:
            return False
        return re.search(r"\bmodel", message, re.IGNORECASE) is not None or not _REQUEST_RESOURCE.search(message)

    def account_key(self, attempt: Attempt, tenant: str = "") -> str:
        """Identify the credential an attempt is sent on, whatever model it names, for one ``tenant``.

        Dispatch may resolve a different credential per organization behind the same planned fields,
        so a credential's failure is kept to the tenant that saw it rather than shared by everyone.
        """
        return self._digest((attempt.instance, attempt.provider.value, _credential_fields(attempt.kwargs), tenant))

    def cool_account(self, key: str, now: float) -> None:
        """Cool every model on a failing credential, backing off further while it keeps failing."""
        previous = self.accounts.pop(key, None)
        failures = _consecutive(previous[1], previous[0], now) if previous else 1
        self.accounts[key] = (now + _backoff(failures), failures)
        while len(self.accounts) > _LIMIT:
            self.accounts.popitem(last=False)

    def remember_refusal(self, scope: str, selector: str) -> None:
        """Record that ``selector`` refused callers in ``scope``, such as hosted access their tenant lacks."""
        key = self._digest((scope, selector))
        self.refusals.pop(key, None)
        self.refusals[key] = self.clock() + _REFUSAL_TTL
        while len(self.refusals) > _LIMIT:
            self.refusals.popitem(last=False)

    def remember_missing(self, tenant: str, mode: str, candidate: str) -> None:
        """Record that the offering ``candidate`` identifies, on its credential, does not serve ``tenant`` in ``mode``.

        Refused for an hour, then three, then twelve while it is still missing each time a refusal lapses.
        A failure during a refusal restarts it at the same length; one long after it lapsed starts over.
        """
        key = self._digest(("missing", tenant, mode, candidate))
        now = self.clock()
        previous = self.missing.pop(key, None)
        step = 0
        if previous is not None and now - previous[0] <= _MISSING_MODEL_STEPS[-1]:
            step = previous[1] if now < previous[0] else min(previous[1] + 1, len(_MISSING_MODEL_STEPS) - 1)
        self.missing[key] = (now + _MISSING_MODEL_STEPS[step], step)
        while len(self.missing) > _LIMIT:
            self.missing.popitem(last=False)

    def _health(self, key: str) -> _Health | None:
        """Get tracked health, evicting an idle entry when full; never evict active calls."""
        if key in self.health:
            self.health.move_to_end(key)
            return self.health[key]
        if len(self.health) >= _LIMIT:
            idle = next((key for key, value in self.health.items() if not value.active), None)
            if idle is None:
                return None
            del self.health[idle]
        value = _Health()
        self.health[key] = value
        return value

    @staticmethod
    def accepts(plan: CompiledPlan, model: str, settings: AutomaticRouting) -> bool:
        """Whether this plan opts into automatic, unpinned catalog selection."""
        return (
            plan.selection_reason == "catalog"
            and settings.enabled
            and not any(f"{attempt.instance}/{attempt.model}" == model for attempt in plan.attempts)
        )

    def rank(
        self,
        plan: CompiledPlan,
        *,
        settings: AutomaticRouting,
        costs: Mapping[str, Decimal | None],
        scope: str,
        anchor: str,
        region: Region | None,
        mode: str,
        own: frozenset[str] = frozenset(),
        limit: int = 5,
        tenant: str = "",
    ) -> CompiledPlan:
        """Order by price availability, health, own key, region and cost/load/latency, preserving eligible affinity.

        ``own`` holds the selectors the caller's organization serves on its own provider keys. A priced,
        healthy own key leads every other offering, so the organization's key is used before one the
        deployment pays for; the other signals order the offerings within each tier.
        """
        from gateway.services.routing.compiler import DroppedCandidate

        now = self.clock()
        keys = {f"{a.instance}:{a.model}": self.candidate_key(a) for a in plan.attempts}
        cooling: set[str] = set()
        refused: set[str] = set()
        for attempt in plan.attempts:
            selector = f"{attempt.instance}:{attempt.model}"
            health = self.health.get(keys[selector])
            account = self.accounts.get(self.account_key(attempt, tenant))
            if (health and health.cooldown > now) or (account and account[0] > now):
                cooling.add(selector)
            missing = self.missing.get(self._digest(("missing", tenant, mode, keys[selector])))
            if self.refusals.get(self._digest((scope, selector)), 0) > now or (missing and missing[0] > now):
                refused.add(selector)
        affinity = self._digest((scope, plan.policy_name, anchor)) if anchor else None
        previous = self.affinities.get(affinity) if affinity else None
        sticky = previous[0] if previous and now - previous[1] < settings.session_ttl_seconds else None
        latencies: dict[str, float] = {}
        for selector, key in keys.items():
            health = self.health.get(key)
            value = health.latency.get(mode) if health else None
            if value and now - value[1] < _TTL:
                latencies[selector] = value[0]
        neutral = median(latencies.values()) if latencies else 1.0

        def group(attempt: Attempt) -> tuple[bool, bool, bool, bool, int]:
            """Partition by a known refusal, missing price, cooldown, ownership and region mismatch.

            A refusal comes first: an unpriced offering may still serve, one that refused this caller will not.
            """
            selector = f"{attempt.instance}:{attempt.model}"
            hint = settings.offerings.get(selector, settings.offerings.get(attempt.instance, AutomaticOffering()))
            locality = 0 if region is None or hint.region == region else 1
            return selector in refused, costs.get(selector) is None, selector in cooling, selector not in own, locality

        floors: dict[tuple[bool, bool, bool, bool, int], Decimal] = {}
        for attempt in plan.attempts:
            cost = costs.get(f"{attempt.instance}:{attempt.model}")
            if cost is not None:
                bucket = group(attempt)
                floors[bucket] = min(floors.get(bucket, cost), cost)

        def score(attempt: Attempt) -> tuple[bool, bool, bool, bool, int, float, int]:
            """Score cost and load within a health/region group, breaking ties by catalog order."""
            selector = f"{attempt.instance}:{attempt.model}"
            hint = settings.offerings.get(selector, settings.offerings.get(attempt.instance, AutomaticOffering()))
            health = self.health.get(keys[selector])
            cost = costs.get(selector)
            load = health.active / hint.capacity if health else 0.0
            latency = min(3.0, latencies.get(selector, neutral) / max(neutral, 0.001))
            bucket = group(attempt)
            floor = max(floors.get(bucket, Decimal(0)), Decimal("0.000001"))
            value = float(cost / floor) if cost is not None else 1.0
            return (
                *bucket,
                value + 0.5 * load + 0.25 * latency,
                attempt.position,
            )

        ranked = sorted(plan.attempts, key=score)
        bound = next((a for a in ranked if keys[f"{a.instance}:{a.model}"] == sticky), None)
        if bound is not None:
            refusing, unpriced, unhealthy, foreign, locality = group(bound)
            if (
                not refusing
                and not unhealthy
                and locality == 0
                and (not unpriced or all(group(a)[1] for a in ranked))
                and (not foreign or group(ranked[0])[3])
            ):
                ranked.remove(bound)
                ranked.insert(0, bound)
            else:
                bound = None
        if affinity and previous and bound is None:
            self.affinities.pop(affinity, None)
        ordered = ranked[:limit]
        attempts = [
            replace(a, position=i, selection_reason="catalog" if i == 1 else "on_failure")
            for i, a in enumerate(ordered, 1)
        ]
        return replace(
            plan,
            attempts=attempts,
            automatic=AutomaticSelection(self, affinity, mode, scope, tenant),
            dropped=[
                *plan.dropped,
                *(
                    DroppedCandidate(f"{a.instance}:{a.model}", "over_cap", f"below the best {limit} offerings")
                    for a in ranked[limit:]
                ),
            ],
        )
