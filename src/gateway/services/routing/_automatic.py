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

from gateway.core.settings.routing import AutomaticOffering, AutomaticRouting, Region

if TYPE_CHECKING:
    from gateway.services.routing.compiler import CompiledPlan
    from gateway.types.attempt import Attempt

# EU member countries, not a guess from language, time zone or an IP address.
_EU = frozenset("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
_TTL = 300.0
_LIMIT = 10000
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


@dataclass
class AutomaticSelection:
    """Private request context shared with the dispatch observer."""

    router: AutomaticRouter
    affinity: str | None
    mode: str
    observations: list[AutomaticObservation] = field(default_factory=list)

    def forget(self, attempt: Attempt) -> None:
        """Invalidate only a binding to the refused candidate, not a newer one."""
        self.forget_key(self.router.candidate_key(attempt))

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
        """Track active load for the credential actually dispatched and return its observer."""
        selector = f"{attempt.instance}:{attempt.model}"
        key = self.router.candidate_key(attempt)
        health = self.router._health(key)
        if health is not None:
            health.active += 1
        observation = AutomaticObservation(self, key, selector, health, self.router.clock())
        self.observations.append(observation)
        return observation


@dataclass
class AutomaticObservation:
    """One dispatched call; finish is idempotent on every terminal path."""

    selection: AutomaticSelection
    key: str
    selector: str
    health: _Health | None
    started: float
    first_at: float | None = None
    finished: bool = False

    def first_response(self) -> None:
        """Record the first output time once for streaming latency feedback."""
        if self.first_at is None:
            self.first_at = self.selection.router.clock()

    def finish(self, *, success: bool = False, unhealthy: bool = False) -> None:
        """Release load once, then update health and session affinity for the reported outcome."""
        if self.finished:
            return
        self.finished = True
        router = self.selection.router
        now = router.clock()
        if self.health is not None:
            self.health.active = max(0, self.health.active - 1)
            if unhealthy:
                self.health.cooldown = now + 15
            if success:
                # Streaming measures first output; non-streaming measures completion.
                elapsed = max(0.001, (self.first_at if self.first_at is not None else now) - self.started)
                old = self.health.latency.get(self.selection.mode)
                value = 0.8 * old[0] + 0.2 * elapsed if old and now - old[1] < _TTL else elapsed
                self.health.latency[self.selection.mode] = (value, now)
                self.health.cooldown = 0
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
        limit: int = 5,
    ) -> CompiledPlan:
        """Order by price availability, health, region and cost/load/latency, preserving eligible affinity."""
        from gateway.services.routing.compiler import DroppedCandidate

        now = self.clock()
        keys = {f"{a.instance}:{a.model}": self.candidate_key(a) for a in plan.attempts}
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

        def group(attempt: Attempt) -> tuple[bool, bool, int]:
            """Partition candidates by missing price, cooldown and region mismatch."""
            selector = f"{attempt.instance}:{attempt.model}"
            hint = settings.offerings.get(selector, settings.offerings.get(attempt.instance, AutomaticOffering()))
            health = self.health.get(keys[selector])
            locality = 0 if region is None or hint.region == region else 1
            return costs.get(selector) is None, bool(health and health.cooldown > now), locality

        floors: dict[tuple[bool, bool, int], Decimal] = {}
        for attempt in plan.attempts:
            cost = costs.get(f"{attempt.instance}:{attempt.model}")
            if cost is not None:
                bucket = group(attempt)
                floors[bucket] = min(floors.get(bucket, cost), cost)

        def score(attempt: Attempt) -> tuple[bool, bool, int, float, int]:
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
            unpriced, unhealthy, locality = group(bound)
            if not unhealthy and locality == 0 and (not unpriced or all(group(a)[0] for a in ranked)):
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
            automatic=AutomaticSelection(self, affinity, mode),
            dropped=[
                *plan.dropped,
                *(
                    DroppedCandidate(f"{a.instance}:{a.model}", "over_cap", f"below the best {limit} offerings")
                    for a in ranked[limit:]
                ),
            ],
        )
