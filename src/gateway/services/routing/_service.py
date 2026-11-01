"""The routing domain's service: what callers get from routing, read for one organization."""

import uuid
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.api_keys import ApiKeyService
from gateway.services.catalog import catalog_models, offering_input_rate
from gateway.services.tools import provider_tools
from gateway.services.usage_and_telemetry import ServedTraffic, UsageService

FALLBACK_WINDOW = timedelta(days=30)
"""How far back the automatic fallbacks look for traffic."""


@dataclass(frozen=True)
class FallbackOffering:
    """One offering a catalog ID tries, and who runs the built-in tools on it."""

    selector: str
    provider: str
    """The instance the selector names."""
    provider_type: str
    input_price: float | None
    """Per million input tokens at this organization's rates, None where unpriced."""
    own_tools: tuple[str, ...]
    """Built-in tools the provider runs itself."""
    otari_tools: tuple[str, ...]
    """Built-in tools Otari runs in the provider's place."""


@dataclass(frozen=True)
class PinnedTraffic:
    """The requests one API key sent to one provider by naming it, so with no automatic fallback."""

    api_key_id: str | None
    """None for requests made without a key or whose key was deleted."""
    key_name: str | None
    provider: str
    requests: int


@dataclass(frozen=True)
class AutomaticFallback:
    """One model callers used in the window, its offerings in the order a catalog ID tries them, and its traffic."""

    id: str
    name: str
    offerings: tuple[FallbackOffering, ...]
    requests: int
    fell_back: int
    """Of ``requests``, the successful ones a later offering served."""
    pinned: int
    """Of ``requests``, those that did not send the catalog ID and so had no automatic fallback."""
    pinned_provider: str | None
    """The instance most of the pinned requests reached."""
    pinned_keys: tuple[PinnedTraffic, ...]
    """The pinned requests by key and provider, most first."""


@dataclass
class _Tally:
    requests: int = 0
    fell_back: int = 0
    pinned: Counter[str] = field(default_factory=Counter)
    pinned_keys: Counter[tuple[str | None, str]] = field(default_factory=Counter)


def _served_selectors(row: ServedTraffic) -> tuple[str, ...]:
    """The spellings of the offering a row reached: ``provider:model``, or the whole selector an older row carries."""
    return (f"{row.provider}:{row.model}", row.model) if row.provider else (row.model,)


class RoutingService:
    """What routing gives an organization's callers."""

    def __init__(self, config: GatewayConfig, usage: UsageService, uow: UnitOfWork, keys: ApiKeyService) -> None:
        self._config = config
        self._usage = usage
        self._uow = uow
        self._keys = keys

    async def automatic_fallbacks(
        self, *, organization_id: uuid.UUID, workspace_ids: Collection[uuid.UUID]
    ) -> list[AutomaticFallback]:
        """Every model these workspaces sent requests to in the window, most requested first.

        A request that sent the model's catalog ID tried its offerings in the
        catalog's order for this organization (``resolve_catalog_offerings``),
        moving to the next when one failed; one that named an offering itself
        (``bedrock:…``) or a routing policy is counted as pinned. A row is
        matched to a model by the catalog ID it sent, else by the offering it
        reached, so traffic on an offering the catalog no longer serves is left
        out. The order is the catalog's, before a key's allow-list or a missing
        credential drops an offering from a request's plan.
        """
        traffic = await self._usage.served_traffic(
            workspace_ids=workspace_ids, since=datetime.now(UTC) - FALLBACK_WINDOW
        )
        models = {model.id: model for model in catalog_models(organization_id=organization_id)}
        model_of = {selector: model.id for model in models.values() for selector in model.offerings}
        tallies: dict[str, _Tally] = {}
        for row in traffic:
            sent = (row.policy_name or "").lower()
            reached = next((selector for selector in _served_selectors(row) if selector in model_of), None)
            model_id = sent if sent in models else model_of.get(reached or "")
            if model_id is None:
                continue
            tally = tallies.setdefault(model_id, _Tally())
            tally.requests += row.requests
            tally.fell_back += row.fell_back
            if sent != model_id:
                provider = row.provider or (reached or row.model).partition(":")[0]
                tally.pinned[provider] += row.requests
                tally.pinned_keys[(row.api_key_id, provider)] += row.requests

        key_ids = {key_id for tally in tallies.values() for key_id, _provider in tally.pinned_keys if key_id}
        async with self._uow:
            key_names = await self._keys.key_names(key_ids)

        tools: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

        def offering(selector: str) -> FallbackOffering:
            instance = selector.partition(":")[0]
            provider_type = self._config.provider_instance_type(instance)
            if provider_type not in tools:
                answer = provider_tools(self._config, provider_type)
                tools[provider_type] = (tuple(sorted(answer.native)), tuple(sorted(answer.claimed)))
            own, otari = tools[provider_type]
            return FallbackOffering(
                selector=selector,
                provider=instance,
                provider_type=provider_type,
                input_price=offering_input_rate(selector, organization_id=organization_id),
                own_tools=own,
                otari_tools=otari,
            )

        def pinned_keys(tally: _Tally) -> tuple[PinnedTraffic, ...]:
            entries = [
                PinnedTraffic(api_key_id=key_id, key_name=key_names.get(key_id or ""), provider=provider, requests=n)
                for (key_id, provider), n in tally.pinned_keys.items()
            ]
            return tuple(sorted(entries, key=lambda entry: (-entry.requests, entry.key_name or "", entry.provider)))

        fallbacks = [
            AutomaticFallback(
                id=model_id,
                name=models[model_id].name,
                offerings=tuple(offering(selector) for selector in models[model_id].offerings),
                requests=tally.requests,
                fell_back=tally.fell_back,
                pinned=sum(tally.pinned.values()),
                pinned_provider=tally.pinned.most_common(1)[0][0] if tally.pinned else None,
                pinned_keys=pinned_keys(tally),
            )
            for model_id, tally in tallies.items()
            if tally.requests > 0
        ]
        return sorted(fallbacks, key=lambda fallback: (-fallback.requests, fallback.id))
