"""What a request for an unpriced model is told, and whether its provider can be called at all."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

from any_llm import LLMProvider

from gateway.core.error_codes import PRICING_REQUIRED
from gateway.log_config import logger
from gateway.services.model_access import ModelRefusal
from gateway.services.pricing_service import no_pricing_error_detail
from gateway.services.provider_kwargs import (
    MANAGED_SURFACE_ROUTING_POLICY,
    ManagedCredentialUnsupportedError,
    MissingCredential,
    missing_declared_credential,
)
from gateway.services.providers._hosted_runtime import organization_to_serve

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.ports.model_provider_port import ModelProviderPort
    from gateway.services.provider_kwargs import ResolvedProvider
    from gateway.services.routing import CompiledPlan
    from gateway.types.attempt import Attempt


async def missing_provider_credential(
    config: GatewayConfig,
    resolved: ResolvedProvider,
    *,
    hosted: ModelProviderPort | None,
    organization_id: uuid.UUID | None,
) -> MissingCredential | None:
    """The missing credential a dispatch of ``resolved`` would be refused for, or ``None``.

    ``None`` whenever something may still serve the candidate: a rung of the
    credential ladder, the provider's own environment variable under a
    ``providers:`` entry, a provider that needs no key, the caller's own
    endpoint, or, where ``hosted`` is asked at dispatch, a hosted provider it
    lists for the organization. ``hosted`` is ``None`` for a path whose dispatch
    never asks it.
    """
    if resolved.owned_endpoint is not None:
        return None
    missing = missing_declared_credential(config, resolved.instance, resolved.provider, resolved.kwargs)
    if missing is None:
        return None
    if hosted is not None:
        offered = await hosted.get_hosted_models(organization_id=organization_to_serve(organization_id))
        if resolved.instance in offered:
            return None
    return missing.behind_alias(resolved.alias)


class Candidate(Protocol):
    """What :func:`managed_only_candidates` reads of a candidate: a ``ResolvedProvider`` or a plan ``Attempt``."""

    @property
    def instance(self) -> str: ...
    @property
    def provider(self) -> LLMProvider: ...
    @property
    def model(self) -> str: ...
    @property
    def kwargs(self) -> dict[str, Any]: ...


async def managed_only_candidates(
    config: GatewayConfig,
    candidates: Iterable[Candidate],
    *,
    hosted: Callable[[], ModelProviderPort],
    organization_id: uuid.UUID | None,
) -> frozenset[tuple[str, str]]:
    """The ``(instance, model)`` pairs among ``candidates`` only the deployment's managed credentials would serve.

    For a path that never asks the port to serve: a candidate nothing declared
    serves, whose model the port advertises to the organization, is refused
    there by name rather than as a provider nobody configured. ``hosted`` builds
    the port, and is called only when some candidate lacks a declared
    credential; only its listing is read, never ``resolve_hosted_credential``,
    which may meter a credential that would not serve. A failure to list
    answers nothing, leaving the ordinary refusal.
    """
    uncredentialed = [
        candidate
        for candidate in candidates
        if missing_declared_credential(config, candidate.instance, candidate.provider, candidate.kwargs) is not None
    ]
    if not uncredentialed:
        return frozenset()
    try:
        offered = await hosted().get_hosted_models(organization_id=organization_to_serve(organization_id))
    except Exception:
        logger.warning("Could not list hosted models to classify uncredentialed candidates")
        return frozenset()
    managed: set[tuple[str, str]] = set()
    for candidate in uncredentialed:
        if candidate.instance not in offered:
            continue
        models = offered[candidate.instance]
        if models is None or candidate.model in models:
            managed.add((candidate.instance, candidate.model))
    return frozenset(managed)


async def refuse_managed_only(
    config: GatewayConfig,
    resolved: ResolvedProvider,
    *,
    hosted: Callable[[], ModelProviderPort],
    organization_id: uuid.UUID | None,
    surface: str,
) -> None:
    """Refuse a single candidate only the deployment's managed credentials would serve, on a ``surface`` they do not.

    Raises:
        ManagedCredentialUnsupportedError: naming the selector the caller wrote,
            the alias rather than its target.
    """
    if await managed_only_candidates(config, [resolved], hosted=hosted, organization_id=organization_id):
        raise ManagedCredentialUnsupportedError(
            resolved.instance,
            selector=resolved.alias or f"{resolved.instance}:{resolved.model}",
            surface=surface,
        )


def refuse_managed_candidate(
    managed_only: frozenset[tuple[str, str]], attempt: Attempt, *, surface: str = MANAGED_SURFACE_ROUTING_POLICY
) -> None:
    """Refuse a plan's candidate that :func:`managed_only_candidates` found, naming what the caller wrote.

    A policy's targets are not the caller's to see, so the candidate is named by
    its ``display_model`` (the policy or alias), never its ``instance:model``.

    Raises:
        ManagedCredentialUnsupportedError: ``attempt`` is in ``managed_only``.
    """
    if (attempt.instance, attempt.model) not in managed_only:
        return
    raise ManagedCredentialUnsupportedError(
        attempt.instance, selector=attempt.display_model or f"{attempt.instance}:{attempt.model}", surface=surface
    )


async def unpriced_model_refusal(
    config: GatewayConfig,
    selector: str,
    resolved: ResolvedProvider | None,
    *,
    plan: CompiledPlan | None,
    hosted: ModelProviderPort | None,
    organization_id: uuid.UUID | None,
) -> ModelRefusal:
    """Why a request that ``require_pricing`` refuses is turned away.

    A provider nothing can call is named ahead of the missing price, because
    pricing the model would not make it servable. A plan with several
    candidates keeps the pricing refusal: only a single candidate is offered
    to the hosted port at dispatch, so only its credential can be judged here.
    ``selector`` is what the caller wrote and is what a pricing refusal names.
    """
    single_candidate = plan is None or len(plan.attempts) == 1
    if resolved is not None and single_candidate:
        missing = await missing_provider_credential(config, resolved, hosted=hosted, organization_id=organization_id)
        if missing is not None:
            return ModelRefusal(code=missing.code, status_code=missing.status_code, detail=missing.detail)
    return ModelRefusal(code=PRICING_REQUIRED, status_code=402, detail=no_pricing_error_detail(selector))
