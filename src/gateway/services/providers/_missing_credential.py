"""What a request for an unpriced model is told, and whether its provider can be called at all."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from gateway.core.config import provider_credential_env_names
from gateway.core.error_codes import PRICING_REQUIRED
from gateway.services.model_access import ModelRefusal
from gateway.services.pricing_service import no_pricing_error_detail
from gateway.services.provider_kwargs import MissingCredential, credential_ladder_exhausted

if TYPE_CHECKING:
    from gateway.ports.model_provider_port import ModelProviderPort
    from gateway.services.provider_kwargs import ResolvedProvider
    from gateway.services.routing import CompiledPlan


async def missing_provider_credential(
    resolved: ResolvedProvider,
    *,
    hosted: ModelProviderPort | None,
    organization_id: uuid.UUID | None,
) -> MissingCredential | None:
    """The missing credential a dispatch of ``resolved`` would fail on, or ``None``.

    ``None`` whenever something may still serve the candidate: a rung of the
    credential ladder, the provider's own environment variable, a provider that
    needs no key, the caller's own endpoint, or, where ``hosted`` is asked at
    dispatch, a hosted provider it lists for the organization. ``hosted`` is
    ``None`` for a path whose dispatch never asks it.
    """
    if resolved.owned_endpoint is not None or not credential_ladder_exhausted(resolved.provider, resolved.kwargs):
        return None
    if hosted is not None and organization_id is not None:
        if resolved.instance in await hosted.get_hosted_models(organization_id=organization_id):
            return None
    env_names = provider_credential_env_names(resolved.provider.value)
    return MissingCredential(provider=resolved.instance, env_var="/".join(env_names) if env_names else None)


async def unpriced_model_refusal(
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
        missing = await missing_provider_credential(resolved, hosted=hosted, organization_id=organization_id)
        if missing is not None:
            return ModelRefusal(code=missing.code, status_code=missing.status_code, detail=missing.detail)
    return ModelRefusal(code=PRICING_REQUIRED, status_code=402, detail=no_pricing_error_detail(selector))
