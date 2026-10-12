"""Whether a resolved candidate is known before dispatch to have no credential."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from gateway.core.config import provider_credential_env_names
from gateway.services.provider_kwargs import MissingCredential, credential_ladder_exhausted

if TYPE_CHECKING:
    from gateway.ports.model_provider_port import ModelProviderPort
    from gateway.services.provider_kwargs import ResolvedProvider


async def missing_provider_credential(
    resolved: ResolvedProvider,
    *,
    hosted: ModelProviderPort | None,
    organization_id: uuid.UUID | None,
) -> MissingCredential | None:
    """The missing credential a dispatch of ``resolved`` would fail on, or ``None``.

    For a refusal that would otherwise hide that cause: a request refused for
    an unpriced model whose provider nothing can call is better told about the
    provider, because pricing the model would not make it servable.

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
