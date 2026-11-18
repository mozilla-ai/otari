"""Model metadata overrides: what an operator or an organization says about a model.

A self-hosted server can list a model with nothing but its name, and models.dev
may not know it, so the catalog would show no limits and no capabilities. An
override sets them, and wins over both.

Two routers, by owner. The deployment's overrides describe its own providers
and are operator-only. An organization's describe the models its own keys
serve, under the prefixes those keys answer to, and only its owners and admins
set them; they are read only by that organization's callers.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from gateway.api.deps import (
    CurrentIdentity,
    ModelMetadataOverrideServiceDep,
    OrgProviderModelServiceDep,
    require_deployment_operator,
    verify_master_key,
)
from gateway.schemas.catalog import ModelMetadataOverridePublic, ModelMetadataOverrideRequest
from gateway.services.catalog import MetadataOverride

router = APIRouter(
    prefix="/model-overrides",
    tags=["model-overrides"],
    dependencies=[Depends(require_deployment_operator)],
)

org_router = APIRouter(
    prefix="/organizations/me/model-overrides",
    tags=["model-overrides"],
    dependencies=[Depends(verify_master_key)],
)


def _selector(selector: str) -> str:
    provider, separator, model = selector.partition(":")
    if not separator or not provider or not model:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Name the model as `provider:model`."
        )
    return selector


def _override(body: ModelMetadataOverrideRequest) -> MetadataOverride:
    fields = body.model_dump()
    if body.input_modalities is not None:
        fields["input_modalities"] = tuple(body.input_modalities)
    return MetadataOverride(**fields)


def _public(selector: str, override: MetadataOverride) -> ModelMetadataOverridePublic:
    return ModelMetadataOverridePublic.model_validate({"selector": selector, **asdict(override)})


@router.put("/{selector:path}")
async def set_model_override(
    selector: str, body: ModelMetadataOverrideRequest, service: ModelMetadataOverrideServiceDep
) -> ModelMetadataOverridePublic:
    """Replace the deployment's override of a model; every field null clears it."""
    selector = _selector(selector)
    return _public(selector, await service.set(selector, _override(body)))


@router.delete("/{selector:path}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_model_override(selector: str, service: ModelMetadataOverrideServiceDep) -> None:
    """Drop the deployment's override of a model."""
    await service.clear(_selector(selector))


@org_router.put("/{selector:path}", responses={status.HTTP_404_NOT_FOUND: {"description": "No key serves the prefix."}})
async def set_organization_model_override(
    selector: str,
    body: ModelMetadataOverrideRequest,
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
) -> ModelMetadataOverridePublic:
    """Replace the organization's override of a model one of its keys serves; every field null clears it.

    Organization owners and admins only.
    """
    selector = _selector(selector)
    override = await service.set_metadata_override(user=current_identity, selector=selector, override=_override(body))
    return _public(selector, override)


@org_router.delete(
    "/{selector:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No key serves the prefix."}},
)
async def clear_organization_model_override(
    selector: str, service: OrgProviderModelServiceDep, current_identity: CurrentIdentity
) -> None:
    """Drop the organization's override of a model. Organization owners and admins only."""
    await service.clear_metadata_override(user=current_identity, selector=_selector(selector))
