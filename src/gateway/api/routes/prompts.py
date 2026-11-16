"""Stored prompt management (``/api/v1/prompts``).

A stored prompt is a named piece of text that a caller can reference by name
instead of embedding it inline. Prompts are operator-scoped and available
deployment-wide, so an admin can update a prompt without redeploying.
"""

import uuid

from fastapi import APIRouter, Depends, Response, status

from gateway.api.deps import PromptServiceDep, require_deployment_operator
from gateway.core.surface import Surface
from gateway.schemas.prompts import PromptCreate, PromptResponse, PromptUpdate

router = APIRouter(
    prefix="/prompts",
    tags=["prompts"],
    dependencies=[Depends(require_deployment_operator)],
)

SURFACE = Surface("prompts")


@router.get("")
async def list_prompts(service: PromptServiceDep) -> list[PromptResponse]:
    """List all stored prompts, ordered by name."""
    return await service.list_prompts()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_prompt(service: PromptServiceDep, body: PromptCreate) -> PromptResponse:
    """Create a new stored prompt.

    The name must be unique across all stored prompts.
    """
    return await service.create_prompt(body)


@router.get("/{prompt_id}")
async def get_prompt(service: PromptServiceDep, prompt_id: uuid.UUID) -> PromptResponse:
    """Get a single stored prompt by ID."""
    return await service.get_prompt(prompt_id)


@router.patch("/{prompt_id}")
async def update_prompt(service: PromptServiceDep, prompt_id: uuid.UUID, body: PromptUpdate) -> PromptResponse:
    """Update a stored prompt's name, content, or both."""
    return await service.update_prompt(prompt_id, body)


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(service: PromptServiceDep, prompt_id: uuid.UUID) -> Response:
    """Delete a stored prompt."""
    await service.delete_prompt(prompt_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
