"""The stored prompts an operator manages from the dashboard."""

import uuid
from typing import Any

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.prompts_exceptions import PromptNameTakenError, PromptNotFoundError
from gateway.repositories.prompts import PromptNameTaken, StoredPromptRepository
from gateway.schemas.prompts import PromptCreate, PromptResponse, PromptUpdate


def _public(row: Any) -> PromptResponse:
    return PromptResponse(
        id=row.id,
        name=row.name,
        content=row.content,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PromptService:
    """Lists, creates, updates and deletes stored prompts."""

    def __init__(self, uow: UnitOfWork, prompts: StoredPromptRepository) -> None:
        self._uow = uow
        self._prompts = prompts

    async def list_prompts(self) -> list[PromptResponse]:
        """All stored prompts, ordered by name."""
        async with self._uow:
            rows = await self._prompts.list_all()
            return [_public(row) for row in rows]

    async def create_prompt(self, request: PromptCreate) -> PromptResponse:
        """Store a new prompt.

        Raises:
            PromptNameTakenError: a prompt with this name already exists.
        """
        async with self._uow:
            try:
                row = await self._prompts.create_new(request.name, request.content)
            except PromptNameTaken as exc:
                raise PromptNameTakenError(exc.name) from exc
            return _public(row)

    async def get_prompt(self, prompt_id: uuid.UUID) -> PromptResponse:
        """Return one prompt by id.

        Raises:
            PromptNotFoundError: no prompt has this id.
        """
        async with self._uow:
            row = await self._prompts.get_by_id(prompt_id)
        if row is None:
            raise PromptNotFoundError(str(prompt_id))
        return _public(row)

    async def update_prompt(self, prompt_id: uuid.UUID, request: PromptUpdate) -> PromptResponse:
        """Update a stored prompt's name, content, or both.

        Raises:
            PromptNotFoundError: no prompt has this id.
            PromptNameTakenError: the new name is already taken.
        """
        async with self._uow:
            row = await self._prompts.get_by_id(prompt_id)
            if row is None:
                raise PromptNotFoundError(str(prompt_id))
            changes: dict[str, object] = {}
            if request.name is not None:
                changes["name"] = request.name
            if request.content is not None:
                changes["content"] = request.content
            if changes:
                try:
                    row = await self._prompts.update(row, changes)
                except PromptNameTaken as exc:
                    raise PromptNameTakenError(exc.name) from exc
            return _public(row)

    async def delete_prompt(self, prompt_id: uuid.UUID) -> None:
        """Delete a stored prompt.

        Raises:
            PromptNotFoundError: no prompt has this id.
        """
        async with self._uow:
            row = await self._prompts.get_by_id(prompt_id)
            if row is None:
                raise PromptNotFoundError(str(prompt_id))
            await self._prompts.delete(row)
