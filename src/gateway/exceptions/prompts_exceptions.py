"""Errors that the prompts domain may raise.

Each carries the HTTP status it renders as.
"""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError


class PromptNotFoundError(TenancyNotFoundError):
    def __init__(self, prompt_id: str) -> None:
        super().__init__(f"Prompt '{prompt_id}' not found")


class PromptNameTakenError(TenancyConflictError):
    def __init__(self, name: str) -> None:
        super().__init__(f"A prompt named '{name}' already exists")
