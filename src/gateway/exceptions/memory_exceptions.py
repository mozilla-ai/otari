"""Memory's refusals, each with its HTTP status. No message carries what an agent wrote."""

from fastapi import status

from gateway.exceptions import TenancyError, TenancyForbiddenError, TenancyNotFoundError, TenancyValidationError

SEALING_UNAVAILABLE = "Memory can't store anything right now. Try again later."
MODEL_UNAVAILABLE = "Memory can't search or store right now. Try again later."
MEMORY_OFF = (
    "Memory is off for this workspace. Its organization's owners or admins, and then the workspace's, can turn it on."
)
WORKSPACE_SCOPE_OFF = "Shared workspace memory is off for this workspace. Use the member scope instead."
ENTRY_NOT_FOUND = "No memory entry with that id."
READ_ONLY = "Workspace viewers can search memory but not change it."
NOT_YOURS = "Only the entry's author, its owner or a workspace admin can forget it."
TOO_LONG = "That is longer than memory keeps. Shorten it to {limit} characters."
REFUSED = "Memory does not keep {what}. Remove it and propose again."

_WHAT = {
    "credential": "credentials or secrets",
    "private_key": "private keys",
    "email_address": "email addresses",
    "card_number": "card numbers",
}


class MemorySealingUnavailableError(TenancyError):
    """No data key could be made or opened, so nothing is stored: memory never keeps text unsealed."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(SEALING_UNAVAILABLE)


class MemoryModelUnavailableError(TenancyError):
    """The embedding model is not configured or did not answer, so nothing is searched or stored."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(MODEL_UNAVAILABLE)


class MemoryOffError(TenancyForbiddenError):
    """The workspace has not turned memory on."""

    def __init__(self) -> None:
        super().__init__(MEMORY_OFF)


class WorkspaceScopeOffError(TenancyForbiddenError):
    """The workspace keeps member memory but not the shared one."""

    def __init__(self) -> None:
        super().__init__(WORKSPACE_SCOPE_OFF)


class MemoryEntryNotFoundError(TenancyNotFoundError):
    """No entry with that id that this caller may see: another member's or workspace's reads the same."""

    def __init__(self) -> None:
        super().__init__(ENTRY_NOT_FOUND)


class MemoryReadOnlyError(TenancyForbiddenError):
    """A workspace viewer may recall but not propose, confirm, flag or forget."""

    def __init__(self) -> None:
        super().__init__(READ_ONLY)


class NotYoursToForgetError(TenancyForbiddenError):
    """The caller may recall the entry but not remove it."""

    def __init__(self) -> None:
        super().__init__(NOT_YOURS)


class MemoryTooLongError(TenancyValidationError):
    """A proposal longer than ``memory_entry_max_length``."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, limit: int) -> None:
        super().__init__(TOO_LONG.format(limit=limit))


class MemoryRefusedError(TenancyValidationError):
    """A proposal carrying something memory does not keep. The message names its kind, never the match."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, finding: str) -> None:
        super().__init__(REFUSED.format(what=_WHAT.get(finding, "that")))
        self.finding = finding
