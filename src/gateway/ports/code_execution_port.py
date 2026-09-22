"""Running model-generated code in a sandbox.

The seam between the gateway's code-execution tool and whatever actually runs
the code. Two implementations ship in the core: the published
`code-execution protocol <../../docs/code-execution-protocol.md>`_ spoken over
HTTP to a backend the operator runs, and a hosted provider driven through its
own SDK in this process. That second one is what the port exists for
(``ARCHITECTURE.md``, rule 7): a deployment that cannot run the reference
container with the isolation untrusted code needs should not have to stand up
a translating service to use a hosted sandbox instead.

What crosses the seam is one session and the six operations Otari performs on
it. Everything above the port, which is to say the workspace policy, the tool
allow-list, the usage tally, the purpose hint, staging the request's uploads
and collecting what a run produced, belongs to
:class:`gateway.services.sandbox_backend.SandboxBackend` and is the same
whichever adapter is bound. An adapter therefore owns transport and nothing
else, which is what keeps the two from drifting in behavior.

Paths are relative to the session's own workspace, never absolute, so an
adapter is free to put that workspace wherever its provider does and a caller
cannot address anything outside it.

Stability: this interface is not frozen while Otari is pre-1.0.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

from gateway.types.code_execution import ResultBlock

__all__ = [
    "CodeExecutionPort",
    "CodeExecutionSession",
    "OutputOverBudget",
    "SandboxFileEntry",
    "SandboxNotReachableError",
    "SandboxUnavailableError",
]


class SandboxNotReachableError(RuntimeError):
    """The sandbox could not be reached, or answered something unusable.

    Adapters raise this for every terminal transport failure, including a
    backend whose response does not match the contract: to the caller the two
    are the same, an execution that produced no result.
    """


class SandboxUnavailableError(SandboxNotReachableError):
    """Temporary sandbox capacity or dependency failure."""

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("sandbox temporarily unavailable")
        # Forward only validated delay-seconds.
        self.retry_after = (
            retry_after
            if retry_after and retry_after.isascii() and retry_after.isdigit() and len(retry_after) <= 6
            else None
        )


class OutputOverBudget(Exception):
    """A produced file ran past the bytes this call may still store.

    Raised by an adapter that learns the size before the bytes arrive, and by
    the backend's own counter while they do, so one call cannot spend more of
    the request's output budget than it was given.
    """


@dataclass(frozen=True)
class SandboxFileEntry:
    """One file in a session's workspace, as a listing reports it.

    ``size_bytes`` and ``modified_at`` are what tell a file a run just produced
    from one that was already there, so an adapter that can supply them should.
    ``modified_at`` is ``None`` where the provider does not say, which leaves
    size as the only signal. Directories are not entries: nothing above the
    port can fetch one, so an adapter leaves them out.
    """

    path: str
    size_bytes: int
    modified_at: float | None = None


class CodeExecutionSession(Protocol):
    """One live sandbox, for the length of one request."""

    @property
    def session_id(self) -> str:
        """The backend's own id for this session, for logging and tracing."""
        ...

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        """Run ``code`` and return the contract's result block.

        A program that fails is a successful execution reported through
        ``return_code`` and ``stderr``. Only a sandbox that could not run it at
        all raises :class:`SandboxNotReachableError`.
        """
        ...

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        """Write one of the request's uploads into the workspace before the model runs."""
        ...

    async def list_files(self) -> list[SandboxFileEntry]:
        """The workspace's files. Empty where the backend cannot list them."""
        ...

    def read_file(self, path: str, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        """Stream one produced file back, never holding more than ``budget_bytes``.

        ``budget_bytes`` is what this call may still store. What a run writes is
        untrusted and can be arbitrarily large, so an adapter must establish the
        size before it holds the bytes: one that learns it up front raises
        :class:`OutputOverBudget` before reading any, and one whose provider
        hands the file over whole asks for the size first rather than reading
        and then measuring. The caller counts the rest as they arrive.
        """
        ...


class CodeExecutionPort(Protocol):
    """Leases sandboxes. One per deployment, chosen by ``sandbox_provider``."""

    @property
    def label(self) -> str:
        """Where code runs, for a span or a log line. Never a credential."""
        ...

    def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
    ) -> AbstractAsyncContextManager[CodeExecutionSession]:
        """Lease a sandbox for the duration of the block, releasing it on exit.

        ``timeout_s`` is the budget for one :meth:`CodeExecutionSession.execute`;
        ``session_ttl_s`` is how long the whole lease may need to live, which is
        every execute the tool loop may make plus the seeding and collecting
        around them. The two differ because one request runs code more than
        once, so a provider that reclaims on a timer of its own must be told the
        second number and never the first. An adapter whose sessions end only
        when this block exits ignores it.

        ``image`` is the workspace's pinned image or the deployment's, in the
        vocabulary of the published protocol (a container image reference). An
        adapter whose provider has no such notion, or names its workspaces in a
        vocabulary of its own, ignores it. ``auth_token`` is the caller's bearer
        credential, set only where the backend authenticates the request rather
        than the deployment.
        """
        ...
