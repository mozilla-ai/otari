"""``CodeExecutionPort`` over E2B, a hosted sandbox provider.

The second implementation the port exists for. A deployment that cannot run
the reference container with the isolation untrusted code needs, a PaaS with
no privileged containers for instance, sets ``sandbox_provider: e2b`` and needs
no sandbox service of its own.

A session is one E2B sandbox and the session id is the sandbox's own, so this
adapter holds no state between calls: any worker can serve any session, and a
sandbox nothing releases is reclaimed by E2B's own lifetime timer.

The SDK is an optional extra (``otari[e2b]``) imported inside
:meth:`E2BCodeExecutionAdapter.open_session`, so a deployment that does not use
it neither installs it nor pays for the import, and the OSS smoke gate keeps
passing with no dev dependencies.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from gateway.log_config import logger
from gateway.ports.code_execution_port import (
    CodeExecutionSession,
    OutputOverBudget,
    SandboxFileEntry,
    SandboxNotReachableError,
    SandboxUnavailableError,
)
from gateway.types.code_execution import ResultBlock

# Where a session's files live. E2B's own default working directory, which is
# what code the model writes resolves a bare filename against.
WORKSPACE_ROOT = "/home/user"
# Extra wall-clock the SDK call gets over what the code is granted, so E2B's own
# teardown does not surface as an unreachable sandbox.
_REQUEST_SLACK_S = 30
# How deep under the workspace root a listing walks. A run that writes into a
# subdirectory still has its output collected, without walking a tree the code
# can nest without bound.
_LIST_DEPTH = 5
_TIMEOUT_EXIT_CODE = 124
_RETRY_AFTER_S = "5"


class E2BCodeExecutionAdapter:
    """Leases E2B sandboxes through the provider's async SDK.

    Takes nothing: which sandboxes it leases is E2B's own ``E2B_API_KEY``, read
    by the SDK, and everything a request narrows arrives through the port.
    """

    @property
    def label(self) -> str:
        return "e2b"

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
    ) -> AsyncIterator[CodeExecutionSession]:
        # ``auth_token`` is the caller's own bearer credential, which only means
        # something to a backend that authenticates the request. E2B
        # authenticates the deployment, through the SDK's own E2B_API_KEY.
        del auth_token
        # ``timeout_s`` is one execute's budget; what E2B's ``timeout`` sets is
        # how long the sandbox lives before it reclaims it, which has to cover
        # every execute a request makes and the file work around them.
        del timeout_s
        if image:
            # ``sandbox_session_image`` is a container image reference, the
            # published protocol's vocabulary. E2B names pre-built templates
            # instead, so honoring it here would fail every sandbox creation.
            logger.warning(
                "sandbox_session_image %r ignored: the e2b provider runs its own templates, not container images",
                image,
            )
        sdk = _sdk()
        try:
            sandbox = await sdk.AsyncSandbox.create(
                timeout=int(session_ttl_s),
                metadata={"otari": "code-execution"},
            )
        except sdk.AuthenticationException as exc:
            # The deployment's own credential, so the message is for its
            # operator and never reaches the caller.
            logger.error("E2B rejected this deployment's credential: %s", exc)
            raise SandboxNotReachableError("sandbox provider misconfigured") from exc
        except (sdk.RateLimitException, sdk.ServiceBusyException, sdk.TimeoutException) as exc:
            raise SandboxUnavailableError(_RETRY_AFTER_S) from exc
        except sdk.SandboxException as exc:
            raise SandboxNotReachableError(f"failed to create an E2B sandbox: {exc}") from exc

        try:
            yield _E2BSession(sandbox, sdk)
        finally:
            try:
                await sandbox.kill()
            except Exception:  # noqa: BLE001 - teardown is best effort; E2B reclaims on its own timer
                logger.warning("E2B sandbox %s cleanup failed", sandbox.sandbox_id, exc_info=True)


class _E2BSession:
    """One E2B sandbox, answering the port's six operations."""

    def __init__(self, sandbox: Any, sdk: Any) -> None:
        self._sandbox = sandbox
        self._sdk = sdk

    @property
    def session_id(self) -> str:
        return str(self._sandbox.sandbox_id)

    def _absolute(self, path: str) -> str:
        """Resolve a workspace-relative path, refusing anything that leaves it.

        Paths above the port are always relative, so this is defense in depth
        for the day one is not.
        """
        candidate = posixpath.normpath(posixpath.join(WORKSPACE_ROOT, path))
        if candidate != WORKSPACE_ROOT and not candidate.startswith(WORKSPACE_ROOT + "/"):
            msg = f"path escapes the session workspace: {path!r}"
            raise ValueError(msg)
        return candidate

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        try:
            execution = await self._sandbox.run_code(
                code, timeout=int(timeout_s), request_timeout=int(timeout_s) + _REQUEST_SLACK_S
            )
        except self._sdk.TimeoutException:
            return _result_block(
                stderr=f"Execution timed out after {int(timeout_s)}s\n", return_code=_TIMEOUT_EXIT_CODE
            )
        except self._sdk.SandboxException as exc:
            raise SandboxNotReachableError(f"sandbox exec failed: {exc}") from exc

        stdout = "".join(execution.logs.stdout)
        stderr = "".join(execution.logs.stderr)
        # A REPL echoes the value of a bare trailing expression; Jupyter carries
        # that as a result rather than on stdout, so surface it as a terminal would.
        for result in execution.results:
            if result.text:
                if stdout and not stdout.endswith("\n"):
                    stdout += "\n"
                stdout += result.text + "\n"
        return_code = 0
        if execution.error is not None:
            error = execution.error
            stderr += error.traceback or f"{error.name}: {error.value}"
            return_code = 1
        return _result_block(stdout=stdout, stderr=stderr, return_code=return_code)

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        del mime_type  # E2B stores bytes; the type travels with the file row, not the sandbox.
        try:
            await self._sandbox.files.write(self._absolute(path), data)
        except self._sdk.SandboxException as exc:
            raise SandboxNotReachableError(f"sandbox refused {path!r}: {exc}") from exc

    async def list_files(self) -> list[SandboxFileEntry]:
        try:
            entries = await self._sandbox.files.list(WORKSPACE_ROOT, depth=_LIST_DEPTH)
        except Exception as exc:  # noqa: BLE001 - an unlistable workspace reports nothing, as over HTTP
            logger.debug("E2B sandbox %s workspace not listable: %s", self.session_id, exc)
            return []
        listed: list[SandboxFileEntry] = []
        for entry in entries:
            if getattr(entry, "type", None) == self._sdk.FileType.DIR:
                continue
            listed.append(
                SandboxFileEntry(
                    path=posixpath.relpath(entry.path, WORKSPACE_ROOT),
                    size_bytes=int(getattr(entry, "size", 0) or 0),
                    modified_at=_modified_at(getattr(entry, "modified_time", None)),
                )
            )
        return listed

    async def _size_of(self, absolute: str) -> int | None:
        """What the provider says the file's size is, or ``None`` where it says nothing.

        One directory at depth 1, not the workspace tree: this runs once per
        produced file, and walking the whole tree again for each would spend a
        deep listing to learn one file's size.
        """
        try:
            entries = await self._sandbox.files.list(posixpath.dirname(absolute), depth=1)
        except Exception as exc:  # noqa: BLE001 - an unlistable workspace reports nothing, as over HTTP
            logger.debug("E2B sandbox %s could not size %s: %s", self.session_id, absolute, exc)
            return None
        for entry in entries:
            if entry.path == absolute and getattr(entry, "type", None) != self._sdk.FileType.DIR:
                return int(getattr(entry, "size", 0) or 0)
        return None

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        absolute = self._absolute(path)
        # The SDK hands the file over whole, so the size is established before
        # the read rather than while it arrives: a run writes whatever it likes,
        # and measuring afterwards would mean holding all of it first.
        size = await self._size_of(absolute)
        if size is not None and size > budget_bytes:
            raise OutputOverBudget
        try:
            data = bytes(await self._sandbox.files.read(absolute, format="bytes"))
        except (self._sdk.FileNotFoundException, self._sdk.NotFoundException) as exc:
            raise SandboxNotReachableError(f"sandbox has no file {path!r}") from exc
        if len(data) > budget_bytes:
            raise OutputOverBudget
        yield data


def _modified_at(value: Any) -> float | None:
    """A listing entry's modification time in POSIX seconds, or ``None``.

    The SDK documents a ``datetime``, and the field is only ever a hint that
    tells a file a run produced from one that was already there. So anything
    else is read as no hint rather than allowed to abort the whole listing,
    which would cost the caller every other file to learn about one.
    """
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _result_block(*, stdout: str = "", stderr: str = "", return_code: int = 0) -> ResultBlock:
    """The contract's result block, built from what the provider reported.

    ``content`` stays empty, as it is with the reference backend: which files a
    run produced is worked out above the port, by diffing the workspace.
    """
    return ResultBlock.model_validate(
        {
            "type": "code_execution_tool_result",
            "content": {
                "type": "code_execution_result",
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "content": [],
            },
        }
    )


def verify_ready() -> None:
    """Raise unless this process could actually lease an E2B sandbox.

    Called at startup (``adapters/code_execution_adapter.verify_code_execution_ready``)
    rather than on the request path: both of these are settled for the whole
    deployment, and finding them per request means answering a caller with a
    502 for something the operator could have been told at boot.
    """
    _sdk()
    if not os.getenv("E2B_API_KEY"):
        msg = "sandbox_provider 'e2b' requires E2B_API_KEY in the environment"
        raise SandboxNotReachableError(msg)


def _sdk() -> Any:
    """The E2B SDK surface this adapter uses, imported on first session.

    One namespace rather than a module-level import, so the optional extra is
    only required by a deployment that selected it, and a test can hand the
    session a stub.
    """
    try:
        import e2b
        from e2b_code_interpreter import AsyncSandbox
    except ImportError as exc:
        msg = "sandbox_provider 'e2b' requires the E2B SDK. Install it with: pip install otari[e2b]"
        raise SandboxNotReachableError(msg) from exc

    # A namespace object rather than a class body, which cannot read a local of
    # the function enclosing it once it binds the same name: `AsyncSandbox =
    # AsyncSandbox` there raises NameError at the first real session.
    return SimpleNamespace(
        AsyncSandbox=AsyncSandbox,
        AuthenticationException=e2b.AuthenticationException,
        FileNotFoundException=e2b.FileNotFoundException,
        FileType=e2b.FileType,
        NotFoundException=e2b.NotFoundException,
        RateLimitException=e2b.RateLimitException,
        SandboxException=e2b.SandboxException,
        ServiceBusyException=e2b.ServiceBusyException,
        TimeoutException=e2b.TimeoutException,
    )
