"""``CodeExecutionPort`` over Monty, a Python interpreter run in this gateway's own worker processes.

The sandbox for a deployment with the ``monty`` extra and no ``sandbox_url``, so
code execution needs no container and no account. Monty has no filesystem,
environment or network of its own and enforces time and memory limits in the
interpreter. It runs a subset of Python, which :attr:`runtime` tells the model.

A session is one worker checked out of a pool the process shares, with a
temporary directory mounted as its working directory: uploads are written into
it and outputs read back from it, so the file bridge above the port is
unchanged. Nothing outlives the lease.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gateway.log_config import logger
from gateway.ports.code_execution_port import (
    CodeExecutionPort,
    CodeExecutionSession,
    OutputOverBudget,
    SandboxFileEntry,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.types.code_execution import ResultBlock

if TYPE_CHECKING:
    import pydantic_monty

WORKSPACE = "/workspace"
_CHECKOUT_TIMEOUT_S = 10.0
_RETRY_AFTER_S = "5"
_TIMEOUT_EXIT_CODE = 124
_READ_CHUNK_BYTES = 64 * 1024
_RUNTIME = (
    "A subset of Python run by Monty: the standard library only (json, math, re, datetime, "
    "collections, itertools, functools, random, decimal, csv, hashlib, base64, pathlib), no "
    "third-party packages, no class inheritance and no generators. There is no open(): read "
    "and write files with pathlib.Path. Code that uses something unsupported is rejected "
    "before it runs, with the reason in stderr."
)

_pool: pydantic_monty.AsyncMonty | None = None


class MontyCodeExecutionAdapter(CodeExecutionPort):
    """Leases Monty workers from one pool per process, capped at the CPU count."""

    def __init__(self, *, max_memory_bytes: int, write_bytes_limit: int) -> None:
        self._max_memory_bytes = max_memory_bytes
        self._write_bytes_limit = write_bytes_limit

    @property
    def label(self) -> str:
        return "monty"

    @property
    def runtime(self) -> str:
        return _RUNTIME

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[CodeExecutionSession]:
        del session_ttl_s, keep_alive_s  # a session ends with the block
        if resume is not None:
            raise SandboxSessionGoneError(f"monty session {resume} is gone: monty never holds one across requests")
        if image:
            logger.warning("sandbox_session_image %r ignored: the monty provider runs no container images", image)
        monty = _sdk()
        pool = await _shared_pool(monty)
        limits: pydantic_monty.ResourceLimits = {
            "max_feed_duration_secs": timeout_s,
            "max_memory": self._max_memory_bytes,
        }
        async with AsyncExitStack() as stack:
            workspace = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="otari-monty-"))
            stack.push_async_callback(_remove_workspace, workspace)
            mount = monty.MountDir(
                host_path=workspace,
                virtual_path=WORKSPACE,
                mode="read-write",
                write_bytes_limit=self._write_bytes_limit,
            )
            stack.callback(mount.close)
            try:
                worker = await stack.enter_async_context(
                    pool.checkout(limits=limits, type_check=True, type_check_format="concise")
                )
            except TimeoutError as exc:
                raise SandboxUnavailableError(_RETRY_AFTER_S) from exc
            yield _MontySession(worker, monty, workspace, mount)


class _MontySession(CodeExecutionSession):
    """One checked-out worker and its working directory."""

    def __init__(self, worker: pydantic_monty.AsyncMontySession, monty: Any, workspace: Path, mount: Any) -> None:
        self._worker = worker
        self._monty = monty
        self._workspace = workspace
        self._mount = mount

    @property
    def session_id(self) -> str:
        return self._workspace.name

    @property
    def holds_across_requests(self) -> bool:
        return False

    def discard(self) -> None:
        return None

    def _host_path(self, path: str) -> Path:
        candidate = (self._workspace / path).resolve()
        if not candidate.is_relative_to(self._workspace.resolve()):
            msg = f"path escapes the session workspace: {path!r}"
            raise ValueError(msg)
        return candidate

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        del timeout_s  # the checkout's feed limit
        monty = self._monty
        output = monty.CollectStreams()
        error, return_code = "", 0
        try:
            await self._worker.feed_run(code, print_callback=output, mount=self._mount, cwd=WORKSPACE)
        except monty.MontyCrashedError as exc:
            raise SandboxNotReachableError(f"monty worker died: {exc}") from exc
        except monty.MontyTypingError as exc:
            error, return_code = exc.display(), 1
        except (monty.MontyRuntimeError, monty.MontySyntaxError) as exc:
            error = exc.display()
            return_code = _TIMEOUT_EXIT_CODE if error.rstrip().rsplit("\n", 1)[-1].startswith("TimeoutError") else 1
        stdout = "".join(text for stream, text in output.output if stream == "stdout")
        stderr = "".join(text for stream, text in output.output if stream == "stderr")
        if error:
            stderr += error if error.endswith("\n") else error + "\n"
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

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        del mime_type
        target = self._host_path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def list_files(self) -> list[SandboxFileEntry]:
        return await asyncio.to_thread(_list_workspace, self._workspace)

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        target = self._host_path(path)
        size = await asyncio.to_thread(lambda: target.stat().st_size if target.is_file() else None)
        if size is None:
            raise SandboxNotReachableError(f"sandbox has no file {path!r}")
        if size > budget_bytes:
            raise OutputOverBudget
        handle = await asyncio.to_thread(target.open, "rb")
        try:
            while chunk := await asyncio.to_thread(handle.read, _READ_CHUNK_BYTES):
                yield chunk
        finally:
            await asyncio.shield(asyncio.to_thread(handle.close))


def _list_workspace(workspace: Path) -> list[SandboxFileEntry]:
    listed: list[SandboxFileEntry] = []
    for root, _dirs, files in os.walk(workspace):
        for name in files:
            stat = (Path(root) / name).lstat()
            listed.append(
                SandboxFileEntry(
                    path=os.path.relpath(Path(root) / name, workspace),
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                )
            )
    return listed


async def _remove_workspace(workspace: Path) -> None:
    # Shielded, so a cancelled request still deletes what its code wrote.
    await asyncio.shield(asyncio.to_thread(shutil.rmtree, workspace, ignore_errors=True))


async def _shared_pool(monty: Any) -> pydantic_monty.AsyncMonty:
    """This process's worker pool, entered once. Its workers exit with the process.

    Entering an active pool again widens its worker cap, so it is never
    re-entered: of two first leases that race, the one that finishes entering
    keeps its pool and the other closes its own.
    """
    global _pool
    if _pool is None:
        pool = monty.AsyncMonty(
            min_processes=1, max_processes=os.cpu_count() or 1, checkout_timeout=_CHECKOUT_TIMEOUT_S
        )
        await pool.__aenter__()
        if _pool is None:
            _pool = pool
        else:
            await pool.__aexit__(None, None, None)
    return _pool


def verify_ready() -> None:
    """Raise unless this process can import Monty. Called at startup, as E2B's is."""
    _sdk()


def _sdk() -> Any:
    try:
        import pydantic_monty
    except ImportError as exc:
        msg = "sandbox_provider 'monty' requires Monty. Install the monty extra: uv sync --extra monty"
        raise SandboxNotReachableError(msg) from exc
    return pydantic_monty
