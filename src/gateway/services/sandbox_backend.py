"""Dispatch `code_execution` tool calls to a sandbox container.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches
to whenever the model emits a ``code_execution(code=…)`` call. The sandbox
container lives in its own repo
(https://github.com/mozilla-ai/otari-sandbox-container) and is pulled from
Docker Hub (``mzdotai/otari-sandbox-container``). It runs a Python REPL
with a curated set of data-science libraries pre-installed.

The contract this drives is specified in ``docs/code-execution-protocol.md``;
the shapes it returns are typed in :mod:`gateway.types.code_execution`. The
three operations used here:

* ``POST /sessions``         → creates a session, returns a handle carrying
                              ``session_id``. Carries ``{image: "…"}`` when a
                              workspace policy or the deployment names one, and
                              an empty body otherwise
* ``POST /sessions/{id}/exec``  with ``{tool: "code_execution",
                                        input: {code: "…"},
                                        timeout_seconds: int}``
                              → returns ``{result_block: {…}}``
* ``DELETE /sessions/{id}``  → tears the session down
* ``POST /sessions/{id}/files``, ``GET /sessions/{id}/files/list`` and
  ``GET /sessions/{id}/files?path=…``
                              → seed the request's uploads into the workspace
                              before the first call, then after each call list
                              the workspace and fetch what appeared or changed,
                              when a :class:`SandboxFiles` bridge is attached

Session lifecycle is per-request: enter creates a session, exit
destroys it. State does not persist across separate chat-completion
requests in this minimum-viable backend. A future stateful variant
(per-conversation session affinity, warm pool, etc.) is the platform's
problem — see ``docs/sandbox-oss-platform-direction.md`` in the
private platform repo for that picture.

This backend satisfies the same duck-typed protocol the MCP loop uses
for tool dispatch (``openai_tools``, ``owns_tool``, ``purpose_hints``,
``call_tool``), so the loop accepts it as a ``pool`` without any
refactor to :func:`gateway.services.mcp_loop.mcp_tool_loop`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from opentelemetry import trace
from pydantic import ValidationError

from gateway.services.tool_usage import ToolUsageTally
from gateway.types.code_execution import ExecResponse, ResultBlock, SessionHandle

if TYPE_CHECKING:
    from types import TracebackType

    from gateway.services.file_service import StagedFile

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

CODE_EXECUTION_TOOL_NAME = "code_execution"
# The gateway's own container ids. OpenAI issues ``cntr_``-prefixed ids and
# Anthropic ``container_``-prefixed ones, so a reserved prefix is what lets an
# echoed native item be told apart from one describing a provider's container.
CONTAINER_ID_PREFIX = "otari_cntr_"
# The code-execution tool kinds a policy may name, which is the vocabulary the
# hosted ``CodeExecutionConfig.tools`` uses and the one the protocol's ``tool``
# field carries on the wire. This backend serves the first of them and no more,
# so a workspace allow-list intersects down to ``code_execution`` or to nothing;
# the other two are here so a policy written today keeps meaning the same thing
# on the day a backend serves them, rather than being refused as unknown now and
# silently ungated later.
CODE_EXECUTION_TOOL_NAMES: tuple[str, ...] = (
    CODE_EXECUTION_TOOL_NAME,
    "bash_code_execution",
    "text_editor_code_execution",
)
# The execution budget one call gets when nothing narrows it. Public because a
# workspace code-execution policy floors its own ceiling against this value
# rather than carrying a second idea of the default (see
# ``services/tenancy/workspace_code_execution_policy_service.py``).
DEFAULT_EXEC_TIMEOUT_S = 60.0
# Headroom added on top of the execution budget for the exec POST's own read
# timeout. The sandbox is granted ``timeout_seconds`` to run the code; the HTTP
# client must wait longer than that (network + serialization + the sandbox's own
# teardown) so a legitimate near-max execution returns its result instead of
# tripping the client read timeout as a spurious ``SandboxNotReachableError``.
_EXEC_TIMEOUT_BUFFER_S = 10.0
_DEFAULT_PURPOSE_HINT = (
    "Prefer `code_execution` for any computation, data analysis, date "
    "arithmetic, statistics, or anything that benefits from exact output. "
    "Python with numpy/pandas/scipy/sympy/matplotlib pre-installed. Files the "
    "user attached are in the working directory under their own names. A file "
    "you write there comes back with a file_id; give the user that file_id so "
    "they can download it."
)


class SandboxFiles(Protocol):
    """What the backend needs to move files in and out of a session.

    Implemented by :class:`gateway.services.files.SandboxFileBridge`;
    a Protocol so the backend does not depend on the database-backed store and
    a test can hand it a stub.
    """

    @property
    def inputs(self) -> list[StagedFile]:
        """The uploads to seed into the session, in message order."""
        ...

    @property
    def max_output_files(self) -> int:
        """Most produced files one call may store; the rest are named but not stored."""
        ...

    @property
    def max_output_bytes(self) -> int:
        """Total bytes one call may store across its produced files."""
        ...

    async def read_input(self, staged: StagedFile) -> bytes: ...

    async def store_output(self, filename: str, chunks: AsyncIterator[bytes]) -> str | None:
        """Persist a produced file from ``chunks``, returning the ``file_id`` a caller downloads it by.

        ``None`` for an empty file, which is not worth a row. An exception the
        chunk source raises propagates after the partial blob is removed.
        """
        ...


def code_execution_tool_definition() -> dict[str, Any]:
    """The OpenAI-shaped function definition the model is given for code execution.

    Module-level, and returning a fresh dict per call, so the ``/v1/tools``
    discovery endpoint can advertise the same schema the tool loop injects without
    constructing a backend. Mirrors
    :func:`gateway.services.web_search_backend.web_search_tool_definition`.
    """
    return {
        "type": "function",
        "function": {
            "name": CODE_EXECUTION_TOOL_NAME,
            "description": (
                "Execute Python code in a sandboxed REPL. Returns stdout, "
                "stderr, and any rich result blocks. State persists across "
                "calls within the same request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }


@dataclass(frozen=True)
class CodeExecution:
    """One executed call, kept for a loop that answers in a provider's native vocabulary.

    ``result`` is the backend's structured result block, so the loop can mint an
    Anthropic ``code_execution_tool_result`` or an OpenAI ``code_interpreter_call``
    from the real stdout, stderr and exit code rather than re-parsing the string
    the model was given. ``None`` when the call never produced one (the backend
    was unreachable), which the loop renders as its vocabulary's error shape.
    """

    code: str
    result: ResultBlock | None
    # Produced filename -> the ``file_id`` it was stored under in ``/v1/files``,
    # for the files a bridge collected. A produced file with no entry here is
    # one the caller cannot download, so a native block does not announce it.
    file_ids: dict[str, str] = field(default_factory=dict)


class SandboxNotReachableError(RuntimeError):
    """Raised when the sandbox container can't be reached or returns malformed data."""


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


def _contract_violation(exc: ValidationError) -> str:
    """Summarise a schema violation without quoting the payload.

    Pydantic's ``ValidationError`` subclasses ``ValueError``, so every handler
    below must catch it *before* the clause that catches ``ValueError``.
    Reordering them silently routes schema violations through the generic
    handler and reintroduces the leak this exists to prevent.

    Pydantic renders the offending values into ``str(exc)`` (``input_value=...``),
    and a result block carries arbitrary program output from model-generated
    code, so rendering it would put that output into logs and spans. The field
    locations and error types are the diagnostically useful part and carry none
    of it.
    """
    fields = ", ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['type']}" for error in exc.errors()
    )
    return f"response does not match the code-execution contract ({fields})"


class _OutputOverBudget(Exception):
    """A produced file ran past the bytes this call may still store."""


class _CountedChunks:
    """An async iterator over ``source`` that counts bytes and stops past ``budget``."""

    def __init__(self, source: AsyncIterator[bytes], budget: int) -> None:
        self._source = source
        self._budget = budget
        self.total = 0

    def __aiter__(self) -> _CountedChunks:
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._source.__anext__()
        self.total += len(chunk)
        if self.total > self._budget:
            raise _OutputOverBudget
        return chunk


class SandboxBackend:
    """Async context manager that owns one sandbox session for a request's lifetime.

    Usage::

        async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
            # backend duck-types as the MCP loop's `pool` parameter
            result = await mcp_tool_loop(
                completion_kwargs=kwargs, pool=backend, max_iterations=N,
            )
    """

    def __init__(
        self,
        *,
        sandbox_url: str,
        purpose_hint: str | None = None,
        timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
        auth_token: str | None = None,
        image: str | None = None,
        allowed_tools: frozenset[str] | None = None,
        tally: ToolUsageTally | None = None,
        files: SandboxFiles | None = None,
        files_base_url: str | None = None,
    ) -> None:
        self._sandbox_url = sandbox_url.rstrip("/")
        # Where this deployment serves ``/v1/files`` from: a loop answering in
        # OpenAI's vocabulary needs a URL to announce a produced image with.
        # None outside a request (tests, direct use), which announces none.
        self.files_base_url = files_base_url
        # The request's file bridge, or None when it has no uploads to seed and
        # nowhere to keep what a run produces (hybrid mode, tests, direct use).
        self._files = files
        # Per-request accounting, owned by the route and passed in. None when the
        # backend runs outside a billed request (tests, direct use).
        self._tally = tally
        self._purpose_hint = purpose_hint or _DEFAULT_PURPOSE_HINT
        self._timeout_s = timeout_s
        # Optional bearer credential forwarded as `Authorization: Bearer` on every
        # call to the sandbox backend. Set in hybrid mode so the platform-hosted
        # /v1/sandbox proxy (which authenticates the caller's workspace token) admits
        # the request and derives tenancy from it. Unset (and unsent) when the
        # backend is a standalone exec-service that needs no auth.
        self._auth_token = auth_token
        # The sandbox image this session asks for: the workspace's pinned image,
        # else the deployment's, else nothing. Sent as an additive field on
        # ``CreateSession`` (see ``docs/code-execution-protocol.md``), so a
        # backend that leases from a fixed pre-baked pool ignores it and the
        # request is byte-for-byte what it was before this existed.
        self._image = image
        # The tool kinds this backend may expose, or None for "no narrowing".
        # An allow-list can only take one away: the intersection with what the
        # backend actually serves is what ``openai_tools`` advertises and what
        # ``owns_tool`` claims, so a name outside it is never offered to the
        # model and never dispatched if the model invents it anyway.
        self._allowed_tools = allowed_tools
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._stack: AsyncExitStack = AsyncExitStack()
        # The calls executed since the last ``take_executions``, in order. A loop
        # that mints native result blocks drains this right after the awaited
        # calls it made, which is what keeps a batch's blocks paired with the
        # right calls: every tool loop runs its calls one at a time and in order.
        self._executions: list[CodeExecution] = []
        # The workspace as last listed, path -> (size, modified_at). What a call
        # produced is whatever differs from this afterwards; see ``_collect_outputs``.
        self._workspace: dict[str, tuple[int, float | None]] = {}
        # Minted per backend, so per request: what a Responses caller sees as the
        # ``container_id`` of every interpreter call this request ran.
        self.container_id = f"{CONTAINER_ID_PREFIX}{uuid.uuid4().hex}"

    async def __aenter__(self) -> SandboxBackend:
        try:
            headers = {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else None
            self._client = await self._stack.enter_async_context(
                httpx.AsyncClient(timeout=self._timeout_s, headers=headers)
            )
            payload = {"image": self._image} if self._image else {}
            response = await self._client.post(f"{self._sandbox_url}/sessions", json=payload)
            if response.status_code == 503:
                await self._stack.aclose()
                raise SandboxUnavailableError(response.headers.get("Retry-After"))
            response.raise_for_status()
            self._session_id = SessionHandle.model_validate(response.json()).session_id
        except ValidationError as exc:
            await self._stack.aclose()
            raise SandboxNotReachableError(
                f"failed to create sandbox session at {self._sandbox_url}: {_contract_violation(exc)}"
            ) from None
        except (httpx.HTTPError, ValueError) as exc:
            await self._stack.aclose()
            raise SandboxNotReachableError(f"failed to create sandbox session at {self._sandbox_url}: {exc}") from exc
        try:
            await self._seed_inputs()
            if self._files is not None:
                self._workspace = await self._list_workspace()
        except BaseException:
            # The session exists but the request cannot run as asked; release it
            # rather than leaving it to the backend's idle reclaim.
            await self.__aexit__(None, None, None)
            raise
        return self

    async def _seed_inputs(self) -> None:
        """Write every staged upload into the session workspace before the model runs.

        A refused seed is terminal for the request: the code the model writes
        would look for a file that is not there, and a run over a silently
        missing input is worse than no run.
        """
        if self._files is None or not self._files.inputs:
            return
        assert self._client is not None and self._session_id is not None
        for staged in self._files.inputs:
            try:
                data = await self._files.read_input(staged)
            except OSError as exc:
                raise SandboxNotReachableError(f"could not read attachment {staged.file_id} for the sandbox") from exc
            try:
                response = await self._client.post(
                    f"{self._sandbox_url}/sessions/{self._session_id}/files",
                    files={"file": (staged.filename, data, staged.mime_type)},
                    data={"path": staged.filename},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise SandboxNotReachableError(f"sandbox refused attachment {staged.file_id}: {exc}") from exc
            logger.info("sandbox session %s seeded with file %s", self._session_id, staged.file_id)

    async def _list_workspace(self) -> dict[str, tuple[int, float | None]]:
        """The session workspace's files, path -> (size, modified_at); empty when unlistable.

        ``ListFiles`` is optional in the contract, so a backend without it (404,
        or any other failure) simply leaves the diff empty and the result block's
        own file list as the only source of produced files.
        """
        assert self._client is not None and self._session_id is not None
        try:
            response = await self._client.get(f"{self._sandbox_url}/sessions/{self._session_id}/files/list")
            response.raise_for_status()
            entries = response.json().get("files")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.debug("sandbox session %s workspace not listable: %s", self._session_id, exc)
            return {}
        listed: dict[str, tuple[int, float | None]] = {}
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            size = entry.get("size_bytes")
            modified = entry.get("modified_at")
            listed[entry["path"]] = (
                size if isinstance(size, int) else -1,
                float(modified) if isinstance(modified, int | float) else None,
            )
        return listed

    async def _produced_files(self, block: ResultBlock) -> list[str]:
        """The files this call produced: what the block names, plus what the workspace diff shows.

        The contract's result block carries a list of produced files, but not
        every backend fills it in (the reference container reports files only
        through ``ListFiles``), so the two sources are unioned: the block's names
        first, in its order, then every path that appeared or changed since the
        last listing. A seeded input the code rewrote counts as produced.
        """
        names = [ref.filename for ref in block.content.content if ref.filename]
        if self._files is None:
            return names
        after = await self._list_workspace()
        if after:
            names += [path for path, stamp in after.items() if path not in names and self._workspace.get(path) != stamp]
            self._workspace = after
        return names

    async def _collect_outputs(self, block: ResultBlock) -> tuple[list[str], dict[str, str]]:
        """Fetch the files a run produced and store each: the names produced, and filename to file_id.

        Best-effort per file: one that cannot be fetched or stored is still named
        in the rendered result, just without an id, and the run itself stands.
        What a run writes is untrusted, so one call may store at most
        ``max_output_files`` files and ``max_output_bytes`` in total; the rest
        are named only.
        """
        if self._files is None:
            return [], {}
        assert self._client is not None and self._session_id is not None
        produced = await self._produced_files(block)
        max_files = self._files.max_output_files
        if len(produced) > max_files:
            logger.warning("sandbox produced %d files; storing the first %d", len(produced), max_files)
        ids: dict[str, str] = {}
        budget = self._files.max_output_bytes
        for filename in produced[:max_files]:
            try:
                stored = await self._store_output(filename, budget)
            except httpx.HTTPError as exc:
                logger.warning("sandbox output %r could not be fetched: %s", filename, exc)
                continue
            except _OutputOverBudget:
                logger.warning(
                    "sandbox output %r skipped: over the %d byte budget left for this call", filename, budget
                )
                continue
            except Exception as exc:  # noqa: BLE001 — a storage failure must not fail the run
                logger.warning("sandbox output %r could not be stored: %s", filename, exc)
                continue
            if stored is None:
                continue
            file_id, size = stored
            ids[filename] = file_id
            budget -= size
        return produced, ids

    async def _store_output(self, filename: str, budget: int) -> tuple[str, int] | None:
        """Stream one produced file from the sandbox into the store, returning its id and size.

        Never holds the file whole: the bytes go from the sandbox's response to
        the store as they arrive, and the count is checked on the way, so a file
        past ``budget`` is abandoned mid-stream (the store removes the partial
        blob). A declared ``Content-Length`` past it is refused before a byte is
        read. ``None`` for an empty file.
        """
        assert self._client is not None and self._session_id is not None and self._files is not None
        async with self._client.stream(
            "GET", f"{self._sandbox_url}/sessions/{self._session_id}/files", params={"path": filename}
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > budget:
                raise _OutputOverBudget
            counted = _CountedChunks(response.aiter_bytes(), budget)
            file_id = await self._files.store_output(filename, counted)
        if file_id is None:
            logger.warning("sandbox output %r skipped: empty", filename)
            return None
        return file_id, counted.total

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._client is not None and self._session_id is not None:
            try:
                await self._client.delete(f"{self._sandbox_url}/sessions/{self._session_id}")
            except httpx.HTTPError:
                logger.warning("sandbox session %s cleanup failed", self._session_id, exc_info=True)
        await self._stack.aclose()

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [code_execution_tool_definition()] if self._serves_code_execution else []

    def owns_tool(self, name: str) -> bool:
        return name == CODE_EXECUTION_TOOL_NAME and self._serves_code_execution

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(CODE_EXECUTION_TOOL_NAME, self._purpose_hint)] if self._serves_code_execution else []

    @property
    def _serves_code_execution(self) -> bool:
        """Whether the one tool this backend implements survives the allow-list.

        Callers are expected to have refused the request already when it does
        not (``prepare_gateway_tools`` answers 403 rather than opening a backend
        with nothing in it). This is the backend's own half of that: it advertises
        and dispatches the same set, so the two cannot disagree.
        """
        return self._allowed_tools is None or CODE_EXECUTION_TOOL_NAME in self._allowed_tools

    def take_executions(self) -> list[CodeExecution]:
        """The calls executed since the last take, in order, clearing them.

        Consumed by a loop building native result blocks right after the calls it
        awaited. Clearing means a later loop round cannot attribute an earlier
        round's executions to its own calls.
        """
        executions, self._executions = self._executions, []
        return executions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute code and record the call on the request's tally.

        See :class:`gateway.services.tool_usage.ToolUsageTally`: a result carrying
        the ``[tool error]`` sentinel is counted and never billed.
        """
        if not self.owns_tool(name):
            raise KeyError(f"SandboxBackend does not own tool {name!r}")
        code = str(arguments.get("code") or "")
        try:
            result, block, file_ids = await self._exec_tool(code)
        except Exception:
            self._executions.append(CodeExecution(code=code, result=None))
            if self._tally is not None:
                self._tally.record_failure(CODE_EXECUTION_TOOL_NAME)
            raise
        self._executions.append(CodeExecution(code=code, result=block, file_ids=file_ids))
        if self._tally is not None:
            self._tally.record_result(CODE_EXECUTION_TOOL_NAME, result)
        return result

    async def _exec_tool(self, code: str) -> tuple[str, ResultBlock, dict[str, str]]:
        if self._client is None or self._session_id is None:
            raise RuntimeError("SandboxBackend not entered as an async context manager")

        payload = {
            "tool": CODE_EXECUTION_TOOL_NAME,
            "input": {"code": code},
            "timeout_seconds": int(self._timeout_s),
        }
        with tracer.start_as_current_span(
            CODE_EXECUTION_TOOL_NAME,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("tool.name", CODE_EXECUTION_TOOL_NAME)
            span.set_attribute("tool.type", "otari_code_execution")
            span.set_attribute("code_execution.code_size", len(code))
            span.set_attribute("code_execution.backend_url", self._sandbox_url)
            try:
                response = await self._client.post(
                    f"{self._sandbox_url}/sessions/{self._session_id}/exec",
                    json=payload,
                    # Override the client default (which equals the exec budget) so the
                    # sandbox always gets to answer before the client read timeout fires.
                    timeout=self._timeout_s + _EXEC_TIMEOUT_BUFFER_S,
                )
                if response.status_code == 503:
                    raise SandboxUnavailableError(response.headers.get("Retry-After"))
                response.raise_for_status()
                # A malformed body is a contract violation, indistinguishable to the
                # caller from an unreachable backend: both mean this exec produced no
                # usable result, so they raise the same error.
                exec_response = ExecResponse.model_validate(response.json())
            except ValidationError as exc:
                # Raised `from None`, and the summary is built rather than rendered,
                # so neither the message nor the chained traceback carries the
                # payload into the span. See _contract_violation.
                err = SandboxNotReachableError(f"sandbox exec failed: {_contract_violation(exc)}")
                span.record_exception(err)
                span.set_status(trace.StatusCode.ERROR, str(err))
                raise err from None
            except (httpx.HTTPError, ValueError) as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise SandboxNotReachableError(f"sandbox exec failed: {exc}") from exc

            produced, file_ids = await self._collect_outputs(exec_response.result_block)
            result = _flatten_result_block(exec_response.result_block, file_ids, produced)
            if result.startswith("[tool error]"):
                span.set_status(trace.StatusCode.ERROR, result)
            return result, exec_response.result_block, file_ids


def _flatten_result_block(
    block: ResultBlock, file_ids: dict[str, str] | None = None, produced: list[str] | None = None
) -> str:
    """Render the structured result as a single string for the model.

    The tool loop hands the model one string per tool call, so the block's
    fields collapse into labeled sections. Errors come through as a non-zero
    ``return_code`` or a non-empty ``stderr``; the contract has no top-level
    ``is_error`` flag.

    ``produced`` is every file the run wrote, as the block and the workspace
    diff found them; ``file_ids`` maps those that were stored to the ``file_id``
    they were stored under, so the model can hand the user something
    downloadable. A produced file with no id is listed by name alone. Passing
    the full structured result through to the caller (file refs as content
    blocks, per-step exit codes) is a future enhancement that lands alongside
    the Anthropic-content-block lift.
    """
    content = block.content
    file_ids = file_ids or {}
    produced = produced or []

    parts: list[str] = []
    if content.stdout:
        parts.append(f"stdout:\n{content.stdout}")
    if content.stderr:
        parts.append(f"stderr:\n{content.stderr}")
    if content.return_code not in (None, 0):
        parts.append(f"return_code: {content.return_code}")
    listed = [ref.filename or "?" for ref in content.content]
    listed += [name for name in [*produced, *file_ids] if name not in listed]
    if listed:
        names = [f"{name} (file_id: {file_ids[name]})" if name in file_ids else name for name in listed]
        parts.append("files: " + ", ".join(names))

    flattened = "\n".join(parts)
    if not flattened:
        return "(no output)"
    # Treat non-zero return_code or stderr-only output as error-shaped so the
    # model gets a clear signal it can recover from.
    if (content.return_code not in (None, 0)) or (content.stderr and not content.stdout):
        return f"[tool error] {flattened}"
    return flattened
