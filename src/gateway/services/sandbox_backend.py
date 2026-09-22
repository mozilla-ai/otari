"""Dispatch `code_execution` tool calls to a sandbox.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches
to whenever the model emits a ``code_execution(code=…)`` call. Everything here
is the half that does not depend on how the code is actually run: the tool the
model is offered, the allow-list, the usage tally, seeding the request's
uploads and collecting what a run produced. Reaching a sandbox at all belongs
to :mod:`gateway.ports.code_execution_port` and the adapters under it, so a
container the operator runs and a hosted provider are the same to this module.
The shapes a run returns are typed in :mod:`gateway.types.code_execution`.

Session lifecycle: enter leases a session, exit releases it, unless a container
registry (``services/code_execution``) is handed in and the adapter says the
session really will outlive the block. Then exit asks the port to hold the
sandbox and records the lease under the ``container_id`` the response carried,
and a later request naming that id enters by resuming it, with the workspace as
the last run left it. A backend that declines to hold a session is released as
it always was, and no container id is reported for it. A warm pool is still
nobody's problem here.

This backend satisfies the same duck-typed protocol the MCP loop uses
for tool dispatch (``openai_tools``, ``owns_tool``, ``purpose_hints``,
``call_tool``), so the loop accepts it as a ``pool`` without any
refactor to :func:`gateway.services.mcp_loop.mcp_tool_loop`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from opentelemetry import trace

from gateway.ports.code_execution_port import (
    CodeExecutionPort,
    CodeExecutionSession,
    OutputOverBudget,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.services.code_execution import CONTAINER_ID_PREFIX, ContainerLease, SandboxContainers, new_container_id
from gateway.services.tool_usage import ToolUsageTally
from gateway.types.code_execution import ResultBlock

if TYPE_CHECKING:
    from types import TracebackType

    from gateway.services.file_service import StagedFile

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ``SandboxNotReachableError`` and ``SandboxUnavailableError`` are the port's,
# because the adapters raise them, and are re-exported here so the routes and
# tests that have always caught them from this module still can.
__all__ = [
    "CODE_EXECUTION_TOOL_NAME",
    "CODE_EXECUTION_TOOL_NAMES",
    "CONTAINER_ID_PREFIX",
    "CodeExecution",
    "SandboxBackend",
    "SandboxFiles",
    "SandboxNotReachableError",
    "SandboxSessionGoneError",
    "SandboxUnavailableError",
    "code_execution_tool_definition",
]

CODE_EXECUTION_TOOL_NAME = "code_execution"
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
# Headroom on top of what the calls themselves may spend, for the lease a
# session is opened with: seeding the uploads and collecting the outputs happen
# outside any execution budget, and a provider that reclaims on its own timer
# must not take the sandbox away while they run.
_SESSION_TTL_SLACK_S = 60.0
# How many code calls one tool-loop round is assumed to carry. The iteration cap
# bounds rounds, not calls: a model may emit several code calls in one round and
# nothing narrows that, so the cap alone under-counts what a request can spend.
# Guessing high is the cheap direction, because the session is released when the
# request ends and the provider's own timer is only the backstop for a release
# that never ran.
_CALLS_PER_ROUND_ALLOWANCE = 4
# The longest lease worth asking for. Providers cap how long they will hold a
# sandbox, and a plan's ceiling is a refused creation rather than a shorter
# lease, so a generous estimate is clamped rather than sent.
_MAX_SESSION_TTL_S = 3600.0
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
            raise OutputOverBudget
        return chunk


class SandboxBackend:
    """Async context manager that owns one sandbox session for a request's lifetime.

    Usage::

        port = ProtocolCodeExecutionAdapter("http://sandbox:8080")
        async with SandboxBackend(port=port, max_executions=N) as backend:
            # backend duck-types as the MCP loop's `pool` parameter
            result = await mcp_tool_loop(
                completion_kwargs=kwargs, pool=backend, max_iterations=N,
            )
    """

    def __init__(
        self,
        *,
        port: CodeExecutionPort,
        purpose_hint: str | None = None,
        timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
        max_executions: int = 1,
        auth_token: str | None = None,
        image: str | None = None,
        allowed_tools: frozenset[str] | None = None,
        tally: ToolUsageTally | None = None,
        files: SandboxFiles | None = None,
        files_base_url: str | None = None,
        container: ContainerLease | None = None,
        containers: SandboxContainers | None = None,
        on_lease: Callable[[ContainerLease], None] | None = None,
    ) -> None:
        self._port = port
        # The lease this request resumes, or None for a fresh sandbox, and the
        # registry that holds the session past this request. Both None where
        # reuse is off or there is no database to remember a lease in.
        self._resume = container
        self._containers = containers
        self._on_lease = on_lease
        # The lease this request holds once the session is open, for the route
        # to report. Only set when a registry will hold the sandbox.
        self.lease: ContainerLease | None = None
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
        # How long the lease has to last, as against what one call may spend.
        # A request runs code at least once per tool-loop round, so a provider
        # that reclaims a sandbox on a timer of its own (E2B) has to be told the
        # whole request's worth or it takes the sandbox away mid-loop.
        self._session_ttl_s = min(
            timeout_s * max(max_executions, 1) * _CALLS_PER_ROUND_ALLOWANCE + _SESSION_TTL_SLACK_S,
            _MAX_SESSION_TTL_S,
        )
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
        self._session: CodeExecutionSession | None = None
        self._stack: AsyncExitStack = AsyncExitStack()
        # The calls executed since the last ``take_executions``, in order. A loop
        # that mints native result blocks drains this right after the awaited
        # calls it made, which is what keeps a batch's blocks paired with the
        # right calls: every tool loop runs its calls one at a time and in order.
        self._executions: list[CodeExecution] = []
        # The workspace as last listed, path -> (size, modified_at). What a call
        # produced is whatever differs from this afterwards; see ``_collect_outputs``.
        self._workspace: dict[str, tuple[int, float | None]] = {}
        # What a caller sees as the container behind this request: the id it
        # resumed, else one minted here. Kept across resumes, so the same
        # sandbox has one name for as long as it lives.
        self.container_id = container.container_id if container is not None else new_container_id()

    async def __aenter__(self) -> SandboxBackend:
        keep_alive_s = self._containers.keep_alive_s(self._resume) if self._containers is not None else None
        session = await self._stack.enter_async_context(
            self._port.open_session(
                image=self._image,
                timeout_s=self._timeout_s,
                session_ttl_s=self._session_ttl_s,
                auth_token=self._auth_token,
                resume=self._resume.provider_session_id if self._resume is not None else None,
                keep_alive_s=keep_alive_s,
            )
        )
        self._session = session
        # Only where the adapter says the session really will outlive this block:
        # a backend that declined to hold it releases on exit, and a container id
        # promising a sandbox that is already gone is worse than none.
        if self._containers is not None and session.holds_across_requests:
            self.lease = self._containers.lease(self.container_id, session.session_id, resumed=self._resume)
            if self._on_lease is not None:
                self._on_lease(self.lease)
        try:
            await self._seed_inputs()
            if self._files is not None:
                self._workspace = await self._list_workspace()
        except BaseException:
            # The session exists but the request cannot run as asked, so it is
            # released rather than held: no caller was told this container id,
            # and holding it would bill the deployment for a sandbox nobody can
            # reach. Dropping the lease is what keeps the row from being written.
            session.discard()
            self.lease = None
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
        assert self._session is not None
        for staged in self._files.inputs:
            try:
                data = await self._files.read_input(staged)
            except OSError as exc:
                raise SandboxNotReachableError(f"could not read attachment {staged.file_id} for the sandbox") from exc
            try:
                await self._session.put_file(staged.filename, data, mime_type=staged.mime_type)
            except SandboxNotReachableError as exc:
                raise SandboxNotReachableError(f"sandbox refused attachment {staged.file_id}: {exc}") from exc
            logger.info("sandbox session %s seeded with file %s", self._session.session_id, staged.file_id)

    async def _list_workspace(self) -> dict[str, tuple[int, float | None]]:
        """The session workspace's files, path -> (size, modified_at); empty when unlistable.

        Listing is optional, so an adapter whose backend cannot do it reports
        nothing, which leaves the result block's own file list as the only
        source of produced files.
        """
        assert self._session is not None
        return {entry.path: (entry.size_bytes, entry.modified_at) for entry in await self._session.list_files()}

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
        produced = await self._produced_files(block)
        max_files = self._files.max_output_files
        if len(produced) > max_files:
            logger.warning("sandbox produced %d files; storing the first %d", len(produced), max_files)
        ids: dict[str, str] = {}
        budget = self._files.max_output_bytes
        for filename in produced[:max_files]:
            # What the listing already said about the file, where it said
            # anything: refusing here keeps an adapter whose provider hands a
            # file over whole from being asked for one that cannot be stored.
            listed = self._workspace.get(filename)
            if listed is not None and listed[0] > budget:
                logger.warning(
                    "sandbox output %r skipped: over the %d byte budget left for this call", filename, budget
                )
                continue
            try:
                stored = await self._store_output(filename, budget)
            except SandboxNotReachableError as exc:
                logger.warning("sandbox output %r could not be fetched: %s", filename, exc)
                continue
            except OutputOverBudget:
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

        Never holds the file whole: the bytes go from the sandbox to the store
        as they arrive, and the count is checked on the way, so a file past
        ``budget`` is abandoned mid-stream and the store removes the partial
        blob. An adapter that learns the size first refuses before a byte is
        read. ``None`` for an empty file.
        """
        assert self._session is not None and self._files is not None
        async with aclosing(self._session.read_file(filename, budget_bytes=budget)) as chunks:
            counted = _CountedChunks(chunks, budget)
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
        # Releasing, or holding, the session is the port's; this unwinds the
        # block it was entered in, which is what triggers either.
        self._session = None
        await self._stack.aclose()
        if self._containers is None:
            return
        if self.lease is not None:
            # Written after the hold, so a row never promises a sandbox the
            # provider was not yet asked to keep. Recording is also what gives
            # back the claim this request took at admission. A write that fails
            # leaves the sandbox held but the id unresumable, which the
            # provider's timer then cleans up; the run itself stands.
            try:
                await self._containers.record(self.lease)
            except Exception:  # noqa: BLE001 - bookkeeping must not fail a completed run
                logger.warning("could not record container lease %s", self.lease.container_id, exc_info=True)
        elif self._resume is not None:
            # Resumed and then not held: nothing will record a lease, so the
            # claim has to be handed back here rather than waiting out its TTL.
            try:
                await self._containers.release(self._resume.container_id)
            except Exception:  # noqa: BLE001 - the claim expires on its own either way
                logger.warning("could not release container claim %s", self._resume.container_id, exc_info=True)

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
        if self._session is None:
            raise RuntimeError("SandboxBackend not entered as an async context manager")

        with tracer.start_as_current_span(
            CODE_EXECUTION_TOOL_NAME,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("tool.name", CODE_EXECUTION_TOOL_NAME)
            span.set_attribute("tool.type", "otari_code_execution")
            span.set_attribute("code_execution.code_size", len(code))
            span.set_attribute("code_execution.backend", self._port.label)
            span.set_attribute("code_execution.session_id", self._session.session_id)
            try:
                block = await self._session.execute(code, timeout_s=self._timeout_s)
            except SandboxNotReachableError as exc:
                # The adapter wrapped whatever went wrong, so the span records
                # the cause where there is one: a connect error is worth seeing
                # by its own type. A reply that failed validation is raised with
                # no cause on purpose, and records as the wrapper, so neither the
                # payload nor a credential reaches the span.
                recorded = exc.__cause__ or exc
                span.record_exception(recorded)
                span.set_status(trace.StatusCode.ERROR, str(recorded))
                raise

            produced, file_ids = await self._collect_outputs(block)
            result = _flatten_result_block(block, file_ids, produced)
            if result.startswith("[tool error]"):
                span.set_status(trace.StatusCode.ERROR, result)
            return result, block, file_ids


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
