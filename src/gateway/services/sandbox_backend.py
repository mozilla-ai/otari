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
                              ``session_id``
* ``POST /sessions/{id}/exec``  with ``{tool: "code_execution",
                                        input: {code: "…"},
                                        timeout_seconds: int}``
                              → returns ``{result_block: {…}}``
* ``DELETE /sessions/{id}``  → tears the session down

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
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx
from opentelemetry import trace
from pydantic import ValidationError

from gateway.services.tool_usage import ToolUsageTally
from gateway.types.code_execution import ExecResponse, ResultBlock, SessionHandle

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

CODE_EXECUTION_TOOL_NAME = "code_execution"
_DEFAULT_TIMEOUT_S = 60.0
# Headroom added on top of the execution budget for the exec POST's own read
# timeout. The sandbox is granted ``timeout_seconds`` to run the code; the HTTP
# client must wait longer than that (network + serialization + the sandbox's own
# teardown) so a legitimate near-max execution returns its result instead of
# tripping the client read timeout as a spurious ``SandboxNotReachableError``.
_EXEC_TIMEOUT_BUFFER_S = 10.0
_DEFAULT_PURPOSE_HINT = (
    "Prefer `code_execution` for any computation, data analysis, date "
    "arithmetic, statistics, or anything that benefits from exact output. "
    "Python with numpy/pandas/scipy/sympy/matplotlib pre-installed."
)


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


class SandboxNotReachableError(RuntimeError):
    """Raised when the sandbox container can't be reached or returns malformed data."""


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
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        auth_token: str | None = None,
        tally: ToolUsageTally | None = None,
    ) -> None:
        self._sandbox_url = sandbox_url.rstrip("/")
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
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._stack: AsyncExitStack = AsyncExitStack()

    async def __aenter__(self) -> SandboxBackend:
        try:
            headers = {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else None
            self._client = await self._stack.enter_async_context(
                httpx.AsyncClient(timeout=self._timeout_s, headers=headers)
            )
            response = await self._client.post(f"{self._sandbox_url}/sessions", json={})
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
        return self

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
        return [code_execution_tool_definition()]

    def owns_tool(self, name: str) -> bool:
        return name == CODE_EXECUTION_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(CODE_EXECUTION_TOOL_NAME, self._purpose_hint)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute code and record the call on the request's tally.

        See :class:`gateway.services.tool_usage.ToolUsageTally`: a result carrying
        the ``[tool error]`` sentinel is counted and never billed.
        """
        if name != CODE_EXECUTION_TOOL_NAME:
            raise KeyError(f"SandboxBackend does not own tool {name!r}")
        try:
            result = await self._exec_tool(arguments)
        except Exception:
            if self._tally is not None:
                self._tally.record_failure(CODE_EXECUTION_TOOL_NAME)
            raise
        if self._tally is not None:
            self._tally.record_result(CODE_EXECUTION_TOOL_NAME, result)
        return result

    async def _exec_tool(self, arguments: dict[str, Any]) -> str:
        if self._client is None or self._session_id is None:
            raise RuntimeError("SandboxBackend not entered as an async context manager")

        code = arguments.get("code") or ""
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

            result = _flatten_result_block(exec_response.result_block)
            if result.startswith("[tool error]"):
                span.set_status(trace.StatusCode.ERROR, result)
            return result


def _flatten_result_block(block: ResultBlock) -> str:
    """Render the structured result as a single string for the model.

    The tool loop hands the model one string per tool call, so the block's
    fields collapse into labeled sections. Errors come through as a non-zero
    ``return_code`` or a non-empty ``stderr``; the contract has no top-level
    ``is_error`` flag.

    Passing the full structured result through to the caller (file refs as
    content blocks, per-step exit codes) is a future enhancement that lands
    alongside the Anthropic-content-block lift.
    """
    content = block.content

    parts: list[str] = []
    if content.stdout:
        parts.append(f"stdout:\n{content.stdout}")
    if content.stderr:
        parts.append(f"stderr:\n{content.stderr}")
    if content.return_code not in (None, 0):
        parts.append(f"return_code: {content.return_code}")
    if content.content:
        parts.append("files: " + ", ".join(ref.filename or "?" for ref in content.content))

    flattened = "\n".join(parts)
    if not flattened:
        return "(no output)"
    # Treat non-zero return_code or stderr-only output as error-shaped so the
    # model gets a clear signal it can recover from.
    if (content.return_code not in (None, 0)) or (content.stderr and not content.stdout):
        return f"[tool error] {flattened}"
    return flattened
