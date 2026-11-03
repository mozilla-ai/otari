"""Code execution in Chat Completions' ``otari_tool_calls`` field (decision 67)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from gateway.services.tools._chat_tool_calls import capped, chat_entry
from gateway.services.tools._code_execution_responses import CODE_INTERPRETER_CALL_ID_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.sandbox_backend import CodeExecution
    from gateway.services.tools._native import NativeCall


class ChatCodeExecutionRendering:
    """Gateway-run code as ``otari_tool_calls`` entries, one when a run starts and one when it ends.

    The ended entry carries what the sandbox printed, so a caller can read the run's
    output itself rather than trust the model's account of it.
    """

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Every caller: a client that does not know the field ignores it."""
        del tool_entry
        return True

    def started(self, call: NativeCall) -> list[Any]:
        """The entry for a run about to start, with its code."""
        return [_entry(call, "in_progress")]

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The ended entry, from the execution the backend kept for this call.

        A call the backend kept none for (the tool budget refused it) is ``failed``,
        as is a run that never reached the sandbox or exited non-zero.
        """
        take_executions = getattr(pool, "take_executions", None)
        executions: list[CodeExecution] = take_executions() if take_executions is not None else []
        execution = executions[-1] if executions else None
        if execution is None or execution.result is None:
            return [_entry(call, "failed")]
        result = execution.result.content
        stdout, stdout_cut = capped(result.stdout)
        stderr, stderr_cut = capped(result.stderr)
        exit_code = result.return_code or 0
        return [
            _entry(
                call,
                "completed" if exit_code == 0 and not call.failed else "failed",
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                file_ids=list(execution.file_ids.values()) or None,
                truncated=(stdout_cut or stderr_cut) or None,
            )
        ]

    def refused(self, call: NativeCall) -> list[Any]:
        """Nothing: code execution has no use cap, so no call of it is refused."""
        return []


def _entry(call: NativeCall, status: Literal["in_progress", "completed", "failed"], **fields: Any) -> dict[str, Any]:
    code = call.arguments.get("code")
    return chat_entry(
        call,
        id_prefix=CODE_INTERPRETER_CALL_ID_PREFIX,
        kind="code_execution",
        status=status,
        code=code if isinstance(code, str) else None,
        **fields,
    )


RENDERING = ChatCodeExecutionRendering()
