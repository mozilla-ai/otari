"""Code execution in the Anthropic Messages server-tool vocabulary."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from anthropic.types import (
    CodeExecutionOutputBlock,
    CodeExecutionResultBlock,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultError,
    ServerToolUseBlock,
)

from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, CodeExecution
from gateway.services.tools._code_execution_declarations import native_code_execution_dialect
from gateway.services.tools._native import SERVER_TOOL_USE_ID_PREFIX, Dialect

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall


def _execution_blocks(execution: CodeExecution) -> list[Any]:
    """A ``server_tool_use`` / ``code_execution_tool_result`` pair for one gateway execution.

    Emitted for a caller that declared code execution in Anthropic's own
    vocabulary and whose request the gateway's sandbox ran instead. The result
    block is the contract's own shape, which mirrors Anthropic's, so a client
    parsing Anthropic responses reads it with no translation. A call the backend
    never answered is reported in the vocabulary's error shape rather than
    dropped, because the model was told about the failure and the client should
    see the same story.
    """
    tool_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}"
    content: CodeExecutionResultBlock | CodeExecutionToolResultError
    if execution.result is None:
        content = CodeExecutionToolResultError(type="code_execution_tool_result_error", error_code="unavailable")
    else:
        result = execution.result.content
        content = CodeExecutionResultBlock(
            type="code_execution_result",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code if result.return_code is not None else 0,
            # The ids are the ones ``/v1/files`` serves, not the sandbox's own: a
            # produced file that was not stored has no id the caller could use.
            content=[
                CodeExecutionOutputBlock(type="code_execution_output", file_id=file_id)
                for file_id in execution.file_ids.values()
            ],
        )
    return [
        ServerToolUseBlock(
            id=tool_use_id,
            name=cast('Literal["code_execution"]', CODE_EXECUTION_TOOL_NAME),
            input={"code": execution.code},
            type="server_tool_use",
        ),
        CodeExecutionToolResultBlock(tool_use_id=tool_use_id, type="code_execution_tool_result", content=content),
    ]


class MessagesCodeExecutionRendering:
    """Gateway-run code as ``server_tool_use`` / ``code_execution_tool_result`` pairs."""

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Whether the caller asked in Anthropic's dated keyword, which is what asks for the pair."""
        return native_code_execution_dialect(tool_entry) is Dialect.MESSAGES

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The pair for each execution the backend kept since the last call, whether or not it succeeded.

        A non-zero exit is a result the vocabulary can carry, and a call the backend
        never ran has an error shape of its own. The executions come from the buffer
        only the gateway's sandbox backend keeps, which a loop drains right after each
        call it awaited, so they are this call's.
        """
        del call
        take_executions = getattr(pool, "take_executions", None)
        if take_executions is None:
            return []
        return [block for execution in take_executions() for block in _execution_blocks(execution)]

    def refused(self, call: NativeCall) -> list[Any]:
        """Nothing: code execution has no use cap, so no call of it is refused."""
        return []


RENDERING = MessagesCodeExecutionRendering()
