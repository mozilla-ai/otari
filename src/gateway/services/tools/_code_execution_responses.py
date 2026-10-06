"""Code execution in the OpenAI Responses server-tool vocabulary."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseCodeInterpreterToolCall
from openai.types.responses.response_code_interpreter_tool_call import OutputImage, OutputLogs

from gateway.services.files import guess_mime_type
from gateway.services.sandbox_backend import CodeExecution
from gateway.services.tools._code_execution_declarations import native_code_execution_dialect
from gateway.services.tools._native import Dialect

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall

# The gateway's own ``code_interpreter_call`` item ids. OpenAI issues ``ci_``
# ids, so a reserved prefix is what lets an echoed item be told apart from one
# describing a run OpenAI's own interpreter did (see ``routes/responses.py``).
CODE_INTERPRETER_CALL_ID_PREFIX = "otari_ci_"


def _produced_image_outputs(execution: CodeExecution, files_base_url: str | None) -> list[OutputImage]:
    """``image`` outputs for the images a run produced, in the caller's vocabulary.

    OpenAI's only shape for a produced file here is a URL, so an image is
    announced as the address Otari serves it from and anything else is left to
    the files API, where every produced file is listed and downloadable by id.
    """
    if not files_base_url:
        return []
    return [
        OutputImage(type="image", url=f"{files_base_url}/{file_id}/content")
        for filename, file_id in execution.file_ids.items()
        if guess_mime_type(filename).startswith("image/")
    ]


def _code_interpreter_call_item(
    execution: CodeExecution, container_id: str, files_base_url: str | None = None
) -> ResponseCodeInterpreterToolCall:
    """The Responses API's native "the server ran code" output item, for one gateway execution.

    Emitted for a caller that declared ``code_interpreter`` and whose request the
    gateway's sandbox ran instead. ``outputs`` carries the run's logs, which is
    what OpenAI's interpreter reports too, and an ``image`` entry per produced
    image (see :func:`_produced_image_outputs`).
    """
    outputs: list[OutputLogs | OutputImage] | None = None
    status: str = "failed"
    if execution.result is not None:
        result = execution.result.content
        logs = "".join(part for part in (result.stdout, result.stderr) if part)
        outputs = [OutputLogs(type="logs", logs=logs)] if logs else None
        images = _produced_image_outputs(execution, files_base_url)
        if images:
            outputs = [*(outputs or []), *images]
        status = "completed" if result.return_code in (None, 0) else "failed"
    return ResponseCodeInterpreterToolCall(
        id=f"{CODE_INTERPRETER_CALL_ID_PREFIX}{uuid.uuid4().hex}",
        code=execution.code,
        container_id=container_id,
        outputs=outputs,
        status=status,  # type: ignore[arg-type]
        type="code_interpreter_call",
    )


class ResponsesCodeExecutionRendering:
    """Gateway-run code as ``code_interpreter_call`` output items."""

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Whether the caller declared OpenAI's ``code_interpreter``, which is what asks for the item."""
        return native_code_execution_dialect(dict(tool_entry or {})) is Dialect.RESPONSES

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """An item for each execution the backend kept since the last call.

        A loop drains the buffer right after each call it awaited, so the items land
        at that call's place among the batch's other native items.
        """
        del call
        take_executions = getattr(pool, "take_executions", None)
        if take_executions is None:
            return []
        container_id = str(getattr(pool, "container_id", "") or "")
        files_base_url = getattr(pool, "files_base_url", None)
        return [_code_interpreter_call_item(execution, container_id, files_base_url) for execution in take_executions()]

    def refused(self, call: NativeCall) -> list[Any]:
        """Nothing: code execution has no use cap, so no call of it is refused."""
        return []


RENDERING = ResponsesCodeExecutionRendering()
