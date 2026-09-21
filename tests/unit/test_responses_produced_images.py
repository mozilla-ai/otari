"""Unit tests for how a gateway-run execution announces produced files on Responses.

OpenAI's ``code_interpreter_call`` has no shape for a file id, only logs and an
``image`` URL, so a produced image is announced as the address Otari serves it
from and anything else is left to the files API.
"""

from __future__ import annotations

from gateway.services.mcp_loop_responses import _code_interpreter_call_item
from gateway.services.sandbox_backend import CodeExecution
from gateway.types.code_execution import CodeExecutionResult, ResultBlock

BASE = "https://otari.example.com/api/v1/files"


def _execution(**file_ids: str) -> CodeExecution:
    return CodeExecution(
        code="print('hi')",
        result=ResultBlock(
            type="code_execution_result",
            content=CodeExecutionResult(stdout="hi\n", stderr="", return_code=0),
        ),
        file_ids=dict(file_ids),
    )


def test_a_produced_image_is_announced_as_the_url_otari_serves_it_from() -> None:
    item = _code_interpreter_call_item(_execution(**{"bar_plot.png": "file-abc"}), "otari_cntr_1", BASE)

    assert item.outputs is not None
    kinds = [output.type for output in item.outputs]
    assert kinds == ["logs", "image"]
    image = item.outputs[1]
    assert image.type == "image"
    assert image.url == f"{BASE}/file-abc/content"
    assert item.status == "completed"


def test_a_non_image_is_left_to_the_files_api() -> None:
    item = _code_interpreter_call_item(_execution(**{"table.csv": "file-abc"}), "otari_cntr_1", BASE)

    assert item.outputs is not None
    assert [output.type for output in item.outputs] == ["logs"]


def test_a_run_outside_a_request_announces_no_url() -> None:
    """No base URL is the direct-use case (tests, a backend built by hand)."""
    item = _code_interpreter_call_item(_execution(**{"bar_plot.png": "file-abc"}), "otari_cntr_1", None)

    assert item.outputs is not None
    assert [output.type for output in item.outputs] == ["logs"]


def test_a_deployment_with_no_public_address_announces_a_root_relative_url() -> None:
    item = _code_interpreter_call_item(_execution(**{"bar_plot.png": "file-abc"}), "otari_cntr_1", "/api/v1/files")

    assert item.outputs is not None
    image = item.outputs[1]
    assert image.type == "image"
    assert image.url == "/api/v1/files/file-abc/content"


def test_an_image_is_announced_even_when_the_run_logged_nothing() -> None:
    execution = CodeExecution(
        code="savefig()",
        result=ResultBlock(
            type="code_execution_result",
            content=CodeExecutionResult(stdout="", stderr="", return_code=0),
        ),
        file_ids={"bar_plot.png": "file-abc"},
    )

    item = _code_interpreter_call_item(execution, "otari_cntr_1", BASE)

    assert item.outputs is not None
    assert [output.type for output in item.outputs] == ["image"]


def test_a_run_the_backend_never_answered_announces_nothing() -> None:
    item = _code_interpreter_call_item(CodeExecution(code="print(1)", result=None), "otari_cntr_1", BASE)

    assert item.outputs is None
    assert item.status == "failed"
