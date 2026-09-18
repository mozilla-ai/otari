"""File history parsing rejects bypasses without interpreting prose."""

import pytest

from gateway.services.provider_files.contracts import FileListRequest, FilesError
from gateway.services.provider_files.references import collect_anthropic_file_references, reject_openai_file_state


@pytest.mark.parametrize("output_type", ["code_execution_output", "bash_code_execution_output"])
def test_complete_history_and_nested_results(output_type: str) -> None:
    history = [
        {"role": "user", "content": [{"type": "document", "source": {"type": "file", "file_id": "file_old"}}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "bash_code_execution_tool_result",
                    "content": {
                        "type": "bash_code_execution_result",
                        "content": [{"type": output_type, "file_id": "file_out"}],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "container_upload", "file_id": "file_old"}]},
    ]
    assert collect_anthropic_file_references(history) == ["file_old", "file_out"]


def test_text_and_tool_input_are_not_references() -> None:
    assert (
        collect_anthropic_file_references(
            [
                {"role": "user", "content": "file_secret"},
                {
                    "type": "tool_use",
                    "input": {"file_id": "file_customer_identifier"},
                },
            ]
        )
        == []
    )


@pytest.mark.parametrize(
    "value",
    [
        {"type": "unknown_file", "file_id": "file_bad"},
        {"type": "file", "file_id": 3},
        [{"type": "file", "file_id": f"file_{n}"} for n in range(101)],
        [None] * 20001,
    ],
)
def test_unsupported_or_excessive_references_fail_closed(value: object) -> None:
    with pytest.raises(FilesError):
        collect_anthropic_file_references(value)


def test_nested_reference_limit() -> None:
    value: object = {"type": "file", "file_id": "file_deep"}
    for _ in range(33):
        value = {"content": value}
    with pytest.raises(FilesError):
        collect_anthropic_file_references(value)


def test_openai_prompt_text_variables_are_not_file_references() -> None:
    reject_openai_file_state(
        {
            "prompt": {
                "id": "pmpt_1",
                "variables": {
                    "doc": "file_mentioned_in_prose",
                    "description": {"type": "input_text", "text": "file_mentioned_in_text"},
                },
            }
        }
    )


def test_ids_cannot_change_pagination_semantics() -> None:
    with pytest.raises(ValueError):
        FileListRequest(ids=["file_a"], limit=20)
