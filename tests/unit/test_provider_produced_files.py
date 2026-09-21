"""Unit tests for files a provider's own sandbox produced.

Covers the pure parts: reading the produced ids out of each vocabulary's
response, which provider Otari can fetch back from, and the request it makes.
The download itself is exercised over HTTP in
tests/integration/test_files_endpoint.py.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.models.tools import FileObject
from gateway.services.files.provider_files import (
    ANTHROPIC_FILES_BETA,
    ProviderFile,
    _request_for,
    anthropic_produced_files,
    produced_files_for,
    responses_produced_files,
    serves_files,
)


def _anthropic_reply(*outputs: list[dict[str, str]]) -> SimpleNamespace:
    """A Messages response whose tool-result blocks carry ``outputs``."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="code_execution_tool_result",
                content=SimpleNamespace(content=[SimpleNamespace(**output) for output in group]),
            )
            for group in outputs
        ]
    )


def test_anthropic_outputs_are_read_whichever_variant_ran() -> None:
    reply = _anthropic_reply(
        [{"type": "code_execution_output", "file_id": "file_01python"}],
        [{"type": "bash_code_execution_output", "file_id": "file_01bash"}],
    )

    assert [file.file_id for file in anthropic_produced_files(reply)] == ["file_01python", "file_01bash"]


def test_anthropic_reply_with_no_files_produces_nothing() -> None:
    assert anthropic_produced_files(_anthropic_reply([])) == []
    assert anthropic_produced_files(SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])) == []


def test_an_id_cited_twice_is_recorded_once() -> None:
    reply = _anthropic_reply(
        [{"type": "code_execution_output", "file_id": "file_01same"}],
        [{"type": "code_execution_output", "file_id": "file_01same"}],
    )

    assert anthropic_produced_files(reply) == [ProviderFile(file_id="file_01same")]


def test_responses_citations_carry_the_container_and_the_name() -> None:
    reply = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        annotations=[
                            SimpleNamespace(
                                type="container_file_citation",
                                container_id="cntr_1",
                                file_id="cfile_1",
                                filename="bar_plot.png",
                            ),
                            SimpleNamespace(type="url_citation", url="https://example.com"),
                        ],
                    )
                ],
            )
        ]
    )

    assert responses_produced_files(reply) == [
        ProviderFile(file_id="cfile_1", filename="bar_plot.png", container_id="cntr_1")
    ]


def test_only_providers_otari_can_read_back_are_recorded() -> None:
    assert serves_files("anthropic")
    assert serves_files("openai")
    assert not serves_files("nebius")


def test_anthropic_download_takes_the_id_alone() -> None:
    record = SimpleNamespace(
        id="file_01abc",
        provider="anthropic",
        provider_container_id=None,
        workspace_id=uuid.uuid4(),
    )

    url, headers = _request_for(cast(FileObject, cast(Any, record)), "sk-ant-test", None)

    assert url == "https://api.anthropic.com/v1/files/file_01abc/content"
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-beta"] == ANTHROPIC_FILES_BETA


def test_openai_download_is_keyed_on_the_container() -> None:
    record = SimpleNamespace(
        id="cfile_1",
        provider="openai",
        provider_container_id="cntr_1",
        workspace_id=uuid.uuid4(),
    )

    url, headers = _request_for(cast(FileObject, cast(Any, record)), "sk-test", None)

    assert url == "https://api.openai.com/v1/containers/cntr_1/files/cfile_1/content"
    assert headers == {"Authorization": "Bearer sk-test"}


def test_an_openai_file_with_no_container_cannot_be_read() -> None:
    # The download is keyed on the container, so a row without one has no URL
    # to build; refusing here is what keeps ``containers/None/...`` off the wire.
    record = SimpleNamespace(id="cfile_1", provider="openai", provider_container_id=None, workspace_id=None)

    with pytest.raises(LookupError):
        _request_for(cast(FileObject, cast(Any, record)), "sk-test", None)


def test_a_streamed_messages_result_block_names_its_files() -> None:
    reply = _anthropic_reply([{"file_id": "file_01abc"}])
    event = SimpleNamespace(type="content_block_start", index=0, content_block=reply.content[0])

    assert produced_files_for("messages", event) == [ProviderFile(file_id="file_01abc")]
    # A delta carries no block, and a completed reply is read whole.
    assert produced_files_for("messages", SimpleNamespace(type="content_block_delta", delta=None)) == []
    assert produced_files_for("messages", reply) == [ProviderFile(file_id="file_01abc")]


def test_a_streamed_responses_completion_names_its_files() -> None:
    citation = SimpleNamespace(
        type="container_file_citation", file_id="cfile_1", filename="bar_plot.png", container_id="cntr_1"
    )
    response = SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(annotations=[citation])])])
    event = SimpleNamespace(type="response.completed", response=response)

    expected = [ProviderFile(file_id="cfile_1", filename="bar_plot.png", container_id="cntr_1")]
    assert produced_files_for("responses", event) == expected
    assert produced_files_for("responses", response) == expected
    assert produced_files_for("responses", SimpleNamespace(type="response.output_text.delta")) == []
    # Chat Completions has no native code tool, so nothing is ever read off it.
    assert produced_files_for("chat", response) == []


def test_a_configured_api_base_is_where_the_download_goes() -> None:
    record = SimpleNamespace(id="file_01abc", provider="anthropic", provider_container_id=None, workspace_id=None)

    url, _ = _request_for(cast(FileObject, cast(Any, record)), "sk-ant-test", "https://anthropic.internal/v1/")

    assert url == "https://anthropic.internal/v1/files/file_01abc/content"
