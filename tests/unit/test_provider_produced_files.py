"""Unit tests for files a provider's own sandbox produced.

Covers reading the produced IDs out of each vocabulary's response, which
providers Otari can read back from, and the requests the client makes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from gateway.services.files.provider_files import (
    ANTHROPIC_FILES_BETA,
    FileOverBudgetError,
    ProviderFile,
    ProviderFileClient,
    ProviderFileUnavailableError,
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
    url, headers = _request_for("anthropic", ProviderFile(file_id="file_01abc"), "sk-ant-test", None)

    assert url == "https://api.anthropic.com/v1/files/file_01abc/content"
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-beta"] == ANTHROPIC_FILES_BETA


def test_openai_download_is_keyed_on_the_container() -> None:
    url, headers = _request_for("openai", ProviderFile(file_id="cfile_1", container_id="cntr_1"), "sk-test", None)

    assert url == "https://api.openai.com/v1/containers/cntr_1/files/cfile_1/content"
    assert headers == {"Authorization": "Bearer sk-test"}


def test_an_openai_file_with_no_container_cannot_be_read() -> None:
    # The download is keyed on the container, so a row without one has no URL
    # to build; refusing here is what keeps ``containers/None/...`` off the wire.
    with pytest.raises(LookupError):
        _request_for("openai", ProviderFile(file_id="cfile_1"), "sk-test", None)


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


def test_a_responses_stream_names_a_file_before_it_completes() -> None:
    citation = {"type": "container_file_citation", "file_id": "cfile_1", "filename": "plot.png", "container_id": "c_1"}
    part = SimpleNamespace(annotations=[citation])
    expected = [ProviderFile(file_id="cfile_1", filename="plot.png", container_id="c_1")]

    annotation_added = SimpleNamespace(type="response.output_text.annotation.added", annotation=citation)
    part_done = SimpleNamespace(type="response.content_part.done", part=part)
    item_done = SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(content=[part]))

    assert produced_files_for("responses", annotation_added) == expected
    assert produced_files_for("responses", part_done) == expected
    assert produced_files_for("responses", item_done) == expected


def test_a_citation_field_that_is_not_text_is_dropped() -> None:
    citation = SimpleNamespace(type="container_file_citation", file_id="cfile_1", filename=7, container_id=["c_1"])
    event = SimpleNamespace(type="response.output_text.annotation.added", annotation=citation)

    assert produced_files_for("responses", event) == [ProviderFile(file_id="cfile_1")]


def test_a_configured_api_base_is_where_the_download_goes() -> None:
    url, _ = _request_for(
        "anthropic", ProviderFile(file_id="file_01abc"), "sk-ant-test", "https://anthropic.internal/v1/"
    )

    assert url == "https://anthropic.internal/v1/files/file_01abc/content"


def test_an_id_stays_one_path_segment() -> None:
    url, _ = _request_for("openai", ProviderFile(file_id="../cfile_1", container_id="cntr/1"), "sk-test", None)

    assert url == "https://api.openai.com/v1/containers/cntr%2F1/files/..%2Fcfile_1/content"


def _serving(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` the client opens to ``handler``, returning the requests it saw."""
    seen: list[httpx.Request] = []

    def _recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response: httpx.Response = handler(request)
        return response

    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(_recording)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return seen


def _client(provider: str = "anthropic") -> ProviderFileClient:
    return ProviderFileClient(provider=provider, provider_instance=provider, api_key="sk-test", api_base=None)


async def _read_all(chunks: AsyncGenerator[bytes, None]) -> bytes:
    async with aclosing(chunks) as stream:
        return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_read_streams_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    data = await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert data == b"chart"
    assert seen[0].url.path == "/v1/files/file_01abc/content"


@pytest.mark.asyncio
async def test_read_refuses_a_declared_size_past_the_budget_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    with pytest.raises(FileOverBudgetError):
        await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=4))


@pytest.mark.asyncio
async def test_read_stops_an_undeclared_size_past_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _body() -> AsyncIterator[bytes]:
        yield b"cha"
        yield b"rt"

    _serving(monkeypatch, lambda request: httpx.Response(200, content=_body()))

    with pytest.raises(FileOverBudgetError):
        await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=4))


@pytest.mark.asyncio
async def test_read_raises_when_the_provider_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(404))

    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))


@pytest.mark.asyncio
async def test_read_raises_when_the_connection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _drop(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _serving(monkeypatch, _drop)

    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))


@pytest.mark.asyncio
async def test_read_refuses_an_openai_file_with_no_container() -> None:
    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(_client("openai").read(ProviderFile(file_id="cfile_1"), budget_bytes=5))


@pytest.mark.asyncio
async def test_get_filename_reads_anthropic_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, json={"filename": "bar_plot.png"}))

    assert await _client().get_filename("file_01abc") == "bar_plot.png"
    assert seen[0].url.path == "/v1/files/file_01abc"
    assert seen[0].headers["anthropic-beta"] == ANTHROPIC_FILES_BETA


@pytest.mark.asyncio
async def test_get_filename_is_none_when_the_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(500))

    assert await _client().get_filename("file_01abc") is None


@pytest.mark.asyncio
async def test_get_filename_is_none_when_the_metadata_is_not_an_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(200, json=["bar_plot.png"]))

    assert await _client().get_filename("file_01abc") is None


@pytest.mark.asyncio
async def test_get_filename_asks_nothing_of_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAI's citation already names the file, so there is no metadata call to make.
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, json={"filename": "x"}))

    assert await _client("openai").get_filename("cfile_1") is None
    assert seen == []


def test_for_run_needs_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    config: Any = SimpleNamespace()
    monkeypatch.setattr("gateway.services.files.provider_files.get_provider_kwargs", lambda *args, **kwargs: {})

    with pytest.raises(LookupError):
        ProviderFileClient.for_run(config, provider="anthropic", provider_instance="anthropic", workspace_id=None)
