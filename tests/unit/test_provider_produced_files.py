"""Unit tests for files a provider's own sandbox produced.

Covers reading the produced IDs out of each vocabulary's response, which
providers Otari can read back from, and the requests the client makes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from any_llm import LLMProvider
from any_llm.types.files import AsyncFileDownload

from gateway.services.files.provider_files import (
    _TIMEOUT,
    FileOverBudgetError,
    ProviderCredential,
    ProviderFile,
    ProviderFileClient,
    ProviderFileUnavailableError,
    _container_file_request,
    _declared_size,
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


def test_openai_download_is_keyed_on_the_container() -> None:
    url, headers = _container_file_request(
        ProviderFile(file_id="cfile_1", container_id="cntr_1"), ProviderCredential(api_key="sk-test")
    )

    assert url == "https://api.openai.com/v1/containers/cntr_1/files/cfile_1/content"
    assert headers == {"Authorization": "Bearer sk-test"}


def test_an_openai_file_with_no_container_cannot_be_read() -> None:
    # The download is keyed on the container, so a row without one has no URL
    # to build; refusing here is what keeps ``containers/None/...`` off the wire.
    with pytest.raises(ProviderFileUnavailableError, match="names no container"):
        _container_file_request(ProviderFile(file_id="cfile_1"), ProviderCredential(api_key="sk-test"))


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


def test_an_id_stays_one_path_segment() -> None:
    url, _ = _container_file_request(
        ProviderFile(file_id="../cfile_1", container_id="cntr/1"), ProviderCredential(api_key="sk-test")
    )

    assert url == "https://api.openai.com/v1/containers/cntr%2F1/files/..%2Fcfile_1/content"


@pytest.mark.asyncio
@pytest.mark.parametrize("file_id", ["../secret", "..", "a\\b", ""])
async def test_an_anthropic_id_that_could_escape_its_path_reaches_no_provider(
    monkeypatch: pytest.MonkeyPatch, file_id: str
) -> None:
    # any-llm rejects these rather than Otari, so this pins the behavior the
    # Anthropic path relies on instead of validating the ID itself.
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(_client().read(ProviderFile(file_id=file_id), budget_bytes=5))
    assert seen == []


def test_a_provider_whose_files_otari_cannot_read_is_refused() -> None:
    # serves_files names the set; refusing here is what stops a third provider's
    # credential reaching OpenAI's container endpoint.
    with pytest.raises(LookupError, match="nebius"):
        ProviderFileClient(
            provider=LLMProvider.NEBIUS, provider_instance="nebius", credential=ProviderCredential(api_key="sk-test")
        )


@pytest.mark.asyncio
async def test_both_providers_read_under_one_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))
    anthropic, openai = _client(), _client("openai")

    # Without this the Anthropic path would inherit the SDK's own 600s read timeout.
    assert cast("Any", anthropic._llm).client.timeout == _TIMEOUT
    assert anthropic._connection.timeout == _TIMEOUT
    assert openai._connection.timeout == _TIMEOUT


@pytest.mark.asyncio
async def test_a_failed_read_is_tried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The copy shares one deadline across every file of a request, so retrying
    # here spends the time the remaining files need. The OpenAI path never retries.
    seen = _serving(monkeypatch, lambda request: httpx.Response(500))

    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_connection_that_will_not_close_does_not_reach_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    async def _refuses() -> None:
        raise RuntimeError("event loop is closed")

    monkeypatch.setattr(client._connection, "aclose", _refuses)

    await client.aclose()


@pytest.mark.asyncio
async def test_aclose_releases_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))
    client = _client()

    await _read_all(client.read(ProviderFile(file_id="file_01abc"), budget_bytes=5))
    await client.aclose()

    assert client._connection.is_closed


@pytest.mark.asyncio
async def test_a_declared_size_is_read_whatever_case_the_header_arrives_in() -> None:
    # AsyncFileDownload promises a plain mapping, which no HTTP header class need back.
    download = AsyncFileDownload(status_code=200, headers={"Content-Length": "9"}, chunks=_no_chunks())

    assert _declared_size(download) == 9


def _serving(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` the client opens to ``handler``, returning the requests it saw.

    Patches the constructor rather than passing a transport, because the
    provider SDK builds clients of its own that a caller cannot reach.
    """
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


def _client(
    provider: str = "anthropic", api_base: str | None = None, client_args: dict[str, Any] | None = None
) -> ProviderFileClient:
    member = LLMProvider(provider)
    credential = ProviderCredential(api_key="sk-test", api_base=api_base, client_args=client_args or {})
    return ProviderFileClient(provider=member, provider_instance=provider, credential=credential)


def _metadata(filename: str = "bar_plot.png") -> dict[str, Any]:
    """Anthropic's file metadata, with every field its SDK requires."""
    return {
        "id": "file_01abc",
        "type": "file",
        "filename": filename,
        "mime_type": "image/png",
        "size_bytes": 5,
        "created_at": "2026-01-01T00:00:00Z",
    }


async def _no_chunks() -> AsyncIterator[bytes]:
    return
    yield  # pragma: no cover


async def _read_all(chunks: AsyncGenerator[bytes, None]) -> bytes:
    async with aclosing(chunks) as stream:
        return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_read_streams_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    data = await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert data == b"chart"
    assert seen[0].url.path == "/v1/files/file_01abc/content"
    assert seen[0].headers["x-api-key"] == "sk-test"


@pytest.mark.asyncio
async def test_an_instances_own_client_args_reach_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # A proxy an instance is configured to dispatch through has to serve its
    # file reads too, so the settings that reach it are the same ones.
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    client = _client(client_args={"default_headers": {"x-proxy-token": "let-me-in"}})
    await _read_all(client.read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert seen[0].headers["x-proxy-token"] == "let-me-in"


@pytest.mark.asyncio
async def test_an_instance_cannot_widen_the_budget_a_read_runs_under(monkeypatch: pytest.MonkeyPatch) -> None:
    # The copy shares one deadline across every file, so these three stay Otari's.
    seen = _serving(monkeypatch, lambda request: httpx.Response(500))
    client = _client(client_args={"timeout": httpx.Timeout(600.0), "max_retries": 5})

    with pytest.raises(ProviderFileUnavailableError):
        await _read_all(client.read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert len(seen) == 1
    assert cast("Any", client._llm).client.timeout == _TIMEOUT


@pytest.mark.asyncio
async def test_a_download_that_redirects_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The connection is handed to the provider SDK, which would otherwise have
    # built one of its own that follows a redirect.
    def _redirects(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.example":
            return httpx.Response(200, content=b"chart")
        return httpx.Response(302, headers={"location": "https://cdn.example/blob"})

    _serving(monkeypatch, _redirects)

    data = await _read_all(_client().read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert data == b"chart"


@pytest.mark.asyncio
async def test_a_configured_api_base_is_where_the_download_goes(monkeypatch: pytest.MonkeyPatch) -> None:
    # An instance's api_base names the server, not a versioned prefix.
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, content=b"chart"))

    client = _client(api_base="https://anthropic.internal")
    await _read_all(client.read(ProviderFile(file_id="file_01abc"), budget_bytes=5))

    assert str(seen[0].url) == "https://anthropic.internal/v1/files/file_01abc/content"


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
    # The reason reaches the caller, which is what an operator reads in the log.
    with pytest.raises(ProviderFileUnavailableError, match="names no container"):
        await _read_all(_client("openai").read(ProviderFile(file_id="cfile_1"), budget_bytes=5))


@pytest.mark.asyncio
async def test_get_filename_reads_anthropic_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, json=_metadata()))

    assert await _client().get_filename("file_01abc") == "bar_plot.png"
    assert seen[0].url.path == "/v1/files/file_01abc"


@pytest.mark.asyncio
async def test_get_filename_is_none_when_the_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _serving(monkeypatch, lambda request: httpx.Response(404))

    assert await _client().get_filename("file_01abc") is None


@pytest.mark.asyncio
async def test_get_filename_is_none_when_the_metadata_is_not_an_object(monkeypatch: pytest.MonkeyPatch) -> None:
    # A body no reader expects costs the caller the name, never the file it holds.
    _serving(monkeypatch, lambda request: httpx.Response(200, json=["bar_plot.png"]))

    assert await _client().get_filename("file_01abc") is None


@pytest.mark.asyncio
async def test_get_filename_asks_nothing_of_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAI's citation already names the file, so there is no metadata call to make.
    seen = _serving(monkeypatch, lambda request: httpx.Response(200, json=_metadata()))

    assert await _client("openai").get_filename("cfile_1") is None
    assert seen == []


def test_for_run_needs_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    config: Any = SimpleNamespace()
    monkeypatch.setattr("gateway.services.files.provider_files.get_provider_kwargs", lambda *args, **kwargs: {})

    with pytest.raises(LookupError):
        ProviderFileClient.for_run(config, provider="anthropic", provider_instance="anthropic", workspace_id=None)
