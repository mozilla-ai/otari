"""Provider-aware gateways must not accept authorities that ignore provider selection."""

import httpx
import pytest

from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FilePage, FilesError


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, "1", "2"])
async def test_authority_version_is_checked_on_success(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Otari-Files-Protocol"] == "2"
        return httpx.Response(200, headers={"X-Otari-Files-Protocol": version} if version else {}, json={"data": []})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))
    client = PlatformFilesClient("https://authority.test", "gateway", "user")
    if version == "2":
        assert (await client.post("list", {"provider": "openai"}, FilePage)).data == []
    else:
        with pytest.raises(FilesError, match="protocol"):
            await client.post("list", {"provider": "openai"}, FilePage)
