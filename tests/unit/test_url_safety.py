"""Unit tests for `gateway.services.url_safety`."""

import pytest

from gateway.services.url_safety import (
    UnsafeURLError,
    validate_mcp_url,
    validate_outbound_fetch_url,
)


@pytest.mark.asyncio
async def test_public_https_accepted() -> None:
    await validate_mcp_url("https://example.com/mcp", has_authorization_token=True)


@pytest.mark.asyncio
async def test_public_http_accepted_without_token() -> None:
    await validate_mcp_url("http://example.com/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_public_http_rejected_with_token() -> None:
    with pytest.raises(UnsafeURLError, match="https"):
        await validate_mcp_url("http://example.com/mcp", has_authorization_token=True)


@pytest.mark.asyncio
async def test_loopback_allowed_by_default() -> None:
    await validate_mcp_url("http://127.0.0.1:9201/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_loopback_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_MCP_ALLOW_LOOPBACK", "false")
    with pytest.raises(UnsafeURLError, match="loopback"):
        await validate_mcp_url("http://127.0.0.1/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_rfc1918_rejected() -> None:
    for ip in ("10.0.0.5", "172.16.5.5", "192.168.1.1"):
        with pytest.raises(UnsafeURLError, match="private"):
            await validate_mcp_url(f"https://{ip}/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        await validate_mcp_url("https://169.254.169.254/latest/", has_authorization_token=False)


@pytest.mark.asyncio
async def test_ipv6_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        await validate_mcp_url("https://[fe80::1]/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_non_http_scheme_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="http or https"):
        await validate_mcp_url("ftp://example.com/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_no_host_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="hostname"):
        await validate_mcp_url("https:///mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_private_override_allows_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", "true")
    await validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_mcp_private_override_reads_otari_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # After promoting these gates to GatewayConfig, the SSRF read path still
    # consults otari_env() directly (the functions have no config in scope), so
    # the canonical OTARI_ prefix must keep toggling the gate.
    monkeypatch.delenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", raising=False)
    monkeypatch.setenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", "true")
    await validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_web_search_private_override_reads_otari_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same gate, web-search fetch path: rejected by default, allowed by the env
    # override that the promoted web_search_allow_private_hosts field mirrors.
    monkeypatch.delenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", raising=False)
    monkeypatch.delenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", raising=False)
    with pytest.raises(UnsafeURLError, match="private"):
        await validate_outbound_fetch_url("https://10.0.0.5/page")

    monkeypatch.setenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    await validate_outbound_fetch_url("https://10.0.0.5/page")


@pytest.mark.asyncio
async def test_unresolvable_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hostnames that fail to resolve are rejected (DNS-rebinding TOCTOU).

    A name that doesn't resolve at validation time could resolve to an internal
    address at fetch time. Operators that genuinely want this behaviour opt in
    via OTARI_MCP_ALLOW_PRIVATE_HOSTS.
    """
    from gateway.services import url_safety

    async def _empty(_host: str) -> list[object]:
        return []

    monkeypatch.setattr(url_safety, "_resolve_all_async", _empty)
    with pytest.raises(UnsafeURLError, match="could not be resolved"):
        await validate_mcp_url("https://does-not-exist.invalid/mcp", has_authorization_token=False)


@pytest.mark.asyncio
async def test_unresolvable_host_allowed_with_private_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private-hosts opt-out also covers unresolvable hostnames."""
    from gateway.services import url_safety

    async def _empty(_host: str) -> list[object]:
        return []

    monkeypatch.setenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setattr(url_safety, "_resolve_all_async", _empty)
    await validate_mcp_url("https://does-not-exist.invalid/mcp", has_authorization_token=False)
