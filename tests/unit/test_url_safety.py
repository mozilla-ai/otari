"""Unit tests for `gateway.services.url_safety`."""

import pytest

from gateway.services.url_safety import UnsafeURLError, validate_mcp_url


def test_public_https_accepted() -> None:
    validate_mcp_url("https://example.com/mcp", has_authorization_token=True)


def test_public_http_accepted_without_token() -> None:
    validate_mcp_url("http://example.com/mcp", has_authorization_token=False)


def test_public_http_rejected_with_token() -> None:
    with pytest.raises(UnsafeURLError, match="https"):
        validate_mcp_url("http://example.com/mcp", has_authorization_token=True)


def test_loopback_allowed_by_default() -> None:
    validate_mcp_url("http://127.0.0.1:9201/mcp", has_authorization_token=False)


def test_loopback_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_LOOPBACK", "false")
    with pytest.raises(UnsafeURLError, match="loopback"):
        validate_mcp_url("http://127.0.0.1/mcp", has_authorization_token=False)


def test_rfc1918_rejected() -> None:
    for ip in ("10.0.0.5", "172.16.5.5", "192.168.1.1"):
        with pytest.raises(UnsafeURLError, match="private"):
            validate_mcp_url(f"https://{ip}/mcp", has_authorization_token=False)


def test_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        validate_mcp_url("https://169.254.169.254/latest/", has_authorization_token=False)


def test_ipv6_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        validate_mcp_url("https://[fe80::1]/mcp", has_authorization_token=False)


def test_non_http_scheme_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="http or https"):
        validate_mcp_url("ftp://example.com/mcp", has_authorization_token=False)


def test_no_host_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="hostname"):
        validate_mcp_url("https:///mcp", has_authorization_token=False)


def test_private_override_allows_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", "true")
    validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)
