"""``mcp_private_hosts``: the operator's exact list of private hosts an MCP server URL may name.

A listed host (and port, where the entry names one) may resolve to a loopback or
private address, and is still dialed through the pinned transport at the address
checked. Link-local, metadata and unspecified addresses are refused even for a
listed host, and everything not listed keeps the public-only rule. Guardrails
never read the list.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from typing import Any

import httpx
import httpx2
import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import parse_mcp_private_hosts
from gateway.models.guardrails import GuardrailConfig
from gateway.services import guardrails as guardrails_module
from gateway.services import mcp_stateless, url_safety, web_retrieval_network
from gateway.services.url_safety import PrivateHosts, UnsafeURLError, mcp_private_hosts, validate_mcp_url
from gateway.services.web_retrieval_network import PinnedAsyncHTTPTransport

pytestmark = pytest.mark.asyncio


def _listed(value: str) -> PrivateHosts:
    return PrivateHosts(parse_mcp_private_hosts(value))


class _Socket:
    def __init__(self) -> None:
        self.dialed: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.dialed.append(request)
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def socket(monkeypatch: pytest.MonkeyPatch) -> _Socket:
    fake = _Socket()
    monkeypatch.setattr(PinnedAsyncHTTPTransport, "_build_transport", lambda self: httpx.MockTransport(fake.handle))
    return fake


@pytest.fixture
def resolves_to(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Per-host answers for both the save-time and the connect-time resolver."""
    answers: dict[str, str] = {}

    async def save_time(host: str) -> list[object]:
        return [ipaddress.ip_address(answers[host])]

    async def connect_time(self: object, host: str, port: int) -> Sequence[Any]:
        return (ipaddress.ip_address(answers[host]),)

    monkeypatch.setattr(url_safety, "_resolve_all_async", save_time)
    monkeypatch.setattr(web_retrieval_network.SystemAddressResolver, "resolve", connect_time)
    return answers


# --------------------------------------------------------------------------- #
# The setting
# --------------------------------------------------------------------------- #


async def test_entries_are_exact_hosts_with_an_optional_port() -> None:
    assert parse_mcp_private_hosts(" MCP.Internal. , localhost:8000, [::1]:9000, fd00::1, 10.0.0.5 ") == {
        ("mcp.internal", None),
        ("localhost", 8000),
        ("::1", 9000),
        ("fd00::1", None),
        ("10.0.0.5", None),
    }
    assert parse_mcp_private_hosts(None) == frozenset()


@pytest.mark.parametrize("entry", ["*.corp.example", ".corp.example", "10.0.0.0/8", "http://mcp", "mcp:0", "mcp:x"])
async def test_a_range_or_a_url_is_refused_at_load(entry: str) -> None:
    with pytest.raises(ValidationError, match="mcp_private_hosts"):
        GatewayConfig(mcp_private_hosts=entry)


async def test_a_config_file_list_is_read_as_the_same_setting() -> None:
    as_written: Any = ["mcp", "localhost:8000"]

    assert GatewayConfig(mcp_private_hosts=as_written).mcp_private_hosts == "mcp,localhost:8000"


async def test_it_is_empty_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_MCP_PRIVATE_HOSTS", raising=False)

    assert GatewayConfig().mcp_private_hosts is None
    assert mcp_private_hosts().entries == frozenset()


# --------------------------------------------------------------------------- #
# The save-time check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("listed", "url", "address"),
    [
        ("mcp", "http://mcp:8000/mcp", "172.18.0.4"),
        ("localhost", "http://localhost:8000/mcp", "127.0.0.1"),
        ("mcp.internal:8000", "http://mcp.internal:8000/mcp", "10.0.0.5"),
        ("127.0.0.1", "http://127.0.0.1:8123/mcp", "127.0.0.1"),
        ("[::1]:8443", "https://[::1]:8443/mcp", "::1"),
        ("tail.example", "https://tail.example/mcp", "100.64.0.7"),
    ],
)
async def test_a_listed_host_on_a_private_address_is_accepted(
    listed: str, url: str, address: str, resolves_to: dict[str, str]
) -> None:
    resolves_to[url.split("//")[1].split("/")[0].rsplit(":", 1)[0].strip("[]")] = address

    await validate_mcp_url(url, has_authorization_token=False, private_hosts=_listed(listed))


@pytest.mark.parametrize(
    "url",
    ["http://other:8000/mcp", "http://127.0.0.1:8000/mcp", "http://127.1:8000/mcp", "http://mcp.internal.evil/mcp"],
)
async def test_an_unlisted_loopback_or_local_host_is_refused(url: str, resolves_to: dict[str, str]) -> None:
    resolves_to.update({"other": "127.0.0.1", "mcp.internal.evil": "127.0.0.1"})

    with pytest.raises(UnsafeURLError):
        await validate_mcp_url(url, has_authorization_token=False, private_hosts=_listed("localhost,mcp.internal"))


@pytest.mark.parametrize(
    "address", ["169.254.169.254", "fe80::1", "fd00:ec2::254", "100.100.100.200", "::ffff:169.254.169.254", "0.0.0.0"]
)
async def test_a_listed_host_resolving_to_a_metadata_or_link_local_address_is_refused(
    address: str, resolves_to: dict[str, str]
) -> None:
    resolves_to["metadata.internal"] = address

    with pytest.raises(UnsafeURLError, match="refused even for a host in mcp_private_hosts"):
        await validate_mcp_url(
            "http://metadata.internal/latest", has_authorization_token=False, private_hosts=_listed("metadata.internal")
        )


async def test_a_listed_link_local_literal_is_still_refused() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        await validate_mcp_url(
            "http://169.254.169.254/latest", has_authorization_token=False, private_hosts=_listed("169.254.169.254")
        )


@pytest.mark.parametrize(
    ("url", "accepted"),
    [
        ("http://mcp.internal:8000/mcp", True),
        ("http://mcp.internal:9000/mcp", False),
        ("http://mcp.internal/mcp", False),
        ("https://mcp.internal/mcp", False),
    ],
)
async def test_an_entrys_port_is_enforced(url: str, accepted: bool, resolves_to: dict[str, str]) -> None:
    resolves_to["mcp.internal"] = "10.0.0.5"
    check = validate_mcp_url(url, has_authorization_token=False, private_hosts=_listed("mcp.internal:8000"))

    if accepted:
        await check
    else:
        with pytest.raises(UnsafeURLError):
            await check


async def test_an_entry_without_a_port_admits_any_port(resolves_to: dict[str, str]) -> None:
    resolves_to["mcp.internal"] = "10.0.0.5"

    for port in (80, 8000, 9443):
        await validate_mcp_url(
            f"http://mcp.internal:{port}/mcp", has_authorization_token=False, private_hosts=_listed("mcp.internal")
        )


async def test_a_guardrail_endpoint_never_reads_the_list(
    monkeypatch: pytest.MonkeyPatch, resolves_to: dict[str, str]
) -> None:
    """The check's default is the public-only rule; only MCP server call sites pass the list."""
    monkeypatch.setenv("OTARI_MCP_PRIVATE_HOSTS", "guard.internal")
    resolves_to["guard.internal"] = "10.0.0.5"

    with pytest.raises(UnsafeURLError, match="public address"):
        await validate_mcp_url("https://guard.internal/validate", has_authorization_token=False)
    with pytest.raises(UnsafeURLError):
        await guardrails_module.check_guardrail_urls([GuardrailConfig(profile="pii", url="https://guard.internal")])


# --------------------------------------------------------------------------- #
# The connection
# --------------------------------------------------------------------------- #


async def _connect(url: str) -> int:
    async with mcp_stateless.build_http_client_factory()(None) as client:
        return (await client.get(url)).status_code


async def test_a_listed_host_on_loopback_is_dialed_at_the_checked_address(
    monkeypatch: pytest.MonkeyPatch, socket: _Socket, resolves_to: dict[str, str]
) -> None:
    monkeypatch.setenv("OTARI_MCP_PRIVATE_HOSTS", "mcp:8000")
    resolves_to["mcp"] = "127.0.0.1"

    assert await _connect("http://mcp:8000/mcp") == 200

    [sent] = socket.dialed
    assert (sent.url.host, sent.headers["host"]) == ("127.0.0.1", "mcp:8000")


@pytest.mark.parametrize(
    ("url", "address"),
    [
        ("http://other:8000/mcp", "127.0.0.1"),
        ("http://mcp:9000/mcp", "127.0.0.1"),
        ("http://mcp:8000/mcp", "169.254.169.254"),
        ("http://mcp:8000/mcp", "fd00:ec2::254"),
    ],
)
async def test_an_unlisted_host_a_wrong_port_or_a_metadata_answer_is_refused_at_connect(
    monkeypatch: pytest.MonkeyPatch, socket: _Socket, resolves_to: dict[str, str], url: str, address: str
) -> None:
    monkeypatch.setenv("OTARI_MCP_PRIVATE_HOSTS", "mcp:8000")
    resolves_to.update({"mcp": address, "other": address})

    with pytest.raises(httpx2.TransportError):
        await _connect(url)

    assert socket.dialed == []


async def test_a_guardrail_url_never_reads_the_list_at_connect(
    monkeypatch: pytest.MonkeyPatch, socket: _Socket, resolves_to: dict[str, str]
) -> None:
    monkeypatch.setenv("OTARI_MCP_PRIVATE_HOSTS", "guard.internal")
    resolves_to["guard.internal"] = "10.0.0.5"

    async with guardrails_module._tenant_endpoint_client() as client:
        with pytest.raises(httpx.TransportError):
            await client.post("https://guard.internal/validate", json={})

    assert socket.dialed == []


@pytest.mark.parametrize(
    "address", ["64:ff9b::a9fe:a9fe", "2002:a9fe:a9fe::", "64:ff9b::6464:64c8", "::ffff:169.254.169.254"]
)
async def test_a_metadata_address_inside_a_nat64_or_6to4_one_is_refused_for_a_listed_host(
    address: str, resolves_to: dict[str, str]
) -> None:
    """A translator on the path would deliver it to the IPv4 address it carries."""
    resolves_to["mcp.internal"] = address

    with pytest.raises(UnsafeURLError, match="refused even for a host in mcp_private_hosts"):
        await validate_mcp_url(
            "http://mcp.internal/mcp", has_authorization_token=False, private_hosts=_listed("mcp.internal")
        )


async def test_a_private_address_inside_a_nat64_one_is_still_admitted_for_a_listed_host(
    resolves_to: dict[str, str],
) -> None:
    resolves_to["mcp.internal"] = "64:ff9b::a00:5"

    await validate_mcp_url(
        "http://mcp.internal/mcp", has_authorization_token=False, private_hosts=_listed("mcp.internal")
    )


@pytest.mark.parametrize("listed", ["localhost", "mcp.internal"])
async def test_the_check_and_the_connection_agree_on_a_trailing_dot(
    monkeypatch: pytest.MonkeyPatch, socket: _Socket, resolves_to: dict[str, str], listed: str
) -> None:
    """The connection dials the name without the dot, so a URL written with one is refused by both."""
    monkeypatch.setenv("OTARI_MCP_PRIVATE_HOSTS", listed)
    resolves_to.update({"localhost": "127.0.0.1", "localhost.": "127.0.0.1"})
    url = "http://localhost.:8000/mcp"

    with pytest.raises(UnsafeURLError, match="ends with a dot|local name"):
        await validate_mcp_url(url, has_authorization_token=False, private_hosts=mcp_private_hosts())
    with pytest.raises(httpx2.TransportError):
        await _connect(url)
    assert socket.dialed == []


async def test_a_retired_mcp_setting_still_set_is_named_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from gateway.core.config import load_config
    from gateway.log_config import logger as gateway_logger

    config_file = tmp_path / "gateway.yml"
    config_file.write_text("mcp_allow_private_hosts: true\n", encoding="utf-8")
    monkeypatch.setenv("OTARI_MCP_ALLOW_LOOPBACK", "true")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        load_config(str(config_file))
    finally:
        gateway_logger.removeHandler(caplog.handler)

    warning = next(record.getMessage() for record in caplog.records if "mcp_private_hosts" in record.getMessage())
    assert "mcp_allow_private_hosts" in warning
    assert "OTARI_MCP_ALLOW_LOOPBACK" in warning
