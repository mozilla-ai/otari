"""A credential with non-ASCII characters is refused with 401, never a 500.

``secrets.compare_digest`` raises TypeError on a non-ASCII str, and a key copied off a masked display
(``otk_…••••…``) is exactly that.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api import deps
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app

NON_ASCII_KEYS = pytest.mark.parametrize(
    "token", ["otk_v1_us_abc••••••••xyz", "otk_v1_us_abc🔑xyz"], ids=["bullets", "emoji"]
)

# A management route (master key only) and a catalog read (API key or master key).
ROUTES = (f"{API_ROOT}/hosted-providers", f"{API_ROOT}/models")


def _header_forms(token: str) -> list[dict[bytes, bytes]]:
    # Sent as UTF-8 bytes, as curl does; httpx refuses a non-ASCII str header value.
    return [
        {b"Authorization": f"Bearer {token}".encode()},
        {b"Otari-Key": token.encode()},
    ]


@pytest.fixture(params=["operator-set", None], ids=["master-key", "generated-master-key"])
def client(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("OTARI_MASTER_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_MASTER_KEY", raising=False)
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'auth.db'}", master_key=request.param, require_pricing=False
    )
    with TestClient(create_app(config), raise_server_exceptions=False) as test_client:
        yield test_client


@NON_ASCII_KEYS
@pytest.mark.parametrize("route", ROUTES)
def test_non_ascii_key_is_an_ordinary_invalid_key(client: TestClient, route: str, token: str) -> None:
    for headers in _header_forms(token):
        response = client.get(route, headers=headers)
        assert response.status_code == 401, response.text
        assert "Invalid" in response.json()["detail"]


@pytest.mark.asyncio
@NON_ASCII_KEYS
async def test_is_valid_master_key_answers_false_for_non_ascii(token: str) -> None:
    # The sign-in route passes a JSON body's master key here without going through the header checks.
    assert await deps.is_valid_master_key(token, GatewayConfig(master_key="operator-set"), db=None) is False  # type: ignore[arg-type]


def test_is_master_key_compares_non_ascii_values() -> None:
    assert GatewayConfig(master_key="clé•").is_master_key("clé•")
    assert not GatewayConfig(master_key="operator-set").is_master_key("clé•")
    assert not GatewayConfig(master_key=None).is_master_key("clé•")
