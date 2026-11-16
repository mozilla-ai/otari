"""A credential with non-ASCII characters is refused with 401, never a 500.

``secrets.compare_digest`` raises TypeError on a non-ASCII str, and the dashboard's masked key display
(``otk_…••••…``) is exactly what a user copies off the screen by mistake.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api import deps
from gateway.api.deps import MASKED_KEY_DETAIL
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app

MASKED_KEY = "otk_v1_us_abc••••••••xyz"
EMOJI_KEY = "otk_v1_us_abc🔑xyz"

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


@pytest.mark.parametrize("route", ROUTES)
def test_masked_key_is_refused_with_copy_hint(client: TestClient, route: str) -> None:
    for headers in _header_forms(MASKED_KEY):
        response = client.get(route, headers=headers)
        assert response.status_code == 401, response.text
        assert response.json()["detail"] == MASKED_KEY_DETAIL


@pytest.mark.parametrize("route", ROUTES)
def test_emoji_key_is_an_ordinary_invalid_key(client: TestClient, route: str) -> None:
    for headers in _header_forms(EMOJI_KEY):
        response = client.get(route, headers=headers)
        assert response.status_code == 401, response.text
        assert "Invalid" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [MASKED_KEY, EMOJI_KEY])
async def test_is_valid_master_key_answers_false_for_non_ascii(token: str) -> None:
    # The sign-in route passes a JSON body's master key here without going through the header checks.
    assert await deps.is_valid_master_key(token, GatewayConfig(master_key="operator-set"), db=None) is False  # type: ignore[arg-type]
