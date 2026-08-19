"""Route-level auth for a key that does not have the ``gw-`` shape (issue #646).

Otari and otari-ai both store an unsalted SHA-256 hex digest of the whole
presented key, so a migrated ``tk_`` key's hash lands on an ``api_keys`` row
unchanged. The format check that used to run inside ``hash_key`` rejected such a
key with a 401 before the lookup ever happened; auth now depends on the row
alone, which is what makes a cutover re-issue unnecessary.
"""

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from gateway.auth.models import hash_key
from gateway.core.config import API_KEY_HEADER
from gateway.models.entities import APIKey

# The shape otari-ai mints: it fails both the ``gw-``/``gw_`` prefix check and the
# ``gw[-_][A-Za-z0-9_-]+`` charset check the old validator applied.
MIGRATED_KEY = "tk_live.migrated-platform-key-0123456789abcdefghij"


def test_migrated_key_authenticates_when_its_hash_is_on_a_row(
    client: TestClient,
    db_session: Session,
    api_key_obj: dict[str, Any],
) -> None:
    """Re-point an existing key row at a migrated key's hash, as the migration would."""
    db_session.execute(
        update(APIKey).where(APIKey.id == api_key_obj["id"]).values(key_hash=hash_key(MIGRATED_KEY)),
    )
    db_session.commit()

    response = client.get("/v1/models", headers={API_KEY_HEADER: MIGRATED_KEY})

    assert response.status_code == 200


def test_unknown_migrated_shape_key_gets_the_ordinary_invalid_key_401(client: TestClient) -> None:
    response = client.get("/v1/models", headers={API_KEY_HEADER: MIGRATED_KEY})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
