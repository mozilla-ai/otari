"""A key's timestamps leave the gateway with their UTC offset.

SQLite hands back the key table's timestamps naive, and a browser reads an
offset-less timestamp in its own zone, so a dashboard west of UTC would count a
key that expired hours ago as still usable. The server's own expiry check reads
a naive value as UTC; the response has to say so.
"""

import uuid
from datetime import UTC, datetime, timedelta, timezone

from gateway.api.routes.keys import KeyInfo
from gateway.models.api_keys import APIKey
from gateway.types.key_scope import KeyScopeSet


def _key(**overrides: object) -> APIKey:
    fields: dict[str, object] = {
        "id": "key-1",
        "workspace_id": uuid.uuid4(),
        "key_hash": "hash-1",
        "scopes": KeyScopeSet.read(["inference:run"]),
        "created_at": datetime(2026, 9, 14, tzinfo=UTC),
    }
    fields.update(overrides)
    return APIKey(**fields)


def test_a_naive_timestamp_reads_back_as_utc() -> None:
    """What SQLite returns: no offset on the row, an explicit one on the wire."""
    naive = datetime(2026, 11, 17, 12, 0)  # noqa: DTZ001 - the shape SQLite hands back
    key_info = KeyInfo.from_model(_key(created_at=naive, last_used_at=naive, expires_at=naive))

    assert key_info.created_at == "2026-11-17T12:00:00+00:00"
    assert key_info.last_used_at == "2026-11-17T12:00:00+00:00"
    assert key_info.expires_at == "2026-11-17T12:00:00+00:00"


def test_an_aware_timestamp_keeps_its_instant() -> None:
    """What PostgreSQL returns passes through unchanged."""
    aware = datetime(2026, 11, 17, 4, 0, tzinfo=timezone(timedelta(hours=-8)))
    key_info = KeyInfo.from_model(_key(expires_at=aware))

    assert key_info.expires_at is not None
    assert datetime.fromisoformat(key_info.expires_at) == datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def test_unset_timestamps_stay_null() -> None:
    key_info = KeyInfo.from_model(_key())

    assert key_info.last_used_at is None
    assert key_info.expires_at is None
