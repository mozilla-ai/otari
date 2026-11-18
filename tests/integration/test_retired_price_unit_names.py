"""The plural unit names Otari retired are still read, for one release, and never written or served."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT

_RATE = {"model_key": "openai:gpt-4o", "input_price_per_million": 2.5, "output_price_per_million": 10.0}


@pytest.mark.parametrize(
    ("sent", "read"), [("tokens", "token"), ("requests", "request"), ("images", "image"), ("token", "token")]
)
def test_a_price_sent_in_a_retired_name_is_stored_and_served_in_the_current_one(
    client: TestClient, master_key_header: dict[str, str], sent: str, read: str
) -> None:
    posted = client.post(f"{API_ROOT}/pricing", json={**_RATE, "unit": sent}, headers=master_key_header)
    assert posted.status_code == 200, posted.text
    assert posted.json()["unit"] == read

    listed = client.get(f"{API_ROOT}/pricing", headers=master_key_header)
    assert listed.status_code == 200, listed.text
    assert [row["unit"] for row in listed.json() if row["model_key"] == "openai:gpt-4o"] == [read]


@pytest.mark.parametrize("sent", ["seconds", "Token", "token "])
def test_a_name_that_is_neither_current_nor_retired_is_refused(
    client: TestClient, master_key_header: dict[str, str], sent: str
) -> None:
    posted = client.post(f"{API_ROOT}/pricing", json={**_RATE, "unit": sent}, headers=master_key_header)

    assert posted.status_code == 422, posted.text


def test_a_row_an_older_writer_stored_in_a_retired_name_reads_in_the_current_one(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The rename migration rewrote every row; this covers one written beside it by code not yet upgraded."""
    posted = client.post(f"{API_ROOT}/pricing", json=_RATE, headers=master_key_header)
    assert posted.status_code == 200, posted.text
    with db_session_factory() as session:
        session.execute(text("UPDATE model_pricing SET unit = 'tokens' WHERE model_key = 'openai:gpt-4o'"))
        session.commit()

    listed = client.get(f"{API_ROOT}/pricing", headers=master_key_header)

    assert listed.status_code == 200, listed.text
    assert [row["unit"] for row in listed.json() if row["model_key"] == "openai:gpt-4o"] == ["token"]
