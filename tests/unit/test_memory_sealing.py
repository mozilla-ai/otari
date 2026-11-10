"""Memory's sealing: a value opens only under its key, for the row and field it was sealed for."""

import os
import uuid

import pytest

from gateway.services.memory._sealing import (
    FORMAT,
    SealedValueError,
    associated_data,
    decode_wrapped,
    encode_wrapped,
    key_context,
    open_sealed,
    pack_embedding,
    seal,
    unpack_embedding,
)

WORKSPACE = uuid.uuid4()
MEMBER = uuid.uuid4()
ROW = uuid.uuid4()


def _aad(**changes: object) -> bytes:
    fields: dict[str, object] = {
        "workspace_id": WORKSPACE,
        "scope": "member",
        "owner_user_id": MEMBER,
        "record": "entry",
        "record_id": ROW,
        "field": "content",
    }
    fields.update(changes)
    return associated_data(**fields)  # type: ignore[arg-type]


def test_a_sealed_value_opens_under_its_key_for_its_row() -> None:
    key = os.urandom(32)
    sealed = seal(key, b"the deploy needs VPN", _aad())

    assert sealed.startswith(f"{FORMAT}.")
    assert "VPN" not in sealed
    assert open_sealed(key, sealed, _aad()) == b"the deploy needs VPN"


def test_two_seals_of_the_same_text_differ() -> None:
    key = os.urandom(32)

    assert seal(key, b"same", _aad()) != seal(key, b"same", _aad())


@pytest.mark.parametrize(
    "changes",
    [
        {"workspace_id": uuid.uuid4()},
        {"scope": "workspace", "owner_user_id": None},
        {"owner_user_id": uuid.uuid4()},
        {"record": "proposal"},
        {"record_id": uuid.uuid4()},
        {"field": "embedding"},
    ],
    ids=["another-workspace", "another-scope", "another-member", "another-table", "another-row", "another-field"],
)
def test_a_sealed_value_moved_anywhere_else_does_not_open(changes: dict[str, object]) -> None:
    key = os.urandom(32)
    sealed = seal(key, b"secret", _aad())

    with pytest.raises(SealedValueError):
        open_sealed(key, sealed, _aad(**changes))


def test_a_sealed_value_does_not_open_under_another_key() -> None:
    sealed = seal(os.urandom(32), b"secret", _aad())

    with pytest.raises(SealedValueError):
        open_sealed(os.urandom(32), sealed, _aad())


@pytest.mark.parametrize("value", ["", "mem1.", "otc1.abc", "mem1.!!!", "mem1.AAAA", "plain text"])
def test_a_malformed_value_is_refused(value: str) -> None:
    with pytest.raises(SealedValueError):
        open_sealed(os.urandom(32), value, _aad())


def test_a_tampered_value_is_refused() -> None:
    key = os.urandom(32)
    sealed = seal(key, b"secret", _aad())
    tampered = sealed[:-2] + ("A" if sealed[-2] != "A" else "B") + sealed[-1]

    with pytest.raises(SealedValueError):
        open_sealed(key, tampered, _aad())


def test_an_embedding_round_trips_as_float32() -> None:
    vector = [0.25, -1.5, 3.0, 0.0]

    assert unpack_embedding(pack_embedding(vector)) == tuple(vector)
    assert len(pack_embedding(vector)) == 4 * len(vector)


def test_bytes_that_are_not_float32_values_are_not_an_embedding() -> None:
    with pytest.raises(SealedValueError):
        unpack_embedding(b"\x00\x01\x02")


def test_a_data_key_is_bound_to_memory_and_its_workspace() -> None:
    assert key_context(WORKSPACE) == {"otari:purpose": "memory-content", "otari:workspace": str(WORKSPACE)}


def test_a_wrapped_key_round_trips_through_its_stored_form() -> None:
    wrapped = os.urandom(48)

    assert decode_wrapped(encode_wrapped(wrapped)) == wrapped
    with pytest.raises(SealedValueError):
        decode_wrapped("!!!")
