"""The fingerprint survives every hop between the key schemas.

``CreateKeyResponse`` is built by splatting a ``KeyInfo`` dump, and pydantic drops
keys the target model does not declare. A field added to one model and not the
other therefore compiles, typechecks and passes every other test while silently
vanishing from the create and rotate responses, which is the failure this file
exists to make loud.
"""

import uuid
from datetime import UTC, datetime

from gateway.api.routes.keys import CreateKeyResponse, KeyInfo
from gateway.models.api_keys import APIKey
from gateway.services.tenancy.workspace_activation_service import ActivationApiKeyPublic


def _key(**overrides: object) -> APIKey:
    fields: dict[str, object] = {
        "id": "key-1",
        "workspace_id": uuid.uuid4(),
        "key_hash": "hash-1",
        "key_prefix": "gw-abcdefg",
        "key_suffix": "wxyz",
        # Applied at flush in real use; these rows are never flushed.
        "created_at": datetime(2026, 9, 14, tzinfo=UTC),
    }
    fields.update(overrides)
    return APIKey(**fields)


def test_the_create_response_keeps_both_fingerprint_halves() -> None:
    key_info = KeyInfo.from_model(_key())

    response = CreateKeyResponse(**key_info.model_dump(exclude={"last_used_at"}), key="gw-secret")

    assert response.key_prefix == "gw-abcdefg"
    assert response.key_suffix == "wxyz"


def test_a_key_minted_before_either_half_existed_reads_back_as_null() -> None:
    """Most rows on the day this ships: neither half is recoverable from the hash."""
    key_info = KeyInfo.from_model(_key(key_prefix=None, key_suffix=None))

    assert key_info.key_prefix is None
    assert key_info.key_suffix is None
    assert key_info.model_dump()["key_suffix"] is None


def test_a_key_minted_between_the_two_columns_reads_back_prefix_only() -> None:
    """The middle of the three display states: a prefix, and no suffix to pair it with."""
    key_info = KeyInfo.from_model(_key(key_suffix=None))

    assert key_info.key_prefix == "gw-abcdefg"
    assert key_info.key_suffix is None


def test_the_activation_key_shape_carries_the_suffix() -> None:
    """Its own schema, built from the row rather than the plaintext, so an unset
    column here would return null with nothing raising anywhere."""
    public = ActivationApiKeyPublic(
        key="gw-secret", key_id="key-1", key_prefix="gw-abcdefg", key_suffix="wxyz", key_name="setup"
    )

    assert public.key_suffix == "wxyz"


def test_the_row_serializer_carries_the_suffix() -> None:
    assert _key().to_dict()["key_suffix"] == "wxyz"
