"""Provider file byte counters use database types that hold configured quotas."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import BigInteger, create_engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

from gateway.models import provider_files  # noqa: F401


def test_provider_file_byte_columns_use_big_integers() -> None:
    bindings = SQLModel.metadata.tables["provider_file_bindings"]
    outputs = SQLModel.metadata.tables["provider_file_output_operations"]

    assert isinstance(bindings.c.size_bytes.type, BigInteger)
    assert isinstance(outputs.c.reserved_bytes.type, BigInteger)


@pytest.mark.parametrize(
    ("table_name", "column", "valid"),
    [
        ("provider_account_generations", "credential_source", "organization_key"),
        ("provider_account_generations", "status", "active"),
        ("provider_file_bindings", "state", "pending_upload"),
        ("provider_file_output_operations", "state", "active"),
    ],
)
def test_database_rejects_unknown_lifecycle_values(table_name: str, column: str, valid: str) -> None:
    engine = create_engine("sqlite://")
    names = ("provider_account_generations", "provider_file_output_operations", "provider_file_bindings")
    tables = [SQLModel.metadata.tables[name] for name in names]
    SQLModel.metadata.create_all(engine, tables=tables)
    now = datetime.now(UTC)
    common: dict[str, Any] = {
        "id": uuid.uuid4(),
        "created_at": now,
        "organization_id": uuid.uuid4(),
        "workspace_id": uuid.uuid4(),
        "user_id": "uploader",
        "provider_account_generation_id": uuid.uuid4(),
        "initiating_gateway_id": "gateway",
        "cleanup_token_hash": "token",
        "deadline": now,
        "expires_at": now,
        "operation_deadline": now,
        "reserved_files": 1,
        "reserved_bytes": 10,
        "credential_source": "organization_key",
        "credential_ref": "key",
        "generation": 1,
        "provider": "anthropic",
        "status": "active",
        "state": "active",
        "size_bytes": 10,
        "downloadable": False,
        "provider_outcome_unknown": False,
        "cleanup_attempts": 0,
        "request_id": "request",
        "attempt_id": "attempt",
    }
    table = SQLModel.metadata.tables[table_name]
    values = {key: value for key, value in common.items() if key in table.c}
    values[column] = valid
    try:
        with engine.begin() as connection:
            connection.execute(table.insert().values(**values))
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(table.update().values(**{column: "typo"}))
    finally:
        engine.dispose()
