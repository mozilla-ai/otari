"""Provider file byte counters use database types that hold configured quotas."""

from sqlalchemy import BigInteger
from sqlmodel import SQLModel

from gateway.models import provider_files  # noqa: F401


def test_provider_file_byte_columns_use_big_integers() -> None:
    bindings = SQLModel.metadata.tables["provider_file_bindings"]
    outputs = SQLModel.metadata.tables["provider_file_output_operations"]

    assert isinstance(bindings.c.size_bytes.type, BigInteger)
    assert isinstance(outputs.c.reserved_bytes.type, BigInteger)
