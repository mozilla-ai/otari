"""The ``--shard`` option that splits a test run across CI jobs."""

import argparse

import pytest

from conftest import Shard, parse_shard

NODE_IDS = [f"tests/unit/test_example_{n}.py::test_case[{n}]" for n in range(500)]


@pytest.mark.parametrize("count", [1, 2, 4, 7])
def test_every_test_lands_in_exactly_one_shard(count: int) -> None:
    for nodeid in NODE_IDS:
        owners = [index for index in range(1, count + 1) if Shard(index=index, count=count).includes(nodeid)]
        assert len(owners) == 1, nodeid


def test_a_shard_is_parsed_from_index_and_count() -> None:
    assert parse_shard("2/4") == Shard(index=2, count=4)


@pytest.mark.parametrize("value", ["0/4", "5/4", "1/0", "4", "a/4", "-1/4", "1/4/2", "１/4"])
def test_a_shard_outside_its_count_or_malformed_is_refused(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shard(value)
