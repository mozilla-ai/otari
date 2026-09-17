"""The ``--shard`` option that splits a test run across CI jobs."""

import argparse
from pathlib import Path

import pytest
import yaml

from conftest import Shard, parse_shard

NODE_IDS = [f"tests/unit/test_example_{n}.py::test_case[{n}]" for n in range(500)]
TESTS_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "otari-tests.yml"


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


def test_each_sharded_ci_job_lists_every_shard() -> None:
    jobs = yaml.safe_load(TESTS_WORKFLOW.read_text())["jobs"]
    matrices = {name: job["strategy"]["matrix"]["shard"] for name, job in jobs.items() if "strategy" in job}

    assert matrices, "no sharded job found"
    for name, values in matrices.items():
        shards = [parse_shard(value) for value in values]
        count = shards[0].count
        assert sorted(shards, key=lambda shard: shard.index) == [
            Shard(index=index, count=count) for index in range(1, count + 1)
        ], name
