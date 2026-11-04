"""Unit tests for the read timeout every Bedrock client is built with."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.config import Config

from gateway.services.provider_kwargs import BEDROCK_READ_TIMEOUT_SECONDS, with_bedrock_read_timeout


def test_the_read_timeout_is_added_when_no_config_is_given() -> None:
    client_args = with_bedrock_read_timeout({"region_name": "us-east-1"})

    assert client_args["region_name"] == "us-east-1"
    assert client_args["config"].read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


def test_an_operator_config_keeps_its_own_settings_and_gains_the_read_timeout() -> None:
    operator = Config(retries={"max_attempts": 2})

    merged = with_bedrock_read_timeout({"config": operator})["config"]

    assert merged.retries == {"max_attempts": 2}
    assert merged.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS


@pytest.mark.parametrize("operator", [Config(read_timeout=30), {"read_timeout": 30}])
def test_an_operator_read_timeout_wins(operator: Config | dict[str, Any]) -> None:
    merged = with_bedrock_read_timeout({"config": operator})["config"]

    assert merged.read_timeout == 30
    assert isinstance(merged, Config)


def test_a_prebuilt_client_and_the_callers_mapping_are_left_alone() -> None:
    prebuilt = object()
    original: dict[str, Any] = {"client": prebuilt}

    assert with_bedrock_read_timeout(original) == {"client": prebuilt}
    assert with_bedrock_read_timeout({"region_name": "us-east-1"}) is not original
    no_config: dict[str, Any] = {"region_name": "us-east-1"}
    with_bedrock_read_timeout(no_config)
    assert "config" not in no_config


def test_a_config_botocore_cannot_take_is_refused() -> None:
    with pytest.raises(ValueError, match="botocore Config or a mapping"):
        with_bedrock_read_timeout({"config": 120})
