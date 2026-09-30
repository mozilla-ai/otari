"""Reading the control plane's answer for a workspace's code execution policy."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.models.tools import CodeExecutor
from gateway.services.tenancy.workspace_code_execution_policy_service import (
    ResolvedCodeExecutionPolicy,
    read_code_execution_policy,
)


def test_a_full_answer_is_read_field_by_field() -> None:
    policy = read_code_execution_policy(
        {
            "enabled": True,
            "default_purpose_hint": "Data analysis",
            "max_iterations": 4,
            "exec_timeout_s": 30,
            "tools": ["code_execution"],
            "executor": "otari",
        }
    )
    assert policy == ResolvedCodeExecutionPolicy(
        enabled=True,
        default_purpose_hint="Data analysis",
        max_iterations=4,
        exec_timeout_s=30,
        image=None,
        tools=frozenset({"code_execution"}),
        executor=CodeExecutor.OTARI,
    )


def test_a_disabled_answer_carries_nothing_else() -> None:
    policy = read_code_execution_policy({"enabled": False})
    assert policy.enabled is False
    assert (policy.default_purpose_hint, policy.max_iterations, policy.exec_timeout_s) == (None, None, None)
    assert (policy.tools, policy.executor) == (None, None)


def test_null_reads_as_absent() -> None:
    policy = read_code_execution_policy(
        {
            "enabled": True,
            "default_purpose_hint": None,
            "max_iterations": None,
            "exec_timeout_s": None,
            "tools": None,
            "executor": None,
        }
    )
    assert policy == read_code_execution_policy({"enabled": True})


def test_a_blank_hint_reads_as_none() -> None:
    assert read_code_execution_policy({"enabled": True, "default_purpose_hint": "  "}).default_purpose_hint is None


def test_a_ceiling_above_this_gateways_own_is_read_as_sent() -> None:
    """Admission applies each ceiling with ``min``, so a larger one is harmless rather than malformed."""
    policy = read_code_execution_policy({"enabled": True, "max_iterations": 10_000, "exec_timeout_s": 10_000})
    assert (policy.max_iterations, policy.exec_timeout_s) == (10_000, 10_000)


def test_an_image_is_not_part_of_the_answer() -> None:
    assert read_code_execution_policy({"enabled": True, "image": "attacker/image:latest"}).image is None


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"enabled": "yes"},
        {"enabled": 1},
        {"enabled": True, "default_purpose_hint": 5},
        {"enabled": True, "max_iterations": 0},
        {"enabled": True, "max_iterations": -1},
        {"enabled": True, "max_iterations": True},
        {"enabled": True, "max_iterations": "4"},
        {"enabled": True, "max_iterations": 1.5},
        {"enabled": True, "exec_timeout_s": 0},
        {"enabled": True, "exec_timeout_s": "30"},
        {"enabled": True, "tools": "code_execution"},
        {"enabled": True, "tools": [1]},
        {"enabled": True, "tools": {"code_execution": True}},
        {"enabled": True, "executor": "sometimes"},
        {"enabled": True, "executor": 3},
    ],
)
def test_a_malformed_answer_is_refused(answer: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        read_code_execution_policy(answer)
