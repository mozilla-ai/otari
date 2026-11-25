"""Which priced model a harness's own name for a model stands for."""

import pytest

from gateway.services.routing._subagent_baseline import baseline_key, model_key


@pytest.mark.parametrize(
    ("harness", "model", "expected"),
    [
        ("claude-code", "claude-opus-5", "anthropic:claude-opus-5"),
        ("claude-code", "claude-opus-5[1m]", "anthropic:claude-opus-5"),
        ("claude-code", "anthropic:claude-sonnet-4-6", "anthropic:claude-sonnet-4-6"),
        ("claude-code", "anthropic:claude-sonnet-4-6[1m]", "anthropic:claude-sonnet-4-6"),
        ("claude-code", "opus", None),
        ("claude-code", "  ", None),
        ("pi", "otari/anthropic:claude-sonnet-5", "anthropic:claude-sonnet-5"),
        ("pi", "openai:gpt-5-mini", "openai:gpt-5-mini"),
        ("pi", "otari/openai:gpt-5:high", "openai:gpt-5"),
        # A Bedrock id ends in a version, which is not a thinking level.
        ("pi", "bedrock:anthropic.claude-haiku-5-v1:0", "bedrock:anthropic.claude-haiku-5-v1:0"),
        # Stripping the level would leave no model.
        ("pi", "openai:high", "openai:high"),
        ("pi", "unknown", None),
        ("pi", "gpt-5", None),
    ],
)
def test_a_harness_model_name_reads_as_a_priced_key(harness: str, model: str, expected: str | None) -> None:
    assert model_key(harness, model) == expected


def test_a_full_id_is_its_own_baseline() -> None:
    assert baseline_key("claude-code", "claude-opus-5", parent_model="claude-sonnet-5", ran_on=[]) == (
        "anthropic:claude-opus-5"
    )


def test_a_provider_qualified_id_is_its_own_baseline() -> None:
    key = baseline_key("claude-code", "anthropic:claude-opus-5", parent_model="claude-sonnet-5", ran_on=[])
    assert key == "anthropic:claude-opus-5"


def test_an_alias_reads_as_the_parents_model_of_that_family() -> None:
    key = baseline_key("claude-code", "opus", parent_model="claude-opus-5[1m]", ran_on=["claude-haiku-5"])
    assert key == "anthropic:claude-opus-5"


def test_an_alias_reads_as_the_model_of_that_family_the_subagent_ran_on() -> None:
    key = baseline_key("claude-code", "Sonnet", parent_model="claude-opus-5", ran_on=["claude-sonnet-5"])
    assert key == "anthropic:claude-sonnet-5"


def test_an_alias_nothing_names_a_model_for_is_unresolved() -> None:
    assert baseline_key("claude-code", "opus", parent_model="claude-sonnet-5", ran_on=["claude-haiku-5"]) is None
    assert baseline_key("claude-code", "opus", parent_model="opus", ran_on=[]) is None


def test_an_alias_means_nothing_to_another_harness() -> None:
    assert baseline_key("pi", "opus", parent_model="anthropic:claude-opus-5", ran_on=[]) is None
