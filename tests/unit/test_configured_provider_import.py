"""Which ``providers:`` entries of config.yml a hosted provider can hold, and the create that copies one."""

import pytest

from gateway.services.providers._configured_provider_import import plan_import


def test_an_entry_with_a_key_and_an_endpoint_is_copied_serving_its_roster() -> None:
    plan = plan_import(
        "openai-eu",
        {"provider_type": "openai", "api_key": "sk-1", "api_base": "https://eu.test/v1", "client_args": {"x": 1}},
        hosted_names=set(),
    )

    assert plan.create is not None
    assert (plan.create.provider, plan.create.name, plan.create.api_key, plan.create.api_base) == (
        "openai",
        "openai-eu",
        "sk-1",
        "https://eu.test/v1",
    )
    assert plan.create.client_args == {"x": 1}
    assert plan.create.serves_unlisted_models is False


def test_an_entry_named_after_its_implementation_keeps_no_name_of_its_own() -> None:
    plan = plan_import("anthropic", {"api_key": "sk-ant"}, hosted_names=set())

    assert plan.create is not None
    assert (plan.create.provider, plan.create.name) == ("anthropic", None)


@pytest.mark.parametrize(
    ("name", "entry", "hosted", "reason"),
    [
        ("local", {"provider_type": "openai", "api_base": "http://l/v1", "models": ["m"]}, set(), "sets models"),
        ("groq", {"provider_type": "openai", "api_key": "sk"}, set(), "names another provider"),
        ("anthropic", {"api_key": "sk"}, {"anthropic"}, "already holds the name"),
        ("bedrock", {"client_args": {"region_name": "eu-west-1"}}, set(), "names no key"),
        ("nope", {"provider_type": "not-a-provider", "api_key": "sk"}, set(), "not-a-provider"),
        ("otari", {"api_base": "http://otari.test/v1"}, set(), "is reserved"),
    ],
    ids=["extra-field", "another-providers-name", "taken", "ambient-credentials", "unknown-implementation", "reserved"],
)
def test_an_entry_a_hosted_provider_cannot_hold_stays_in_config(
    name: str, entry: dict[str, object], hosted: set[str], reason: str
) -> None:
    plan = plan_import(name, entry, hosted_names=hosted)

    assert plan.create is None
    assert plan.reason is not None and reason in plan.reason


def test_a_value_the_create_refuses_names_the_field_and_never_the_value() -> None:
    secret = "sk-" + "x" * 2000
    plan = plan_import("anthropic", {"api_key": secret}, hosted_names=set())

    assert plan.create is None
    assert plan.reason is not None and "api_key" in plan.reason
    assert secret not in plan.reason
