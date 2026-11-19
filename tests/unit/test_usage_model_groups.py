"""A usage breakdown's model rows, grouped into the models they are spellings of."""

from collections.abc import Iterator

import pytest

from gateway.schemas.usage_and_telemetry import UsageGroupRow
from gateway.services import catalog as selectors
from gateway.services.usage_and_telemetry._model_groups import grouped_models


def _row(key: str | None, cost: float, *, is_other: bool = False) -> UsageGroupRow:
    return UsageGroupRow(key=key, cost=cost, tokens=10, requests=1, is_other=is_other)


def _index(rows: list[selectors.OfferingRow], identities: selectors.Identities, names: dict[str, str]) -> None:
    selectors.set_selector_index(selectors.build_selector_index(rows, identities, names=names))


@pytest.fixture(autouse=True)
def _empty_index() -> Iterator[None]:
    selectors.reset_selector_index()
    yield
    selectors.reset_selector_index()


def _groups(rows: list[UsageGroupRow]) -> list[tuple[str | None, str | None, str | None]]:
    return [(row.key, row.model_group, row.label) for row in grouped_models(rows)]


def test_the_catalogs_model_wins_and_an_unserved_id_joins_it_by_its_key() -> None:
    """The catalog files two ids the id alone reads apart under one model; an id it does not serve joins by key."""
    _index(
        [
            selectors.OfferingRow("fireworks:accounts/fireworks/models/foo-1p2", "fireworks", "fireworks", 0.5),
            selectors.OfferingRow("nebius:acme/Foo-1.2-Turbo", "nebius", "nebius", 0.6),
        ],
        {"acme/foo-1.2": ("foo12", ("fireworks:accounts/fireworks/models/foo-1p2", "nebius:acme/Foo-1.2-Turbo"))},
        {"acme/foo-1.2": "Foo 1.2"},
    )
    assert _groups(
        [
            _row("ACME/foo-1.2-turbo", 5.0),
            _row("accounts/fireworks/models/foo-1p2", 4.0),
            _row("ACCOUNTS/FIREWORKS/MODELS/foo-1p2", 3.5),
            _row("foo-1.2", 3.0),
            _row("bar-3", 2.0),
            _row(None, 1.5),
            _row(None, 1.0, is_other=True),
        ]
    ) == [
        ("ACME/foo-1.2-turbo", "acme/foo-1.2", "Foo 1.2"),
        ("accounts/fireworks/models/foo-1p2", "acme/foo-1.2", "Foo 1.2"),
        ("ACCOUNTS/FIREWORKS/MODELS/foo-1p2", "acme/foo-1.2", "Foo 1.2"),
        ("foo-1.2", "acme/foo-1.2", "Foo 1.2"),
        ("bar-3", "id:bar3", "bar-3"),
        (None, None, None),
        (None, None, None),
    ]


def test_two_catalog_models_whose_ids_read_alike_stay_two_and_take_no_stranger() -> None:
    _index(
        [
            selectors.OfferingRow("groq:llama-4-70b", "groq", "groq", 0.5),
            selectors.OfferingRow("together:meta/Llama-4-70B", "together", "together", 0.6),
        ],
        {
            "meta/llama-4-70b": ("llama470b", ("groq:llama-4-70b",)),
            "meta/llama-4-70b-instruct": ("llama470binstruct", ("together:meta/Llama-4-70B",)),
        },
        {"meta/llama-4-70b": "Llama 4 70B", "meta/llama-4-70b-instruct": "Llama 4 70B Instruct"},
    )
    assert _groups([_row("llama-4-70b", 3.0), _row("meta/Llama-4-70B", 2.0), _row("Llama_4_70B", 1.0)]) == [
        ("llama-4-70b", "meta/llama-4-70b", "Llama 4 70B"),
        ("meta/Llama-4-70B", "meta/llama-4-70b-instruct", "Llama 4 70B Instruct"),
        ("Llama_4_70B", "id:llama470b", "Llama_4_70B"),
    ]


def test_a_key_two_catalog_models_share_stays_ambiguous_when_only_one_has_usage() -> None:
    """Which models a key could mean is the catalog's answer, not the window's."""
    _index(
        [
            selectors.OfferingRow("groq:llama-4-70b", "groq", "groq", 0.5),
            selectors.OfferingRow("together:meta/Llama-4-70B", "together", "together", 0.6),
        ],
        {
            "meta/llama-4-70b": ("llama470b", ("groq:llama-4-70b",)),
            "meta/llama-4-70b-instruct": ("llama470binstruct", ("together:meta/Llama-4-70B",)),
        },
        {"meta/llama-4-70b": "Llama 4 70B", "meta/llama-4-70b-instruct": "Llama 4 70B Instruct"},
    )
    assert _groups([_row("meta/Llama-4-70B", 2.0), _row("Llama_4_70B", 1.0)]) == [
        ("meta/Llama-4-70B", "meta/llama-4-70b-instruct", "Llama 4 70B Instruct"),
        ("Llama_4_70B", "id:llama470b", "Llama_4_70B"),
    ]


def test_an_unserved_id_joins_the_one_catalog_model_its_key_names_with_no_usage_on_it() -> None:
    _index(
        [selectors.OfferingRow("nebius:zai-org/GLM-5.3", "nebius", "nebius", 0.5)],
        {"z-ai/glm-5.3": ("glm53", ("nebius:zai-org/GLM-5.3",))},
        {"z-ai/glm-5.3": "GLM 5.3"},
    )
    assert _groups([_row("accounts/fireworks/models/glm-5p3", 1.0)]) == [
        ("accounts/fireworks/models/glm-5p3", "z-ai/glm-5.3", "GLM 5.3")
    ]


def test_a_catalog_model_with_no_name_is_named_by_its_catalog_id() -> None:
    _index(
        [selectors.OfferingRow("nebius:zai-org/GLM-5.3", "nebius", "nebius", 0.5)],
        {"z-ai/glm-5.3": ("glm53", ("nebius:zai-org/GLM-5.3",))},
        {},
    )
    assert _groups([_row("zai-org/glm-5.3", 1.0)]) == [("zai-org/glm-5.3", "z-ai/glm-5.3", "z-ai/glm-5.3")]


def test_ids_the_catalog_lacks_group_by_their_key_under_a_name_that_holds_still() -> None:
    """The least of the names, so the row reads the same whichever spelling was busiest in the window."""
    expected = ("id:deepseekv41flash1017", "DeepSeek-V4.1-Flash-1017")
    for order in ((3.0, 1.0), (1.0, 3.0)):
        rows = grouped_models(
            [
                _row("accounts/fireworks/models/deepseek-v4p1-flash-1017", order[0]),
                _row("deepseek-ai/DeepSeek-V4.1-Flash-1017", order[1]),
                _row("deepseek-ai/DeepSeek-V4.1-Pro-1017", 2.0),
            ]
        )
        assert [(row.model_group, row.label) for row in rows] == [
            expected,
            expected,
            ("id:deepseekv41pro1017", "DeepSeek-V4.1-Pro-1017"),
        ]


def test_a_deployment_a_routing_profile_or_an_accounts_own_model_is_its_own_row_named_as_written() -> None:
    """The id rung keeps only an id's last segment, which for these is not the whole model."""
    deployment = "accounts/fireworks/models/llama-v3p1-8b-instruct#accounts/northwind/deployments/k3x9a1"
    profile = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/k8x2m4p9q1z7"
    assert _groups(
        [
            _row(deployment, 5.0),
            _row(profile, 4.0),
            _row("accounts/northwind/models/glm-5p3", 3.0),
            _row("acme/Llama-3.1-8B-Instruct", 2.9),
            _row("meta-llama/Llama-3.1-8B-Instruct", 2.8),
            _row("accounts/fireworks/models/glm-5p3", 2.0),
            _row("TEE/zai-org/GLM-5.3", 1.5),
            _row("GPT-4o", 1.0),
            _row("gpt-4o", 0.5),
        ]
    ) == [
        (deployment, f"id:raw:{deployment.lower()}", deployment),
        (profile, f"id:raw:{profile}", profile),
        (
            "accounts/northwind/models/glm-5p3",
            "id:raw:accounts/northwind/models/glm-5p3",
            "accounts/northwind/models/glm-5p3",
        ),
        ("acme/Llama-3.1-8B-Instruct", "id:raw:acme/llama-3.1-8b-instruct", "acme/Llama-3.1-8B-Instruct"),
        ("meta-llama/Llama-3.1-8B-Instruct", "id:llama318binstruct", "Llama-3.1-8B-Instruct"),
        ("accounts/fireworks/models/glm-5p3", "id:glm53", "GLM-5.3"),
        ("TEE/zai-org/GLM-5.3", "id:glm53", "GLM-5.3"),
        ("GPT-4o", "id:gpt4o", "GPT-4o"),
        ("gpt-4o", "id:gpt4o", "GPT-4o"),
    ]


def test_a_rebuilt_index_is_read_afresh() -> None:
    rows = [selectors.OfferingRow("nebius:zai-org/GLM-5.3", "nebius", "nebius", 0.5)]
    identities: selectors.Identities = {"z-ai/glm-5.3": ("glm53", ("nebius:zai-org/GLM-5.3",))}
    _index(rows, identities, {"z-ai/glm-5.3": "GLM 5.3"})
    assert _groups([_row("zai-org/GLM-5.3", 1.0)])[0][2] == "GLM 5.3"
    selectors.reset_selector_index()
    assert _groups([_row("zai-org/GLM-5.3", 1.0)])[0][2] == "GLM-5.3"


def test_an_id_kept_whole_stays_whole_where_the_catalog_files_it_under_a_public_model() -> None:
    _index(
        [
            selectors.OfferingRow("fireworks:accounts/fireworks/models/glm-5p3", "fireworks", "fireworks", 0.5),
            selectors.OfferingRow("fireworks:accounts/northwind/models/glm-5p3", "fireworks", "fireworks", 0.6),
        ],
        {
            "z-ai/glm-5.3": (
                "glm53",
                ("fireworks:accounts/fireworks/models/glm-5p3", "fireworks:accounts/northwind/models/glm-5p3"),
            )
        },
        {"z-ai/glm-5.3": "GLM 5.3"},
    )
    assert _groups(
        [_row("accounts/fireworks/models/glm-5p3", 2.0), _row("accounts/northwind/models/glm-5p3", 1.0)]
    ) == [
        ("accounts/fireworks/models/glm-5p3", "z-ai/glm-5.3", "GLM 5.3"),
        (
            "accounts/northwind/models/glm-5p3",
            "id:raw:accounts/northwind/models/glm-5p3",
            "accounts/northwind/models/glm-5p3",
        ),
    ]


def test_a_public_path_in_any_case_reads_as_the_same_model() -> None:
    assert _groups(
        [
            _row("accounts/fireworks/models/us.anthropic.claude-sonnet-4-5", 2.0),
            _row("ACCOUNTS/FIREWORKS/MODELS/us.anthropic.claude-sonnet-4-5", 1.0),
        ]
    ) == [
        ("accounts/fireworks/models/us.anthropic.claude-sonnet-4-5", "id:claudesonnet45", "claude-sonnet-4-5"),
        ("ACCOUNTS/FIREWORKS/MODELS/us.anthropic.claude-sonnet-4-5", "id:claudesonnet45", "claude-sonnet-4-5"),
    ]
