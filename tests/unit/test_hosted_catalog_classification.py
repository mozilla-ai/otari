"""The rule a hosted-provider catalog sweep applies to the deployment price list.

Pure: no database, no provider. What is pinned is which priced keys the surface
keeps, which it removes, and why it keeps the ones it does not offer.
"""

from gateway.services.providers._hosted_catalog import (
    KEEP_DEPLOYMENT_INSTANCE,
    KEEP_GATEWAY_TOOL,
    KEEP_SEARCH_TOOL,
    KEEP_UNATTRIBUTABLE,
    classify_priced_keys,
)


def _verdicts(
    *,
    priced: list[tuple[str, int]],
    offered: set[str] = set(),  # noqa: B006
    deployment_instances: set[str] = set(),  # noqa: B006
    search_providers: set[str] = set(),  # noqa: B006
) -> dict[str, tuple[str, str | None]]:
    results = classify_priced_keys(
        priced,
        offered=frozenset(offered),
        deployment_instances=frozenset(deployment_instances),
        search_providers=frozenset(search_providers),
    )
    return {verdict.model_key: (verdict.verdict, verdict.reason) for verdict in results}


def test_an_offered_model_is_the_catalog_working() -> None:
    verdicts = _verdicts(priced=[("openai:gpt-4o", 2)], offered={"openai:gpt-4o"})

    assert verdicts == {"openai:gpt-4o": ("offered", None)}


def test_a_model_nothing_offers_is_removed() -> None:
    verdicts = _verdicts(priced=[("nebius:llama", 1)], offered={"openai:gpt-4o"})

    assert verdicts == {"nebius:llama": ("removed", None)}


def test_the_legacy_slash_spelling_is_folded_onto_the_roster() -> None:
    """``openai/gpt-4o`` names the same model as the roster's ``openai:gpt-4o``."""
    results = classify_priced_keys(
        [("openai/gpt-4o", 1)], offered=frozenset({"openai:gpt-4o"}), deployment_instances=frozenset()
    )

    [verdict] = results
    assert verdict.verdict == "offered"
    assert verdict.canonical_key == "openai:gpt-4o"


def test_a_gateway_tool_rate_is_kept_and_says_why() -> None:
    verdicts = _verdicts(priced=[("otari:web_search", 1)], offered={"openai:gpt-4o"})

    assert verdicts == {"otari:web_search": ("kept", KEEP_GATEWAY_TOOL)}


def test_a_configured_search_tool_rate_is_kept() -> None:
    verdicts = _verdicts(priced=[("tavily:search", 1)], offered={"openai:gpt-4o"}, search_providers={"tavily"})

    assert verdicts == {"tavily:search": ("kept", KEEP_SEARCH_TOOL)}


def test_a_model_served_by_a_configured_instance_is_kept() -> None:
    verdicts = _verdicts(priced=[("my-vllm:llama", 1)], offered={"openai:gpt-4o"}, deployment_instances={"my-vllm"})

    assert verdicts == {"my-vllm:llama": ("kept", KEEP_DEPLOYMENT_INSTANCE)}


def test_a_key_with_no_prefix_is_kept_rather_than_guessed_at() -> None:
    verdicts = _verdicts(priced=[("gpt-4o", 1)], offered={"openai:gpt-4o"})

    assert verdicts == {"gpt-4o": ("kept", KEEP_UNATTRIBUTABLE)}


def test_offered_outranks_every_keep_reason() -> None:
    """A key the surface offers is kept as offered, whatever else also serves it."""
    verdicts = _verdicts(
        priced=[("my-vllm:llama", 1)],
        offered={"my-vllm:llama"},
        deployment_instances={"my-vllm"},
    )

    assert verdicts == {"my-vllm:llama": ("offered", None)}


def test_version_counts_travel_with_the_verdict() -> None:
    [verdict] = classify_priced_keys([("nebius:llama", 7)], offered=frozenset(), deployment_instances=frozenset())

    assert verdict.versions == 7
