"""A provider's undated name for a dated build an organization's key offers."""

import uuid
from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog as selectors
from gateway.services.catalog._dated import undated_name, undated_spellings
from gateway.services.provider_kwargs import resolve_provider_selector

_HAIKU = "anthropic:claude-haiku-4-5-20251001"
_SONNET_46 = "anthropic:claude-sonnet-4-6"
_TWIN_EARLY = "anthropic:claude-twin-20250101"
_TWIN_LATE = "anthropic:claude-twin-20250601"
_ORG = uuid.uuid4()
_WORKSPACE = uuid.uuid4()
_OTHER_WORKSPACE = uuid.uuid4()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("claude-sonnet-4-6", None),
        ("claude-haiku-4-5", None),
        # Eight digits that are not a calendar date are not a release date.
        ("model-20251399", None),
        ("model-2025-0806", None),
        ("20251001", None),
    ],
)
def test_undated_name_strips_only_a_real_release_date(model_id: str, expected: str | None) -> None:
    assert undated_name(model_id) == expected


def test_an_undated_name_stands_for_exactly_one_dated_build() -> None:
    spellings = undated_spellings({_HAIKU, _SONNET_46, _TWIN_EARLY, _TWIN_LATE})
    assert spellings == {"anthropic:claude-haiku-4-5": _HAIKU}


def test_an_undated_name_the_instance_offers_itself_is_not_a_spelling() -> None:
    assert undated_spellings({_HAIKU, "anthropic:claude-haiku-4-5"}) == {}


def test_dated_builds_are_matched_per_name() -> None:
    offered = ["claude-twin-20250601", "claude-twin-20250101", "claude-twin-2-20250101", "claude-twin"]
    assert selectors.dated_builds("claude-twin", offered) == ("claude-twin-20250101", "claude-twin-20250601")
    assert selectors.dated_builds("claude-none", offered) == ()


@pytest.fixture
def organization_index() -> Iterator[selectors.SelectorIndex]:
    """The organization serves dated Haiku, undated Sonnet 4.6 and two builds of one model."""
    own = [
        selectors.OfferingRow(selector, "anthropic", "anthropic", 1.0)
        for selector in (_HAIKU, _SONNET_46, _TWIN_EARLY, _TWIN_LATE)
    ]
    view = selectors.build_organization_selectors([], own, {})
    built = selectors.build_selector_index(
        [], {}, organizations={_ORG: view}, workspace_organization={_WORKSPACE: _ORG}
    )
    selectors.set_selector_index(built)
    yield built
    selectors.reset_selector_index()


@pytest.mark.usefixtures("organization_index")
def test_the_undated_name_resolves_to_the_one_dated_build_for_the_organization() -> None:
    assert selectors.resolve_undated_selector("anthropic:claude-haiku-4-5", workspace_id=_WORKSPACE) == _HAIKU


@pytest.mark.usefixtures("organization_index")
def test_the_undated_name_resolves_for_nobody_else() -> None:
    assert selectors.resolve_undated_selector("anthropic:claude-haiku-4-5") is None
    assert selectors.resolve_undated_selector("anthropic:claude-haiku-4-5", workspace_id=_OTHER_WORKSPACE) is None


@pytest.mark.usefixtures("organization_index")
def test_ambiguous_and_already_named_offerings_are_left_alone() -> None:
    assert selectors.resolve_undated_selector("anthropic:claude-twin", workspace_id=_WORKSPACE) is None
    # Listed undated, so it names an offering and is never rewritten.
    assert selectors.resolve_undated_selector(_SONNET_46, workspace_id=_WORKSPACE) is None
    assert selectors.resolve_undated_selector(_HAIKU, workspace_id=_WORKSPACE) is None
    # A provider's own id keeps its casing.
    assert selectors.resolve_undated_selector("anthropic:Claude-Haiku-4-5", workspace_id=_WORKSPACE) is None


@pytest.mark.usefixtures("organization_index")
def test_the_resolver_relabels_the_undated_name_like_an_alias() -> None:
    config = GatewayConfig(master_key="k")
    resolved = resolve_provider_selector(config, "anthropic:claude-haiku-4-5", workspace_id=_WORKSPACE)
    assert (resolved.instance, resolved.model, resolved.alias) == (
        "anthropic",
        "claude-haiku-4-5-20251001",
        "anthropic:claude-haiku-4-5",
    )
    verbatim = resolve_provider_selector(config, _SONNET_46, workspace_id=_WORKSPACE)
    assert (verbatim.model, verbatim.alias) == ("claude-sonnet-4-6", None)


@pytest.mark.usefixtures("organization_index")
def test_a_providers_instance_sends_the_undated_name_as_written() -> None:
    """The organization's offer never reaches an instance-addressed selector, so it never rewrites one."""
    config = GatewayConfig(master_key="k", providers={"anthropic": {"api_key": "x"}})
    resolved = resolve_provider_selector(config, "anthropic:claude-haiku-4-5", workspace_id=_WORKSPACE)
    assert (resolved.instance, resolved.model, resolved.alias) == ("anthropic", "claude-haiku-4-5", None)
    # The catalog's own spellings are not part of this rule.
    assert selectors.resolve_catalog_selector("anthropic:claude-haiku-4-5", workspace_id=_WORKSPACE) is None
