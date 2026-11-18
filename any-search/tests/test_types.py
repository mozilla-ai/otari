"""The rules the metadata types hold a provider's description to."""

import pytest
from pydantic import ValidationError

from any_search import OptionSpec, ProviderMetadata


def _metadata(*options: OptionSpec) -> ProviderMetadata:
    return ProviderMetadata(
        name="example",
        doc_url="https://example.com/docs",
        env_key=None,
        env_api_base=None,
        requires_api_key=False,
        requires_api_base=False,
        default_api_base=None,
        tier="test",
        max_results=10,
        query_in_url=False,
        key_in_url=False,
        options=list(options),
    )


def _tier(name: str = "depth") -> OptionSpec:
    return OptionSpec(name=name, type="string", enum=["basic", "advanced"], default="basic", cost_tier=True)


def test_a_cost_tier_lists_its_tiers_and_the_default_among_them() -> None:
    assert _tier().cost_tier
    with pytest.raises(ValidationError, match="needs an enum and a default"):
        OptionSpec(name="depth", type="string", cost_tier=True)
    with pytest.raises(ValidationError, match="needs an enum and a default"):
        OptionSpec(name="depth", type="string", enum=["basic", "advanced"], cost_tier=True)
    with pytest.raises(ValidationError, match="needs an enum and a default"):
        OptionSpec(name="depth", type="string", enum=["basic"], default="advanced", cost_tier=True)


def test_a_provider_has_one_cost_tier_option_at_most() -> None:
    assert [option.name for option in _metadata(_tier()).options] == ["depth"]
    with pytest.raises(ValidationError, match="more than one cost tier option: depth, mode"):
        _metadata(_tier(), _tier("mode"))


def test_a_provider_reports_no_cost_unless_it_says_so() -> None:
    assert _metadata().reports_cost is False
