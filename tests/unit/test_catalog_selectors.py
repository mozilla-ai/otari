"""The short spellings a request may use for a model, and where they resolve."""

from collections.abc import Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import catalog_selectors as selectors
from gateway.services.provider_kwargs import resolve_provider_selector

_ROWS = [
    ("nebius:zai-org/GLM-5.3", "nebius", "glm-5.3", 0.5),
    ("fireworks:accounts/fireworks/models/glm-5p3", "fireworks", "glm-5p3", 0.7),
    ("fireworks:accounts/fireworks/models/glm-5p3-fp8", "fireworks", "glm-5p3", 0.6),
    ("cerebras:gpt-oss-120b", "cerebras", "gpt-oss-120b", None),
]
_IDENTITIES = {
    "glm-5-3": (
        "glm-5.3",
        (
            "nebius:zai-org/GLM-5.3",
            "fireworks:accounts/fireworks/models/glm-5p3",
            "fireworks:accounts/fireworks/models/glm-5p3-fp8",
        ),
    ),
    "gpt-oss-120b": ("gpt-oss-120b", ("cerebras:gpt-oss-120b",)),
}


@pytest.fixture
def index() -> Iterator[selectors.SelectorIndex]:
    built = selectors.build_selector_index(_ROWS, _IDENTITIES)
    selectors.set_selector_index(built)
    yield built
    selectors.reset_selector_index()


def test_a_short_spelling_is_kept_only_where_it_is_unambiguous(index: selectors.SelectorIndex) -> None:
    assert index.short["nebius:glm-5.3"] == "nebius:zai-org/GLM-5.3"
    # Two Fireworks builds clean to the same id, so neither gets the short form.
    assert "fireworks:glm-5p3" not in index.short
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") == "nebius:glm-5.3"
    assert selectors.short_selector_for("fireworks:accounts/fireworks/models/glm-5p3") is None


def test_a_slug_resolves_to_the_cheapest_priced_offering(index: selectors.SelectorIndex) -> None:
    assert selectors.model_selector_for_slug("glm-5-3") == "nebius:zai-org/GLM-5.3"
    # Unpriced: the first offering, rather than nothing.
    assert selectors.model_selector_for_slug("gpt-oss-120b") == "cerebras:gpt-oss-120b"


def test_a_real_selector_is_never_rewritten(index: selectors.SelectorIndex) -> None:
    assert selectors.resolve_catalog_selector("cerebras:gpt-oss-120b") is None
    assert selectors.resolve_catalog_selector("nebius:glm-5.3") == "nebius:zai-org/GLM-5.3"
    assert selectors.resolve_catalog_selector("glm-5-3") == "nebius:zai-org/GLM-5.3"
    assert selectors.resolve_catalog_selector("nope") is None


def test_the_resolver_relabels_a_short_or_model_selector_like_an_alias(index: selectors.SelectorIndex) -> None:
    config = GatewayConfig(
        master_key="k",
        providers={"nebius": {"api_key": "x"}, "fireworks": {"api_key": "x"}, "cerebras": {"api_key": "x"}},
    )
    short = resolve_provider_selector(config, "nebius:glm-5.3")
    assert (short.instance, short.model, short.alias) == ("nebius", "zai-org/GLM-5.3", "nebius:glm-5.3")
    slug = resolve_provider_selector(config, "glm-5-3")
    assert (slug.instance, slug.model, slug.alias) == ("nebius", "zai-org/GLM-5.3", "glm-5-3")
    verbatim = resolve_provider_selector(config, "cerebras:gpt-oss-120b")
    assert (verbatim.model, verbatim.alias) == ("gpt-oss-120b", None)


def test_an_empty_index_resolves_nothing() -> None:
    selectors.reset_selector_index()
    assert selectors.resolve_catalog_selector("glm-5-3") is None
    assert selectors.short_selector_for("nebius:zai-org/GLM-5.3") is None
