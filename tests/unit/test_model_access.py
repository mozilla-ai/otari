"""Unit tests for the per-key model access-control matcher and validation."""

import uuid
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.error_codes import INVALID_MODEL, MODEL_NOT_ALLOWED, MODEL_NOT_FOUND, MODEL_NOT_SERVING
from gateway.core.settings.tools import ALL_SEARCH_TOOLS
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.api_keys import ApiKeyService
from gateway.services.catalog import (
    OfferingRow,
    build_organization_selectors,
    build_selector_index,
    reset_selector_index,
    set_selector_index,
)
from gateway.services.model_access import (
    effective_allowlist,
    effective_tool_allowlist,
    intersect_allowlists,
    is_allowlist_subset,
    is_model_allowed,
    is_tool_allowed,
    is_tool_allowlist_subset,
    key_model_refusal,
    legacy_tool_grants,
    misspelled_model_refusal,
    org_model_refusal,
    tool_not_allowed_detail,
    tools_within_user,
    validate_allowed_models,
    validate_allowed_tools,
)
from gateway.services.organizations import AccessLists
from gateway.services.tenancy import org_provider_key_service as org_store
from gateway.services.tenancy.org_provider_key_service import KeyOffer

NO_TOOLS: frozenset[str] = frozenset()


class _Key:
    """Minimal stand-in for an APIKey row carrying only ``allowed_models``."""

    def __init__(self, allowed_models: list[str] | None) -> None:
        self.allowed_models = allowed_models


class _User:
    """Minimal stand-in for a User row carrying only ``allowed_models``."""

    def __init__(self, allowed_models: list[str] | None) -> None:
        self.allowed_models = allowed_models


def test_none_allowlist_is_unrestricted() -> None:
    assert is_model_allowed(None, "openai:gpt-4o") is True


def test_empty_allowlist_is_deny_all() -> None:
    # The load-bearing distinction: [] must NOT collapse into None.
    assert is_model_allowed([], "openai:gpt-4o") is False
    assert is_model_allowed([], "anthropic:claude-3") is False


def test_exact_match() -> None:
    assert is_model_allowed(["openai:gpt-4o"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:gpt-4o"], "openai:gpt-4o-mini") is False


def test_instance_wildcard() -> None:
    assert is_model_allowed(["openai:*"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:*"], "anthropic:claude-3") is False


def test_prefix_glob() -> None:
    assert is_model_allowed(["openai:gpt-4*"], "openai:gpt-4o") is True
    assert is_model_allowed(["openai:gpt-4*"], "openai:gpt-3.5-turbo") is False


def test_effective_allowlist_key_wins() -> None:
    assert effective_allowlist(_Key(["openai:*"])) == ["openai:*"]  # type: ignore[arg-type]
    assert effective_allowlist(_Key(None)) is None  # type: ignore[arg-type]
    assert effective_allowlist(None) is None
    # A deny-all key stays deny-all (not conflated with unrestricted).
    assert effective_allowlist(_Key([])) == []  # type: ignore[arg-type]


def test_effective_allowlist_inherits_user_default() -> None:
    # A key with no list of its own inherits the user's default.
    assert effective_allowlist(_Key(None), _User(["openai:*"])) == ["openai:*"]  # type: ignore[arg-type]
    # The key's own list wins over the user default (it may only narrow it).
    assert effective_allowlist(_Key(["openai:gpt-4o"]), _User(["openai:*"])) == ["openai:gpt-4o"]  # type: ignore[arg-type]
    # A deny-all user default is inherited, not conflated with unrestricted.
    assert effective_allowlist(_Key(None), _User([])) == []  # type: ignore[arg-type]
    # No user, no key list -> unrestricted.
    assert effective_allowlist(_Key(None), None) is None  # type: ignore[arg-type]


def test_is_allowlist_subset_inherit_and_unrestricted() -> None:
    # A child that inherits (None) never broadens -> always a subset.
    assert is_allowlist_subset(None, ["openai:gpt-4o"], tool_keys=NO_TOOLS) is True
    # An unrestricted parent (None) covers any child.
    assert is_allowlist_subset(["openai:*"], None, tool_keys=NO_TOOLS) is True
    # Deny-all child fits any parent; deny-all parent rejects a granting child.
    assert is_allowlist_subset([], ["openai:*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:gpt-4o"], [], tool_keys=NO_TOOLS) is False


def test_is_allowlist_subset_concrete_and_wildcards() -> None:
    # Concrete child within a parent wildcard.
    assert is_allowlist_subset(["openai:gpt-4o"], ["openai:*"], tool_keys=NO_TOOLS) is True
    # Different instance is never covered.
    assert is_allowlist_subset(["anthropic:claude-3"], ["openai:*"], tool_keys=NO_TOOLS) is False
    # A wildcard child needs a parent at least as broad.
    assert is_allowlist_subset(["openai:*"], ["openai:*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:*"], ["openai:gpt-4*"], tool_keys=NO_TOOLS) is False
    # A child glob that extends the parent glob is covered; a shorter one is not.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-*"], tool_keys=NO_TOOLS) is True
    assert is_allowlist_subset(["openai:gpt-*"], ["openai:gpt-4*"], tool_keys=NO_TOOLS) is False
    # A concrete parent cannot cover a wildcard child.
    assert is_allowlist_subset(["openai:gpt-4*"], ["openai:gpt-4o"], tool_keys=NO_TOOLS) is False
    # Every child entry must be covered.
    assert is_allowlist_subset(["openai:gpt-4o", "anthropic:claude-3"], ["openai:*"], tool_keys=NO_TOOLS) is False


def test_validate_passthrough_and_dedup() -> None:
    config = GatewayConfig()
    assert validate_allowed_models(config, None) is None
    assert validate_allowed_models(config, []) == []
    assert validate_allowed_models(config, ["openai:gpt-4o", "openai:gpt-4o"]) == ["openai:gpt-4o"]
    assert validate_allowed_models(config, ["openai:*", "anthropic:claude-3*"]) == ["openai:*", "anthropic:claude-3*"]


@pytest.mark.parametrize(
    "bad",
    [
        "gpt-4o",  # no instance prefix
        "openai:gpt-*-turbo",  # mid-string glob
        "openai:*extra",  # glob not trailing
        "openai:a*b",  # glob not trailing
        "openai:*:*",  # multiple globs / bad shape
        "bogusprovider:x",  # unknown provider/instance
        "openai:",  # empty model
    ],
)
def test_validate_rejects_bad_entries(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_allowed_models(GatewayConfig(), [bad])


# --- catalog IDs in allowed_models ---------------------------------------------

_ORG = uuid.uuid4()
_ORG_WORKSPACE = uuid.uuid4()


@pytest.fixture
def author_namespace_catalog() -> Iterator[None]:
    """Catalog IDs named for the model's author, which names no provider (otari#3069).

    ``mistralai/mistral-small-3`` is on two instances, ``openai/gpt-4o`` on its
    own provider and another, and ``alibaba/qwen3.8-flash`` only on one
    organization's keys, one of them reached by its slug.
    """
    deployment = [
        OfferingRow("deepinfra:mistralai/mistral-small-3", "deepinfra", "deepinfra", 0.10),
        OfferingRow("groq:mistralai/mistral-small-3", "groq", "groq", 0.20),
        OfferingRow("openai:gpt-4o", "openai", "openai", 2.50),
        OfferingRow("azure:gpt-4o", "azure", "azure", 2.00),
    ]
    organization = [
        OfferingRow("deepinfra:alibaba/qwen3.8-flash", "deepinfra", "deepinfra", 0.05),
        OfferingRow("team-key:qwen3.8-flash", "team-key", "together", 0.01),
    ]
    identities = {
        "mistralai/mistral-small-3": ("mistralai/mistral-small-3", tuple(row.selector for row in deployment[:2])),
        "openai/gpt-4o": ("openai/gpt-4o", ("openai:gpt-4o", "azure:gpt-4o")),
        "alibaba/qwen3.8-flash": ("alibaba/qwen3.8-flash", tuple(row.selector for row in organization)),
    }
    view = build_organization_selectors(deployment, organization, identities)
    set_selector_index(
        build_selector_index(
            deployment, identities, organizations={_ORG: view}, workspace_organization={_ORG_WORKSPACE: _ORG}
        )
    )
    yield
    reset_selector_index()


@pytest.mark.usefixtures("author_namespace_catalog")
def test_validate_stores_a_catalog_id_as_its_offerings() -> None:
    """The picker offers catalog IDs; the stored list holds the selectors dispatch matches."""
    stored = validate_allowed_models(GatewayConfig(), ["mistralai/mistral-small-3"])

    assert stored == ["deepinfra:mistralai/mistral-small-3", "groq:mistralai/mistral-small-3"]
    assert is_model_allowed(stored, "groq:mistralai/mistral-small-3")
    assert not is_model_allowed(stored, "groq:mistralai/mistral-large-3")


@pytest.mark.usefixtures("author_namespace_catalog")
def test_validate_dedupes_a_catalog_id_against_its_own_offerings() -> None:
    assert validate_allowed_models(
        GatewayConfig(), ["groq:mistralai/mistral-small-3", "MistralAI/Mistral-Small-3"]
    ) == ["groq:mistralai/mistral-small-3", "deepinfra:mistralai/mistral-small-3"]


@pytest.mark.usefixtures("author_namespace_catalog")
@pytest.mark.parametrize(
    ("within", "expected"),
    [
        (["groq:*"], ["groq:mistralai/mistral-small-3"]),
        (["deepinfra:mistralai/mistral-small-3"], ["deepinfra:mistralai/mistral-small-3"]),
        # Covering none keeps every offering, so the owner check refuses the list.
        (["anthropic:*"], ["deepinfra:mistralai/mistral-small-3", "groq:mistralai/mistral-small-3"]),
    ],
)
def test_validate_narrows_a_catalog_id_to_the_offerings_the_owner_covers(
    within: list[str], expected: list[str]
) -> None:
    assert validate_allowed_models(GatewayConfig(), ["mistralai/mistral-small-3"], within=within) == expected


def test_validate_never_narrows_an_entry_that_names_its_offering() -> None:
    """Only a catalog ID's expansion is narrowed; a selector the caller wrote is kept, and refused if it exceeds."""
    assert validate_allowed_models(GatewayConfig(), ["openai:gpt-4o"], within=["groq:*"]) == ["openai:gpt-4o"]


@pytest.mark.usefixtures("gpt_oss_catalog")
def test_validate_reads_a_catalog_id_whose_author_is_also_a_provider_as_the_model() -> None:
    """``openai/gpt-oss-20b`` was stored as ``openai:gpt-oss-20b``, which OpenAI does not serve."""
    assert validate_allowed_models(GatewayConfig(), ["openai/gpt-oss-20b"]) == ["groq:openai/gpt-oss-20b"]


@pytest.mark.usefixtures("author_namespace_catalog")
def test_validate_keeps_the_legacy_spelling_of_a_served_offering_to_that_offering() -> None:
    """Reading it as the catalog ID would widen an existing restriction to every provider of the model."""
    assert validate_allowed_models(GatewayConfig(), ["openai/gpt-4o"]) == ["openai:gpt-4o"]


@pytest.mark.usefixtures("author_namespace_catalog")
def test_validate_reads_a_catalog_id_from_the_callers_organization() -> None:
    with pytest.raises(ValueError, match="unknown provider or instance 'alibaba'"):
        validate_allowed_models(GatewayConfig(), ["alibaba/qwen3.8-flash"])

    # The offering on the key reached by its slug is one no entry can name.
    expected = ["deepinfra:alibaba/qwen3.8-flash"]
    assert validate_allowed_models(GatewayConfig(), ["alibaba/qwen3.8-flash"], organization_id=_ORG) == expected
    assert validate_allowed_models(GatewayConfig(), ["alibaba/qwen3.8-flash"], workspace_id=_ORG_WORKSPACE) == expected


def test_validate_refuses_a_catalog_id_served_only_on_a_key_reached_by_its_slug() -> None:
    rows = [OfferingRow("team-key:qwen3.8-flash", "team-key", "together", 0.01)]
    identities = {"alibaba/qwen3.8-flash": ("alibaba/qwen3.8-flash", ("team-key:qwen3.8-flash",))}
    view = build_organization_selectors([], rows, identities)
    set_selector_index(build_selector_index([], identities, organizations={_ORG: view}))
    try:
        with pytest.raises(ValueError, match="served only through providers an allowed_models entry cannot name"):
            validate_allowed_models(GatewayConfig(), ["alibaba/qwen3.8-flash"], organization_id=_ORG)
    finally:
        reset_selector_index()


def test_validate_keeps_a_provider_spelling_whose_model_no_entry_can_name_as_before() -> None:
    """``openai/gpt-4o`` served only on a renamed hosted provider was accepted before catalog IDs were read."""
    rows = [OfferingRow("openai-eu:gpt-4o", "openai-eu", "openai", 2.0)]
    set_selector_index(build_selector_index(rows, {"openai/gpt-4o": ("openai/gpt-4o", ("openai-eu:gpt-4o",))}))
    try:
        assert validate_allowed_models(GatewayConfig(), ["openai/gpt-4o"]) == ["openai:gpt-4o"]
    finally:
        reset_selector_index()


@pytest.mark.parametrize("entry", ["acme-openai:gpt-4o", "acme-openai:*", "acme-openai:gpt-4*"])
def test_an_entry_may_name_an_organization_key_by_its_slug(entry: str) -> None:
    assert validate_allowed_models(GatewayConfig(), [entry], key_slugs={"acme-openai"}) == [entry]
    with pytest.raises(ValueError, match="unknown provider or instance 'acme-openai'"):
        validate_allowed_models(GatewayConfig(), [entry])


def test_a_catalog_id_stands_for_an_offering_on_a_slugged_key() -> None:
    rows = [OfferingRow("acme-openai:gpt-4o", "acme-openai", "openai", 2.0)]
    set_selector_index(build_selector_index(rows, {"openai/gpt-4o": ("openai/gpt-4o", ("acme-openai:gpt-4o",))}))
    try:
        stored = validate_allowed_models(GatewayConfig(), ["openai/gpt-4o"], key_slugs={"acme-openai"})
    finally:
        reset_selector_index()

    assert stored == ["acme-openai:gpt-4o"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("author_namespace_catalog")
@pytest.mark.parametrize(
    "scope",
    [
        {"organization_id": _ORG},
        {"workspace_id": _ORG_WORKSPACE},
        {"workspace_id": uuid.uuid4(), "organization_id": _ORG},
    ],
)
async def test_a_changed_key_list_reads_a_catalog_id_from_the_keys_organization(scope: dict[str, uuid.UUID]) -> None:
    """An edit reads the catalog the key was created against, also for a workspace the index has not mapped yet."""
    service = ApiKeyService(cast(UnitOfWork, MagicMock()), cast(Any, MagicMock()), users=cast(Any, MagicMock()))

    access = await service.check_access_lists(
        GatewayConfig(),
        AccessLists(["alibaba/qwen3.8-flash"], None),
        stored=AccessLists(None, None),
        owner_id=None,
        workspace_id=scope.get("workspace_id"),
        organization_id=scope.get("organization_id"),
    )

    assert access.models == ["deepinfra:alibaba/qwen3.8-flash"]


@pytest.mark.usefixtures("author_namespace_catalog")
@pytest.mark.parametrize("entry", ["acme/no-such-model", "mistralai/*", "mistralai/mistral-small-*"])
def test_validate_refuses_a_slash_spelling_that_names_no_catalog_model(entry: str) -> None:
    with pytest.raises(ValueError, match="unknown provider or instance"):
        validate_allowed_models(GatewayConfig(), [entry])


# --- search tool access ------------------------------------------------------

# DeepSeek searches and serves models under one prefix: the shape that put tools beside models.
_TOOLS = GatewayConfig(
    search_tools={"deepseek": {"provider": "deepseek", "api_key": "k"}, "exa-search": {"provider": "exa"}}
)


def test_a_model_list_grants_a_tool_only_by_naming_its_key_whole() -> None:
    assert legacy_tool_grants(_TOOLS, None) is None
    assert legacy_tool_grants(_TOOLS, []) == []
    assert legacy_tool_grants(_TOOLS, ["deepseek:deepseek", "openai:gpt-4o"]) == ["deepseek"]
    assert legacy_tool_grants(_TOOLS, ["deepseek:*", "exa:exa-*"]) == []


def test_the_key_list_wins_then_the_users_then_the_model_lists() -> None:
    def effective(**lists: list[str] | None) -> list[str] | None:
        given = {"key_tools": None, "key_models": None, "user_tools": None, "user_models": None, **lists}
        return effective_tool_allowlist(_TOOLS, **given)

    assert effective() is None
    assert effective(key_tools=["exa-search"], user_tools=["*"]) == ["exa-search"]
    # [] is a real answer, never collapsed into inheriting.
    assert effective(key_tools=[], user_tools=["*"]) == []
    assert effective(user_tools=["*"], key_models=["openai:gpt-4o"]) == ["*"]
    # Neither list set: the key's model list, else its user's, read the deprecated way.
    assert effective(key_models=["deepseek:deepseek"]) == ["deepseek"]
    assert effective(user_models=["exa:exa-search"]) == ["exa-search"]
    # A key's grant is bounded by its user's: a user's wildcard admits a key's exact entry and grants no tool.
    assert effective(key_models=["deepseek:deepseek"], user_models=["exa:exa-search"]) == []
    assert effective(key_models=["deepseek:deepseek"], user_models=["deepseek:*"]) == []
    assert effective(key_models=["deepseek:deepseek", "exa:exa-search"], user_models=["exa:exa-search"]) == [
        "exa-search"
    ]
    # A key restricted to models stays refused every tool until one is granted.
    assert effective(key_models=["deepseek:*"]) == []


def test_a_tool_is_allowed_by_its_name_or_the_star() -> None:
    assert is_tool_allowed(None, "deepseek") is True
    assert is_tool_allowed([], "deepseek") is False
    assert is_tool_allowed([ALL_SEARCH_TOOLS], "deepseek") is True
    assert is_tool_allowed(["exa-search"], "deepseek") is False


def test_a_key_narrows_its_users_tools_and_the_star_covers_only_the_star() -> None:
    assert is_tool_allowlist_subset(None, []) is True
    assert is_tool_allowlist_subset(["*"], None) is True
    assert is_tool_allowlist_subset(["deepseek"], ["*"]) is True
    assert is_tool_allowlist_subset(["*"], ["deepseek", "exa-search"]) is False
    assert is_tool_allowlist_subset(["deepseek"], ["exa-search"]) is False
    assert is_tool_allowlist_subset([], []) is True


def test_a_users_tools_bound_a_key_by_its_tool_list_else_its_model_list() -> None:
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=["*"], user_models=[]) is True
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=["exa-search"], user_models=None) is False
    # With no tool list, the user's model list grants only the tools it names exactly.
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=None, user_models=["deepseek:deepseek"]) is True
    assert tools_within_user(_TOOLS, ["deepseek"], user_tools=None, user_models=["deepseek:*"]) is False
    assert tools_within_user(_TOOLS, ["*"], user_tools=None, user_models=None) is True


def test_a_model_wildcard_does_not_cover_a_key_naming_a_tool_the_deprecated_way() -> None:
    legacy = _TOOLS.legacy_search_tool_keys()
    assert is_allowlist_subset(["deepseek:deepseek"], ["deepseek:*"], tool_keys=legacy) is False
    assert is_allowlist_subset(["deepseek:deepseek"], ["deepseek:deepseek"], tool_keys=legacy) is True
    assert is_allowlist_subset(["deepseek:deepseek-chat"], ["deepseek:*"], tool_keys=legacy) is True


def test_a_tool_list_names_configured_tools_or_the_star() -> None:
    assert validate_allowed_tools(_TOOLS, None) is None
    assert validate_allowed_tools(_TOOLS, []) == []
    assert validate_allowed_tools(_TOOLS, [" deepseek ", "deepseek", "*"]) == ["deepseek", "*"]
    for bad in (["deepseek:deepseek"], ["nope"], [""]):
        with pytest.raises(ValueError):
            validate_allowed_tools(_TOOLS, bad)
    with pytest.raises(ValueError):
        validate_allowed_tools(_TOOLS, "deepseek")  # type: ignore[arg-type]


def test_a_tool_list_names_fetch_tools_and_the_built_in_fetcher_too() -> None:
    config = GatewayConfig(
        search_tools={"exa-search": {"provider": "exa"}}, fetch_tools={"tab": {"provider": "tabstack"}}
    )
    assert validate_allowed_tools(config, ["tab", "builtin_fetch", "exa-search"]) == [
        "tab",
        "builtin_fetch",
        "exa-search",
    ]
    with pytest.raises(ValueError, match="is not a search or fetch tool"):
        validate_allowed_tools(config, ["nope"])


def test_the_refusal_names_the_tools_kind() -> None:
    assert tool_not_allowed_detail("tab", "tabstack:tab", kind="fetch").startswith("Fetch tool 'tab' is not permitted")
    assert tool_not_allowed_detail("deepseek", "deepseek:deepseek").startswith(
        "Search tool 'deepseek' is not permitted"
    )


def test_the_refusal_says_a_model_wildcard_grants_models_only_when_one_matched() -> None:
    plain = tool_not_allowed_detail("deepseek", "deepseek:deepseek")
    assert "Name it in allowed_tools" in plain and "wildcard" not in plain
    legacy = tool_not_allowed_detail("deepseek", "deepseek:deepseek", [["deepseek:*"]])
    assert "A wildcard in allowed_models allows models only" in legacy
    # The user's wildcard is the one that matched when the key names the tool exactly.
    bounded = [["deepseek:deepseek"], ["deepseek:*"]]
    assert "allows models only" in tool_not_allowed_detail("deepseek", "deepseek:deepseek", bounded)
    # An exact entry on the user is no wildcard: the key's own list is what leaves the tool out.
    narrowed = [["openai:gpt-4o"], ["openai:*", "deepseek:deepseek"]]
    assert "wildcard" not in tool_not_allowed_detail("deepseek", "deepseek:deepseek", narrowed)


def test_a_model_list_names_a_search_tool_only_by_its_exact_key() -> None:
    """``exa`` names search tools and no model, so a wildcard or another name under it grants nothing."""
    assert validate_allowed_models(_TOOLS, ["exa:exa-search", "deepseek:*"]) == ["exa:exa-search", "deepseek:*"]
    for bad in ("exa:*", "exa:exa-*", "exa:nope"):
        with pytest.raises(ValueError, match="name tools in allowed_tools"):
            validate_allowed_models(_TOOLS, [bad])


def test_a_tool_named_against_the_rules_has_no_deprecated_grant() -> None:
    """A search entry named ``*`` still loads, with a warning; ``exa:*`` must not read as every tool."""
    config = GatewayConfig(search_tools={"*": {"provider": "exa"}, "exa-search": {"provider": "exa"}})
    assert config.legacy_search_tool_keys() == {"exa:exa-search"}
    assert legacy_tool_grants(config, ["exa:*", "exa:exa-search"]) == ["exa-search"]
    with pytest.raises(ValueError, match="name tools in allowed_tools"):
        validate_allowed_models(config, ["exa:*"])


# --- organization-key refusals ---------------------------------------------


@pytest.fixture
def workspace_id() -> Iterator[uuid.UUID]:
    """A workspace whose overlay entries are cleared after the test."""
    workspace_id = uuid.uuid4()
    org_store._org_model_restrictions.clear()
    org_store._org_key_offers.clear()
    yield workspace_id
    org_store._org_model_restrictions.clear()
    org_store._org_key_offers.clear()


def _narrow(workspace_id: uuid.UUID, *, served: list[str], offered: dict[str, bool] | None) -> None:
    org_store._org_model_restrictions[(workspace_id, "openai")] = served
    org_store._org_key_offers[(workspace_id, "openai")] = KeyOffer(key_name="primary", offered=offered)


def test_an_unnarrowed_provider_refuses_nothing(workspace_id: uuid.UUID) -> None:
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o", through_alias=False) is None


def test_a_served_model_is_not_refused(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    assert org_model_refusal(workspace_id, "openai", "gpt-4o", selector="openai:gpt-4o", through_alias=False) is None


def test_a_model_switched_off_is_refused_as_not_serving(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": False})
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert "offered on provider key 'primary' but not serving" in refusal.detail
    assert "openai:gpt-4o-mini" in refusal.detail


def test_a_model_no_key_offers_is_not_found(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True})
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-does-not-exist", selector="openai:gpt-does-not-exist", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert "Provider key 'primary' does not offer 'openai:gpt-does-not-exist'" in refusal.detail


def test_a_workspace_restriction_on_a_served_model_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    """The key serves it; this workspace chose not to reach it, which is a permission, not a switch."""
    _narrow(workspace_id, served=["gpt-4o"], offered={"gpt-4o": True, "gpt-4o-mini": True})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o-mini", selector="fast", through_alias=True)
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_ALLOWED, 403)
    assert refusal.detail == "Model 'fast' is not permitted for this API key."


def test_a_restriction_on_a_key_offering_no_rows_stays_not_allowed(workspace_id: uuid.UUID) -> None:
    _narrow(workspace_id, served=["gpt-4o"], offered=None)
    refusal = org_model_refusal(
        workspace_id, "openai", "gpt-4o-mini", selector="openai:gpt-4o-mini", through_alias=False
    )
    assert refusal is not None
    assert refusal.code == MODEL_NOT_ALLOWED


def test_the_detail_names_the_selector_not_the_resolved_model(workspace_id: uuid.UUID) -> None:
    """An alias exists partly to keep its target off the wire."""
    _narrow(workspace_id, served=[], offered={"gpt-4o": False})
    refusal = org_model_refusal(workspace_id, "openai", "gpt-4o", selector="fast", through_alias=True)
    assert refusal is not None
    assert "fast" in refusal.detail
    assert "gpt-4o" not in refusal.detail


def test_an_unrestricted_key_is_never_refused_by_its_allow_list() -> None:
    assert key_model_refusal(None, None, selector="no-such-model") is None
    assert key_model_refusal(None, "openai:gpt-4o", selector="openai:gpt-4o") is None


def test_a_restricted_key_is_told_an_unresolvable_selector_is_invalid_not_forbidden() -> None:
    refusal = key_model_refusal(["openai:gpt-4o"], None, selector="no-such-model")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (INVALID_MODEL, 400)
    assert "no-such-model" in refusal.detail


def test_an_unresolvable_selector_points_to_the_workspace_routing_policies() -> None:
    refusal = key_model_refusal(["openai:gpt-4o"], None, selector="faber-verify")
    assert refusal is not None
    assert refusal.detail == (
        "Unknown or unsupported model 'faber-verify'. Send a catalog model id such as 'vendor/model', "
        "a 'provider:model' selector, or an alias or routing policy name. "
        "GET /api/v1/models lists the model ids, aliases and routing policies you can use."
    )


def test_a_restricted_key_is_refused_a_model_its_list_excludes() -> None:
    refusal = key_model_refusal(["openai:gpt-4o"], "gemini:gemini-2.5-flash", selector="gemini:gemini-2.5-flash")
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_ALLOWED, 403)
    assert key_model_refusal(["openai:*"], "openai:gpt-4o", selector="fast") is None


def _narrow_anthropic(workspace_id: uuid.UUID, *, offered: dict[str, bool]) -> None:
    org_store._org_model_restrictions[(workspace_id, "anthropic")] = [m for m, on in offered.items() if on]
    org_store._org_key_offers[(workspace_id, "anthropic")] = KeyOffer(key_name="primary", offered=offered)


def test_an_undated_name_whose_dated_builds_are_all_off_is_not_serving(workspace_id: uuid.UUID) -> None:
    _narrow_anthropic(workspace_id, offered={"claude-haiku-4-5-20251001": False, "claude-sonnet-4-6": True})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-haiku-4-5", selector="anthropic:claude-haiku-4-5", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert refusal.detail == (
        "Model 'anthropic:claude-haiku-4-5' is offered on provider key 'primary' "
        "as 'anthropic:claude-haiku-4-5-20251001' but not serving. Turn it on under Organization > Providers."
    )


def test_an_undated_name_with_two_dated_builds_is_not_found_and_names_them(workspace_id: uuid.UUID) -> None:
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": True, "claude-x-20250601": True})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-x", selector="anthropic:claude-x", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert refusal.detail == (
        "Provider key 'primary' does not offer 'anthropic:claude-x'. "
        "It offers 'anthropic:claude-x-20250101', 'anthropic:claude-x-20250601'; send one of those."
    )


def test_an_undated_name_with_every_dated_build_off_is_not_serving_and_names_them(workspace_id: uuid.UUID) -> None:
    """Two builds is ambiguous only while one serves; with both off, the switch is what refuses."""
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": False, "claude-x-20250601": False})
    refusal = org_model_refusal(
        workspace_id, "anthropic", "claude-x", selector="anthropic:claude-x", through_alias=False
    )
    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_SERVING, 403)
    assert "as 'anthropic:claude-x-20250101', 'anthropic:claude-x-20250601' but not serving" in refusal.detail


@pytest.mark.parametrize("alias", ["fast", "claude-x"])
def test_an_undated_name_behind_an_alias_does_not_name_the_dated_builds(workspace_id: uuid.UUID, alias: str) -> None:
    """A bare alias may share its target's model name, so provenance decides, not the spelling."""
    _narrow_anthropic(workspace_id, offered={"claude-x-20250101": True, "claude-x-20250601": True})
    refusal = org_model_refusal(workspace_id, "anthropic", "claude-x", selector=alias, through_alias=True)
    assert refusal is not None
    assert refusal.code == MODEL_NOT_FOUND
    assert "20250101" not in refusal.detail
    assert "20250601" not in refusal.detail


@pytest.mark.parametrize(
    ("first", "second", "allowed", "refused"),
    [
        (None, ["openai:*"], ["openai:gpt-5"], ["anthropic:claude"]),
        (["openai:*"], None, ["openai:gpt-5"], ["anthropic:claude"]),
        (["openai:*"], ["openai:gpt-5"], ["openai:gpt-5"], ["openai:gpt-4o"]),
        (["openai:gpt-*"], ["openai:gpt-5*", "anthropic:*"], ["openai:gpt-5-mini"], ["openai:gpt-4o", "anthropic:x"]),
        (["openai:gpt-5"], ["anthropic:*"], [], ["openai:gpt-5", "anthropic:x"]),
    ],
)
def test_intersect_allowlists_permits_exactly_what_both_permit(
    first: list[str] | None, second: list[str] | None, allowed: list[str], refused: list[str]
) -> None:
    both = intersect_allowlists(first, second)
    for key in allowed:
        assert is_model_allowed(both, key), key
    for key in refused:
        assert not is_model_allowed(both, key), key


def test_intersect_allowlists_of_two_unrestricted_lists_is_unrestricted() -> None:
    assert intersect_allowlists(None, None) is None


@pytest.fixture
def gpt_oss_catalog() -> Iterator[None]:
    """Two gpt-oss models offered on groq."""
    offerings = {"openai/gpt-oss-120b": "groq:openai/gpt-oss-120b", "openai/gpt-oss-20b": "groq:openai/gpt-oss-20b"}
    rows = [OfferingRow(selector, "groq", "groq", 0.15) for selector in offerings.values()]
    models = {model: (model, (selector,)) for model, selector in offerings.items()}
    set_selector_index(build_selector_index(rows, models))
    yield
    reset_selector_index()


@pytest.mark.usefixtures("gpt_oss_catalog")
def test_a_misspelled_model_is_refused_naming_the_nearest_ids() -> None:
    refusal = misspelled_model_refusal(
        None, instance="openai", provider="openai", model="gpt-oss-12", selector="OpenAI/gpt-oss-12"
    )

    assert refusal is not None
    assert (refusal.code, refusal.status_code) == (MODEL_NOT_FOUND, 404)
    assert refusal.detail == (
        "Unknown model 'OpenAI/gpt-oss-12'. Did you mean openai/gpt-oss-120b or openai/gpt-oss-20b?"
        " See GET /api/v1/models."
    )


@pytest.mark.usefixtures("gpt_oss_catalog")
@pytest.mark.parametrize(
    ("instance", "model"),
    [("openai", "gpt-oss-120b"), ("mistral", "no-such-model")],
    ids=["a model the catalog has", "a prefix the catalog has nothing under"],
)
def test_a_model_that_does_not_look_misspelled_is_not_refused(instance: str, model: str) -> None:
    assert misspelled_model_refusal(None, instance=instance, provider=instance, model=model, selector=model) is None
