"""The pure rules behind hosted provider regions: names, locations, the region a request takes, its price key."""

import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.core.config import hosted_provider_name_problem, region_instance, region_slug_problem
from gateway.core.settings.inference import data_location, data_location_admitted, data_locations
from gateway.exceptions.providers_exceptions import HostedProviderInvalidNameError, HostedProviderRegionKeyRequiredError
from gateway.ports.model_provider_port import HostedCredential
from gateway.services.pricing_service import configure_region_parents, pricing_key_forms
from gateway.services.providers import CountryGate, CountryNotServedError, ensure_hosted_country
from gateway.services.providers import _hosted_name_index as index
from gateway.services.providers._hosted_credentials import normalize_name
from gateway.services.providers._hosted_regions import ensure_regions_follow
from gateway.services.providers._workspace_hosted_access import (
    WorkspaceHostedAccessDeniedError,
    narrow_to_workspace_locations,
)


def _region(name: str, *, countries: tuple[str, ...] | None = None, location: str | None = None) -> HostedCredential:
    return HostedCredential(
        api_key=f"sk-{name}",
        api_base=None,
        response_provider=name,
        supported_countries=countries,
        data_location=location,
    )


def _provider(*regions: HostedCredential) -> HostedCredential:
    return replace(regions[0], alternates=regions[1:])


@pytest.mark.parametrize(("value", "expected"), [("us", "US"), (" de ", "DE"), ("eu", "EU"), ("Global", "GLOBAL")])
def test_a_data_location_is_a_country_code_or_a_zone(value: str, expected: str) -> None:
    assert data_location(value) == expected


@pytest.mark.parametrize("value", ["Mars", "USA", "", "EUR"])
def test_anything_else_is_not_a_data_location(value: str) -> None:
    with pytest.raises(ValueError, match="not a data location"):
        data_location(value)


def test_a_list_of_data_locations_keeps_its_order_without_repeats() -> None:
    assert data_locations(["eu", "US", "EU"]) == ["EU", "US"]


@pytest.mark.parametrize(
    ("location", "allowed", "admitted"),
    [
        ("US", {"US"}, True),
        ("DE", {"EU"}, True),
        ("EU", {"EU"}, True),
        ("CH", {"EU"}, False),
        ("GLOBAL", {"US", "EU"}, False),
        ("GLOBAL", {"GLOBAL"}, True),
        ("US", {"GLOBAL"}, False),
        (None, {"US", "EU", "GLOBAL"}, False),
    ],
)
def test_a_location_is_admitted_only_where_the_list_names_it(
    location: str | None, allowed: set[str], admitted: bool
) -> None:
    assert data_location_admitted(location, allowed) is admitted


@pytest.mark.parametrize("slug", ["us", "eu-west-1", "0", "a" * 32])
def test_a_region_slug_a_selector_can_pin(slug: str) -> None:
    assert region_slug_problem(slug) is None


@pytest.mark.parametrize("slug", ["", "US", "-us", "us west", "us@1", "us:1", "a" * 33])
def test_a_region_slug_a_selector_could_not_pin(slug: str) -> None:
    assert region_slug_problem(slug) is not None


def test_a_configured_name_holding_the_region_separator_still_loads() -> None:
    """Configured before regions existed, it keeps loading; a stored provider is refused it at create."""
    assert hosted_provider_name_problem("token@hub", "openai") is None


def test_a_stored_provider_name_may_not_hold_the_region_separator() -> None:
    with pytest.raises(HostedProviderInvalidNameError):
        normalize_name("token@hub", "openai")


def test_a_region_instance_joins_the_name_and_the_slug() -> None:
    assert (region_instance("tokenhub", "sg"), region_instance("tokenhub", None)) == ("tokenhub@sg", "tokenhub")


@pytest.fixture
def _hosted_regions() -> Iterator[None]:
    configure_region_parents({"tokenhub@sg": "tokenhub", "edge@eu@west": "edge@eu"}.get)
    yield
    configure_region_parents(None)


@pytest.mark.usefixtures("_hosted_regions")
def test_a_region_price_key_falls_back_to_its_provider() -> None:
    assert pricing_key_forms("tokenhub@sg:gpt-4o") == [
        "tokenhub@sg:gpt-4o",
        "tokenhub@sg/gpt-4o",
        "tokenhub:gpt-4o",
        "tokenhub/gpt-4o",
    ]
    assert pricing_key_forms("tokenhub:gpt-4o") == ["tokenhub:gpt-4o", "tokenhub/gpt-4o"]


@pytest.mark.usefixtures("_hosted_regions")
def test_a_region_of_a_provider_whose_name_holds_the_separator_falls_back_to_that_provider() -> None:
    assert pricing_key_forms("edge@eu@west:m")[2:] == ["edge@eu:m", "edge@eu/m"]


@pytest.mark.usefixtures("_hosted_regions")
def test_an_instance_that_is_no_hosted_region_keeps_its_own_price_key() -> None:
    """``team@eu`` from config.yml is an instance of its own, not a region of ``team``."""
    assert pricing_key_forms("team@eu:gpt-4o") == ["team@eu:gpt-4o", "team@eu/gpt-4o"]


@pytest.mark.asyncio
async def test_a_request_takes_the_first_region_whose_countries_admit_it() -> None:
    provider = _provider(_region("us", countries=("US",)), _region("sg", countries=("SG",)), _region("any"))

    assert (await ensure_hosted_country(CountryGate("SG"), provider)).response_provider == "sg"
    assert (await ensure_hosted_country(CountryGate("FR"), provider)).response_provider == "any"
    assert await ensure_hosted_country(None, provider) is provider


@pytest.mark.asyncio
async def test_a_request_no_region_admits_is_refused() -> None:
    provider = _provider(_region("us", countries=("US",)), _region("sg", countries=("SG",)))

    with pytest.raises(CountryNotServedError):
        await ensure_hosted_country(CountryGate("FR"), provider)


@pytest.fixture
def _workspace() -> uuid.UUID:
    return uuid.uuid4()


def test_a_workspace_keeps_the_regions_holding_data_where_it_allows(_workspace: uuid.UUID) -> None:
    provider = _provider(_region("us", location="US"), _region("de", location="DE"), _region("fr", location="FR"))

    narrowed = narrow_to_workspace_locations(provider, frozenset({"EU"}), _workspace, "tokenhub")

    assert [candidate.response_provider for candidate in narrowed.candidates()] == ["de", "fr"]


def test_a_workspace_with_no_region_where_it_allows_is_refused(_workspace: uuid.UUID) -> None:
    provider = _provider(_region("us", location="US"), _region("undeclared"))

    with pytest.raises(WorkspaceHostedAccessDeniedError, match="keeps data where this workspace allows"):
        narrow_to_workspace_locations(provider, frozenset({"EU"}), _workspace, "tokenhub")


def test_a_workspace_listing_no_locations_keeps_every_region() -> None:
    provider = _provider(_region("us", location="US"), _region("undeclared"))

    assert narrow_to_workspace_locations(provider, None, uuid.uuid4(), "tokenhub") is provider


def test_a_region_name_is_known_by_its_provider_and_a_provider_name_never_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider named before regions with the separator in it stays a provider of its own."""
    monkeypatch.setattr(index, "_region_parents", {"tokenhub@sg": "tokenhub"})
    monkeypatch.setattr(index, "_providers", frozenset({"tokenhub", "team", "team@eu"}))

    assert index.hosted_region_parent("tokenhub@sg") == "tokenhub"
    assert index.hosted_region_parent("tokenhub@jp") == "tokenhub", "a region another worker just added"
    assert index.hosted_region_parent("team@eu") is None
    assert index.hosted_region_parent("nobody@eu") is None


def test_a_usage_row_takes_the_location_the_serving_credential_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declaration changed after the request chose its credential does not reach the row."""
    monkeypatch.setattr(index, "_locations", {"tokenhub@sg": ("DE",)})
    monkeypatch.setattr(index, "_served_locations", ContextVar("served", default=None))

    before = index.hosted_data_location("tokenhub@sg")
    index.remember_served_location("tokenhub@sg", "SG")
    index.remember_served_location("tokenhub@us", None)

    assert before == "DE", "outside a served request, the declaration as it stands"
    assert index.hosted_data_location("tokenhub@sg") == "SG"
    assert index.hosted_data_location("tokenhub@us") is None, "an undeclared serving credential records nothing"


def _row(**fields: object) -> SimpleNamespace:
    base: dict[str, object] = {"provider": "openai", "encrypted_api_key": "k", "api_base": "https://a.example/v1"}
    return SimpleNamespace(**{**base, **fields})


def test_a_region_on_its_own_origin_may_not_use_a_key_the_provider_gains() -> None:
    """A keyless provider given a key, endpoint unchanged, would send it to the region's other origin."""
    region = SimpleNamespace(slug="sg", encrypted_api_key=None, api_base="https://b.example/v1")

    with pytest.raises(HostedProviderRegionKeyRequiredError):
        ensure_regions_follow(_row(), [region], moved=False)  # type: ignore[arg-type, list-item]


def test_a_region_on_the_provider_origin_may_use_its_key() -> None:
    region = SimpleNamespace(slug="sg", encrypted_api_key=None, api_base="https://a.example/v2")

    ensure_regions_follow(_row(), [region], moved=False)  # type: ignore[arg-type, list-item]
