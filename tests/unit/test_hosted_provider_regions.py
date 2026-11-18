"""The pure rules behind hosted provider regions: names, locations, the region a request takes, its price key."""

import uuid
from collections.abc import Iterator
from dataclasses import replace

import pytest

from gateway.core.config import hosted_provider_name_problem, region_instance, region_slug_problem, split_region
from gateway.core.settings.inference import data_location, data_location_admitted, data_locations
from gateway.ports.model_provider_port import HostedCredential
from gateway.services.pricing_service import configure_hosted_regions, pricing_key_forms
from gateway.services.providers import CountryGate, CountryNotServedError, ensure_hosted_country
from gateway.services.providers import _workspace_hosted_access as access
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


def test_a_provider_name_may_not_hold_the_region_separator() -> None:
    assert hosted_provider_name_problem("token@hub", "openai") is not None


@pytest.mark.parametrize(
    ("name", "split"), [("tokenhub@sg", ("tokenhub", "sg")), ("tokenhub", ("tokenhub", None)), ("x@", ("x@", None))]
)
def test_a_name_splits_into_its_provider_and_the_region_it_pins(name: str, split: tuple[str, str | None]) -> None:
    assert split_region(name) == split


def test_a_region_instance_joins_the_name_and_the_slug() -> None:
    assert (region_instance("tokenhub", "sg"), region_instance("tokenhub", None)) == ("tokenhub@sg", "tokenhub")


@pytest.fixture
def _hosted_regions() -> Iterator[None]:
    configure_hosted_regions(lambda name: name == "tokenhub@sg")
    yield
    configure_hosted_regions(None)


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
def test_an_instance_that_is_no_hosted_region_keeps_its_own_price_key() -> None:
    """``team@eu`` from config.yml is an instance of its own, not a region of ``team``."""
    assert pricing_key_forms("team@eu:gpt-4o") == ["team@eu:gpt-4o", "team@eu/gpt-4o"]


def test_a_request_takes_the_first_region_whose_countries_admit_it() -> None:
    provider = _provider(_region("us", countries=("US",)), _region("sg", countries=("SG",)), _region("any"))

    assert ensure_hosted_country(CountryGate("SG"), provider).response_provider == "sg"
    assert ensure_hosted_country(CountryGate("FR"), provider).response_provider == "any"
    assert ensure_hosted_country(None, provider) is provider


def test_a_request_no_region_admits_is_refused() -> None:
    provider = _provider(_region("us", countries=("US",)), _region("sg", countries=("SG",)))

    with pytest.raises(CountryNotServedError):
        ensure_hosted_country(CountryGate("FR"), provider)


@pytest.fixture
def _workspace(monkeypatch: pytest.MonkeyPatch) -> uuid.UUID:
    workspace_id = uuid.uuid4()
    monkeypatch.setattr(access, "_allowed_locations", {workspace_id: frozenset({"EU"})})
    return workspace_id


def test_a_workspace_keeps_the_regions_holding_data_where_it_allows(_workspace: uuid.UUID) -> None:
    provider = _provider(_region("us", location="US"), _region("de", location="DE"), _region("fr", location="FR"))

    narrowed = narrow_to_workspace_locations(provider, _workspace, "tokenhub")

    assert [candidate.response_provider for candidate in narrowed.candidates()] == ["de", "fr"]


def test_a_workspace_with_no_region_where_it_allows_is_refused(_workspace: uuid.UUID) -> None:
    provider = _provider(_region("us", location="US"), _region("undeclared"))

    with pytest.raises(WorkspaceHostedAccessDeniedError, match="keeps data where this workspace allows"):
        narrow_to_workspace_locations(provider, _workspace, "tokenhub")


def test_a_workspace_listing_no_locations_keeps_every_region() -> None:
    provider = _provider(_region("us", location="US"), _region("undeclared"))

    assert narrow_to_workspace_locations(provider, uuid.uuid4(), "tokenhub") is provider
