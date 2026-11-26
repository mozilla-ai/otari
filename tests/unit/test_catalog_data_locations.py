"""Which offerings take a data location: a hosted provider's, a pinned region's, and nothing else's."""

from gateway.services.merged_catalog_service import DataLocations

LOCATIONS = DataLocations(
    hosted={"berlin": ("DE", "US"), "legacy@eu": ("DE",)},
    regions={"berlin@virginia": ("US",)},
)


def test_a_hosted_offering_is_its_providers() -> None:
    assert LOCATIONS.of("berlin", "hosted") == ["DE", "US"]
    assert LOCATIONS.of("legacy@eu", "hosted") == ["DE"]


def test_an_offering_pinned_to_a_region_is_that_regions_whichever_key_the_catalog_names() -> None:
    assert LOCATIONS.of("berlin@virginia", "hosted") == ["US"]
    assert LOCATIONS.of("berlin@virginia", "organization") == ["US"]


def test_an_instance_nothing_hosted_declares_takes_no_location() -> None:
    # An organization's key or a ``providers:`` instance that shares a hosted provider's name is not that provider.
    assert LOCATIONS.of("berlin", "organization") == []
    assert LOCATIONS.of("berlin", "deployment") == []
    assert LOCATIONS.of("legacy@eu", "organization") == []
    assert LOCATIONS.of("nebius", "hosted") == []


def test_a_providers_instance_named_like_a_region_takes_no_location() -> None:
    assert LOCATIONS.of("berlin@virginia", "deployment") == []
