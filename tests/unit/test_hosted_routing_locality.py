"""A hosted provider's routing locality: declared once, inherited by its regions, and held at dispatch."""

import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.core.settings.providers import parse_hosted_providers
from gateway.models.providers import HostedProvider, HostedProviderRegion
from gateway.ports.model_provider_port import HostedAccessDeniedError, HostedCredential
from gateway.schemas.providers import (
    HostedProviderCreateRequest,
    HostedProviderPublic,
    HostedProviderUpdateRequest,
)
from gateway.services.providers import HostedLocalityRefusedError, hosted_credential_for
from gateway.services.providers._hosted_credentials import credential_of, region_credential_of
from gateway.services.providers._hosted_runtime import narrow_to_localities


def _credential(name: str, locality: str | None, *alternates: HostedCredential) -> HostedCredential:
    return HostedCredential(
        api_key="sk-test",
        api_base=None,
        response_provider=name,
        routing_locality=locality,
        alternates=alternates,
    )


def test_narrowing_keeps_the_regions_whose_locality_is_allowed_in_order() -> None:
    credential = _credential(
        "edge", None, _credential("edge@a", "cloud"), _credential("edge@b", "local"), _credential("edge@c", "private")
    )

    narrowed = narrow_to_localities(credential, {"local", "private"}, None, "edge")

    assert [candidate.response_provider for candidate in narrowed.candidates()] == ["edge@b", "edge@c"]


def test_narrowing_refuses_when_no_region_declares_an_allowed_locality() -> None:
    credential = _credential("edge", "cloud", _credential("edge@a", None))

    with pytest.raises(HostedLocalityRefusedError) as caught:
        narrow_to_localities(credential, {"local"}, uuid.uuid4(), "edge")

    # The walker skips a refused candidate as it skips any access refusal.
    assert isinstance(caught.value, HostedAccessDeniedError)


def test_an_undeclared_locality_never_matches() -> None:
    with pytest.raises(HostedLocalityRefusedError):
        narrow_to_localities(_credential("edge", None), {"local", "private", "cloud"}, None, "edge")


class _Port:
    def __init__(self, credential: HostedCredential | None) -> None:
        self.credential = credential

    async def resolve_hosted_credential(self, **_: Any) -> HostedCredential | None:
        return self.credential

    async def get_hosted_models(self, **_: Any) -> dict[str, frozenset[str] | None]:
        return {}


@pytest.mark.asyncio
async def test_the_dispatch_lookup_narrows_only_when_asked() -> None:
    port = _Port(_credential("edge", "cloud", _credential("edge@lan", "local")))
    common: dict[str, Any] = {"organization_id": None, "workspace_id": None, "provider": "edge", "model": "m"}

    unchanged = await hosted_credential_for(port, **common)
    narrowed = await hosted_credential_for(port, allowed_localities=frozenset({"local"}), **common)

    assert unchanged is not None
    assert [c.response_provider for c in unchanged.candidates()] == ["edge", "edge@lan"]
    assert narrowed is not None
    assert [c.response_provider for c in narrowed.candidates()] == ["edge@lan"]
    with pytest.raises(HostedLocalityRefusedError):
        await hosted_credential_for(port, allowed_localities=frozenset({"private"}), **common)


def test_a_region_inherits_its_providers_locality() -> None:
    row = HostedProvider(name="edge", provider="openai", api_base="https://edge.example/v1", routing_locality="private")
    own = credential_of(row)
    assert own is not None
    region = HostedProviderRegion(slug="eu", label="EU", enabled=True)

    regional = region_credential_of(own, region)

    assert own.routing_locality == "private"
    assert regional is not None
    assert regional.routing_locality == "private"


def test_a_configured_entry_declares_a_locality_and_refuses_an_unknown_one() -> None:
    entries = parse_hosted_providers({"edge": {"provider": "openai", "api_key": "sk", "routing_locality": "local"}})
    assert entries["edge"].routing_locality == "local"
    assert parse_hosted_providers({"openai": {"api_key": "sk"}})["openai"].routing_locality is None

    with pytest.raises(ValueError, match=r"hosted_providers\.edge\.routing_locality"):
        parse_hosted_providers({"edge": {"provider": "openai", "api_key": "sk", "routing_locality": "orbit"}})


def test_the_api_accepts_only_a_known_locality() -> None:
    assert HostedProviderCreateRequest(provider="openai", routing_locality="cloud").routing_locality == "cloud"
    with pytest.raises(ValidationError):
        HostedProviderCreateRequest.model_validate({"provider": "openai", "routing_locality": "orbit"})
    with pytest.raises(ValidationError):
        HostedProviderUpdateRequest.model_validate({"routing_locality": "orbit"})
    # Null clears the declaration, and is distinguishable from leaving it out.
    cleared = HostedProviderUpdateRequest.model_validate({"routing_locality": None})
    assert "routing_locality" in cleared.model_fields_set
    assert "routing_locality" not in HostedProviderUpdateRequest.model_validate({}).model_fields_set


def test_the_public_view_reads_a_stored_locality_and_an_unknown_one_as_undeclared() -> None:
    def public(locality: str | None) -> HostedProviderPublic:
        row = HostedProvider(
            id=uuid.uuid4(),
            name="edge",
            provider="openai",
            enabled=True,
            serves_unlisted_models=False,
            zero_data_retention=False,
            routing_locality=locality,
        )
        return HostedProviderPublic.from_row(row, client_args=None, api_base=None, decryptable=True)

    assert public("local").routing_locality == "local"
    assert public(None).routing_locality is None
    assert public("orbit").routing_locality is None
