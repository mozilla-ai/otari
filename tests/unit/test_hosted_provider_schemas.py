"""The wire shapes of the hosted-providers surface.

Two things are pinned without a database. No response shape has a field that
could carry a credential, which is what keeps the stored key from ever leaving
the process over the API. And a price is the input/output pair or nothing, so
``model_pricing`` is never asked to store half a row.
"""

import pytest
from pydantic import ValidationError
from sqlmodel import SQLModel

import gateway.schemas.providers as schemas

RESPONSE_SHAPES = (
    schemas.HostedProviderPublic,
    schemas.HostedProvidersPublic,
    schemas.HostedModelPublic,
    schemas.HostedModelsPublic,
    schemas.HostedModelsRefreshPublic,
    schemas.HostedCatalogProviderRefreshPublic,
    schemas.HostedCatalogKeptGroupPublic,
    schemas.HostedCatalogRefreshPublic,
    schemas.HostedAvailableModelsPublic,
)


@pytest.mark.parametrize("shape", RESPONSE_SHAPES, ids=lambda shape: shape.__name__)
def test_no_response_shape_can_carry_a_credential(shape: type[SQLModel]) -> None:
    fields = set(shape.model_fields)

    assert "api_key" not in fields
    assert "encrypted_api_key" not in fields
    assert not any("secret" in name or "token" in name for name in fields)


def test_only_the_write_shapes_accept_a_credential() -> None:
    assert "api_key" in schemas.HostedProviderCreateRequest.model_fields
    assert "api_key" in schemas.HostedProviderUpdateRequest.model_fields


def test_a_public_provider_names_a_key_by_its_tail_only() -> None:
    assert "api_key_last4" in schemas.HostedProviderPublic.model_fields
    assert schemas.HostedProviderPublic.model_fields["api_key_last4"].default is None


@pytest.mark.parametrize("shape", [schemas.HostedModelCreateRequest, schemas.HostedModelUpdateRequest])
def test_half_a_price_is_refused(shape: type[SQLModel]) -> None:
    body = {"model": "gpt-4o", "input_price_per_million": 2.5}
    if shape is schemas.HostedModelUpdateRequest:
        body.pop("model")

    with pytest.raises(ValidationError, match="set together"):
        shape.model_validate(body)


@pytest.mark.parametrize("shape", [schemas.HostedModelCreateRequest, schemas.HostedModelUpdateRequest])
def test_a_cache_rate_needs_the_pair_beside_it(shape: type[SQLModel]) -> None:
    body = {"model": "gpt-4o", "cache_read_price_per_million": 0.25}
    if shape is schemas.HostedModelUpdateRequest:
        body.pop("model")

    with pytest.raises(ValidationError, match="cache rates need"):
        shape.model_validate(body)


def test_a_toggle_travels_alone() -> None:
    request = schemas.HostedModelUpdateRequest.model_validate({"enabled": False})

    assert request.enabled is False
    assert request.input_price_per_million is None


def test_a_full_price_with_cache_rates_is_accepted() -> None:
    request = schemas.HostedModelCreateRequest.model_validate(
        {
            "model": "gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10,
            "cache_read_price_per_million": 1.25,
        }
    )

    assert request.cache_read_price_per_million == 1.25
    assert request.cache_write_price_per_million is None


def test_a_negative_rate_is_refused() -> None:
    with pytest.raises(ValidationError):
        schemas.HostedModelCreateRequest.model_validate(
            {"model": "gpt-4o", "input_price_per_million": -1, "output_price_per_million": 10}
        )
