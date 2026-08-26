"""Credential-shaped entries in a free-form settings dict never round-trip.

Three tables hand a provider SDK arbitrary operator-supplied JSON, and all three
are places a real credential legitimately lives: standalone Bedrock keeps its
``aws_secret_access_key`` in ``client_args``, because any-llm's BedrockProvider
never forwards ``api_key`` into the boto3 client it builds. ``OrgProviderKey``
has masked its own since it shipped; otari-ai#1880 is the other two, where
``ProviderCredential.client_args`` returned a live AWS secret to anyone who
could reach ``GET /v1/provider-credentials``.

Masking on read creates the other half of the problem, so it is asserted here
too: an editor that loads a row and saves it back would otherwise store the mask
over the credential it was never shown.
"""

from datetime import UTC, datetime

from gateway.models.entities import ProviderCredential, SearchToolCredential
from gateway.models.secret_fields import (
    REDACTED_VALUE,
    redact_secret_like_values,
    restore_redacted_values,
)


class TestRedactSecretLikeValues:
    def test_masks_by_key_name_and_passes_the_rest_through(self) -> None:
        assert redact_secret_like_values(
            {
                "region_name": "us-east-1",
                "timeout": 1800,
                "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG",
            }
        ) == {
            "region_name": "us-east-1",
            "timeout": 1800,
            "aws_access_key_id": REDACTED_VALUE,
            "aws_secret_access_key": REDACTED_VALUE,
        }

    def test_matches_a_substring_so_a_variant_spelling_cannot_slip_through(self) -> None:
        # Substring and not an exact-name allow-list: an operator names a client
        # kwarg however the SDK expects it, and a fixed set would miss a variant.
        submitted = {"projectAuthorizationToken": "t", "X-Api-Key": "k", "service_account_credentials": "c"}

        assert redact_secret_like_values(submitted) == dict.fromkeys(submitted, REDACTED_VALUE)

    def test_none_stays_none(self) -> None:
        assert redact_secret_like_values(None) is None


class TestRestoreRedactedValues:
    def test_the_mask_echoed_back_keeps_the_stored_value(self) -> None:
        stored = {"region_name": "us-east-1", "aws_secret_access_key": "wJalrXUtnFEMI"}
        submitted = {"region_name": "eu-west-1", "aws_secret_access_key": REDACTED_VALUE}

        assert restore_redacted_values(submitted, stored) == {
            "region_name": "eu-west-1",
            "aws_secret_access_key": "wJalrXUtnFEMI",
        }

    def test_a_real_new_value_replaces_the_stored_one(self) -> None:
        stored = {"aws_secret_access_key": "old"}
        assert restore_redacted_values({"aws_secret_access_key": "new"}, stored) == {"aws_secret_access_key": "new"}

    def test_a_dropped_entry_stays_dropped(self) -> None:
        # The caller is submitting the whole object, so an entry it left out is a
        # removal and not something to put back.
        assert restore_redacted_values({"region_name": "us-east-1"}, {"api_key": "live"}) == {
            "region_name": "us-east-1"
        }

    def test_the_mask_on_a_key_that_is_not_stored_is_taken_literally(self) -> None:
        # Nothing to restore, so there is no stored value to prefer; storing the
        # mask is the only answer available and beats dropping the entry.
        assert restore_redacted_values({"api_key": REDACTED_VALUE}, None) == {"api_key": REDACTED_VALUE}


class TestSerializers:
    def test_provider_credential_masks_its_client_args(self) -> None:
        row = ProviderCredential(
            instance="bedrock",
            provider_type="bedrock",
            last4="MPLE",
            client_args={"region_name": "us-east-1", "aws_secret_access_key": "wJalrXUtnFEMI"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        public = row.to_public_dict()

        assert public["client_args"] == {"region_name": "us-east-1", "aws_secret_access_key": REDACTED_VALUE}
        # The encrypted key was never on the wire to begin with; only last4 is.
        assert "encrypted_api_key" not in public
        assert public["last4"] == "MPLE"

    def test_search_tool_credential_masks_its_options(self) -> None:
        row = SearchToolCredential(
            name="searxng",
            provider="searxng",
            options={"engines": "google", "secondary_api_key": "live-key"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert row.to_public_dict()["options"] == {"engines": "google", "secondary_api_key": REDACTED_VALUE}

    def test_an_empty_settings_dict_still_serializes_as_an_object(self) -> None:
        # The column is non-null and the response type is a dict, so masking must
        # not turn "no options" into null.
        row = ProviderCredential(instance="openai", client_args={}, created_at=datetime.now(UTC))
        assert row.to_public_dict()["client_args"] == {}
