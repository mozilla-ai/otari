"""Each credential shape ``secret_shapes`` masks, and the text beside it that it must leave."""

import pytest

from gateway.secret_shapes import REDACTION_PLACEHOLDER, mask_credentials

# Built at runtime so no literal credential sits in the repository.
CREDENTIALS = {
    "bearer header": "Bearer " + "abcd1234" * 3,
    "openai key": "sk-proj-" + "Ab12" * 6,
    "google key": "AIza" + "Sy12" * 5,
    "github oauth token": "gho_" + "A1b2" * 9,
    "github personal token": "ghp_" + "A1b2" * 9,
    "github fine-grained token": "github_pat_" + "A1b2_" * 6,
    "aws access key id": "AKIA" + "ABCD2345" * 2,
    "aws session key id": "ASIA" + "ABCD2345" * 2,
    "slack bot token": "xoxb-" + "1234-abcd-" * 2,
}


@pytest.mark.parametrize("credential", CREDENTIALS.values(), ids=CREDENTIALS.keys())
def test_a_credential_is_masked_and_its_context_kept(credential: str) -> None:
    masked = mask_credentials(f"provider said: {credential} was rejected")

    assert masked == f"provider said: {REDACTION_PLACEHOLDER} was rejected"


def test_a_url_password_is_masked_with_its_user_and_the_host_kept() -> None:
    masked = mask_credentials("redis://default:hunter2@cache.internal:6379/0")

    assert masked == f"redis://{REDACTION_PLACEHOLDER}@cache.internal:6379/0"


def test_a_named_credential_in_a_query_string_is_masked() -> None:
    masked = mask_credentials("GET /v1/models?api_key=abc123")

    assert "abc123" not in masked


@pytest.mark.parametrize(
    "text",
    [
        # Twelve characters after the prefix, four short of an access key ID.
        "AKIA" + "ABCD" * 3,
        # An access key ID runs to a word boundary, so a longer run is not one.
        "AKIA" + "ABCD2345" * 3,
        # A GitHub prefix with too short a body.
        "ghp_" + "A1b2" * 3,
        # A URL with a user and no password.
        "https://otari@github.com/mozilla-ai/otari",
        # A reference to where a credential comes from.
        "password: ${SMTP_PASSWORD}",
    ],
)
def test_text_that_only_resembles_a_credential_is_kept(text: str) -> None:
    assert mask_credentials(text) == text


def test_the_placeholder_can_be_chosen() -> None:
    assert mask_credentials("Bearer " + "abcd1234" * 3, "<redacted>") == "<redacted>"
