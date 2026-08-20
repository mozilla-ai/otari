"""Unit tests for organization-scoped provider key resolution and caching.

Mirrors `test_provider_store_service.py`'s shape: pure precedence logic and
the cache accessors are tested with in-memory fixtures, no database.
"""

import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverride
from gateway.repositories.tenancy.org_provider_key_repository import resolve_active_key
from gateway.services.secret_box import SecretDecryptionError, encrypt_secret, generate_secret_key
from gateway.services.tenancy import org_provider_key_service as store
from gateway.services.tenancy.org_provider_key_service import cached_org_provider_kwargs, reset_org_provider_cache

ORG_ID = uuid.uuid4()
WORKSPACE_ID = uuid.uuid4()
OTHER_WORKSPACE_ID = uuid.uuid4()


def _key(*, name: str, is_org_default: bool = False, created_at: datetime | None = None) -> OrgProviderKey:
    return OrgProviderKey(
        organization_id=ORG_ID,
        provider="openai",
        name=name,
        is_org_default=is_org_default,
        created_at=created_at or datetime.now(UTC),
    )


def _override(key: OrgProviderKey, *, is_default: bool = False, disabled: bool = False) -> WorkspaceProviderKeyOverride:
    return WorkspaceProviderKeyOverride(
        workspace_id=WORKSPACE_ID,
        organization_id=key.organization_id,
        org_provider_key_id=key.id,
        is_default=is_default,
        disabled=disabled,
    )


# --------------------------------------------------------------------------- #
# resolve_active_key: the three-tier precedence, no database involved
# --------------------------------------------------------------------------- #


def test_resolve_active_key_returns_none_for_no_candidates() -> None:
    assert resolve_active_key([]) is None


def test_resolve_active_key_falls_back_to_earliest_created_with_no_default_or_pin() -> None:
    older = _key(name="a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = _key(name="b", created_at=datetime(2026, 1, 2, tzinfo=UTC))
    # Ordered oldest-first, as the repository queries it (see the docstring).
    assert resolve_active_key([(older, None), (newer, None)]) is older


def test_resolve_active_key_prefers_org_default_over_earliest_created() -> None:
    older = _key(name="a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    default = _key(name="b", is_org_default=True, created_at=datetime(2026, 1, 2, tzinfo=UTC))
    assert resolve_active_key([(older, None), (default, None)]) is default


def test_resolve_active_key_prefers_workspace_pin_over_org_default() -> None:
    default = _key(name="a", is_org_default=True)
    pinned = _key(name="b")
    assert resolve_active_key([(default, None), (pinned, _override(pinned, is_default=True))]) is pinned


def test_resolve_active_key_skips_a_disabled_default_and_falls_back() -> None:
    default = _key(name="a", is_org_default=True, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    fallback = _key(name="b", created_at=datetime(2026, 1, 2, tzinfo=UTC))
    disabled = _override(default, disabled=True)
    assert resolve_active_key([(default, disabled), (fallback, None)]) is fallback


def test_resolve_active_key_returns_none_when_every_candidate_is_disabled() -> None:
    key = _key(name="a")
    assert resolve_active_key([(key, _override(key, disabled=True))]) is None


# --------------------------------------------------------------------------- #
# The overlay cache
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


def test_cached_org_provider_kwargs_returns_none_until_loaded() -> None:
    assert cached_org_provider_kwargs(WORKSPACE_ID, "openai") is None


def test_cached_org_provider_kwargs_is_scoped_by_workspace_and_provider() -> None:
    store._org_cache[(WORKSPACE_ID, "openai")] = {"api_key": "sk-org"}
    store._org_cached_at = time.monotonic()

    assert cached_org_provider_kwargs(WORKSPACE_ID, "openai") == {"api_key": "sk-org"}
    assert cached_org_provider_kwargs(OTHER_WORKSPACE_ID, "openai") is None
    assert cached_org_provider_kwargs(WORKSPACE_ID, "anthropic") is None


def test_reset_clears_the_cache() -> None:
    store._org_cache[(WORKSPACE_ID, "openai")] = {"api_key": "sk-org"}
    store._org_cached_at = time.monotonic()
    assert cached_org_provider_kwargs(WORKSPACE_ID, "openai") == {"api_key": "sk-org"}, "the entry is set before reset"

    reset_org_provider_cache()
    assert cached_org_provider_kwargs(WORKSPACE_ID, "openai") is None


# --------------------------------------------------------------------------- #
# _row_to_entry: decrypt-or-skip, same contract as provider_store_service's
# --------------------------------------------------------------------------- #


def test_row_to_entry_decrypts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = OrgProviderKey(
        organization_id=ORG_ID,
        provider="openai",
        name="default",
        encrypted_api_key=encrypt_secret("sk-live"),
        last4="live",
    )
    assert store._row_to_entry(key) == {"api_key": "sk-live"}


def test_row_to_entry_carries_base_and_client_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = OrgProviderKey(
        organization_id=ORG_ID,
        provider="openai",
        name="home_lab",
        api_base="http://x/v1",
        encrypted_api_key=encrypt_secret("tok"),
        client_args={"timeout": 30},
    )
    assert store._row_to_entry(key) == {
        "api_base": "http://x/v1",
        "client_args": {"timeout": 30},
        "api_key": "tok",
    }


def test_row_to_entry_raises_when_key_cannot_be_decrypted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = OrgProviderKey(
        organization_id=ORG_ID, provider="openai", name="default", encrypted_api_key=encrypt_secret("sk")
    )
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with pytest.raises(SecretDecryptionError):
        store._row_to_entry(key)


# --------------------------------------------------------------------------- #
# to_public: client_args never round-trips a credential-shaped field
# --------------------------------------------------------------------------- #


def test_to_public_redacts_credential_shaped_client_arg_keys() -> None:
    """Bedrock's classic IAM shape genuinely needs `aws_access_key_id` /
    `aws_secret_access_key` inside `client_args` (see
    `services/bedrock_gateway_auth.py`), so the field cannot be rejected
    outright; it still must never come back over the API, the same treatment
    `encrypted_api_key` itself already gets. A non-matching field (`region_name`)
    passes through unchanged."""
    key = _key(name="bedrock-primary")
    key.client_args = {
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAABCDEFGHIJKLMNOP",
        "aws_secret_access_key": "supersecretvalue",
        "api_key": "sk-smuggled",
        "Authorization": "Bearer xyz",
    }

    public = key.to_public()

    assert public.client_args == {
        "region_name": "us-east-1",
        "aws_access_key_id": "***",
        "aws_secret_access_key": "***",
        "api_key": "***",
        "Authorization": "***",
    }


def test_to_public_with_no_credential_shaped_keys_is_unchanged() -> None:
    key = _key(name="openai-primary")
    key.client_args = {"region_name": "us-east-1", "timeout": 30}

    assert key.to_public().client_args == {"region_name": "us-east-1", "timeout": 30}


def test_to_public_client_args_none_stays_none() -> None:
    key = _key(name="plain")
    assert key.client_args is None
    assert key.to_public().client_args is None


def test_to_public_include_client_args_false_still_redacts_nothing_since_it_is_already_none() -> None:
    key = _key(name="plain")
    key.client_args = {"aws_secret_access_key": "supersecretvalue"}
    assert key.to_public(include_client_args=False).client_args is None
