"""Durable files require explicit or unambiguous account selection."""

import uuid
from datetime import UTC, datetime

import pytest

from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverride
from gateway.services.provider_files.accounts import select_file_key
from gateway.services.provider_files.contracts import FilesError


def _key(*, default: bool = False) -> OrgProviderKey:
    return OrgProviderKey(organization_id=uuid.uuid4(), provider="anthropic", name="test", is_org_default=default)


def test_ambiguous_keys_do_not_use_inference_oldest_fallback() -> None:
    with pytest.raises(FilesError, match="ambiguous"):
        select_file_key([(_key(), None), (_key(), None)])


def test_workspace_pin_precedes_organization_default() -> None:
    default, pinned = _key(default=True), _key()
    override = WorkspaceProviderKeyOverride(
        organization_id=pinned.organization_id,
        workspace_id=uuid.uuid4(),
        org_provider_key_id=pinned.id,
        is_default=True,
    )
    assert select_file_key([(default, None), (pinned, override)]) is pinned


def test_disabled_explicit_default_fails_closed() -> None:
    key = _key(default=True)
    override = WorkspaceProviderKeyOverride(
        organization_id=key.organization_id, workspace_id=uuid.uuid4(), org_provider_key_id=key.id, disabled=True
    )
    with pytest.raises(FilesError, match="unavailable"):
        select_file_key([(key, override), (_key(), None)])


def test_archived_explicit_default_fails_closed() -> None:
    key = _key(default=True)
    key.archived_at = datetime.now(UTC)
    with pytest.raises(FilesError, match="unavailable"):
        select_file_key([(key, None), (_key(), None)])


def test_empty_and_unique_selection() -> None:
    assert select_file_key([]) is None
    key = _key()
    assert select_file_key([(key, None)]) is key
