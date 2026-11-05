"""Revoking identity or billing authority cannot race an app credential into existence."""

import asyncio
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.api.routes import users as user_routes
from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget
from gateway.models.tenancy import OrganizationMember, User
from gateway.models.users import User as BillingUser
from gateway.repositories.api_keys.api_key_repository import ApiKeyRepository
from gateway.repositories.tenancy.user_repository import UserRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .conftest import MODEL_NAME
from .test_oauth_apps import approve, code_from, redeem
from .test_oauth_apps import test_config as test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world


def _session(world: _World, who: str) -> dict[str, str]:
    return {"Cookie": f"{SESSION_COOKIE_NAME}={world.sessions[who]}"}


def _deactivate(client: TestClient, world: _World) -> Any:
    return client.patch(
        f"{API_ROOT}/admin/users/{world.users['alpha_member']}",
        json={"is_active": False},
        headers=_session(world, "alpha_operator"),
    )


@pytest.mark.parametrize("clear_metadata", [False, True])
def test_identity_deactivation_revokes_issued_app_keys_permanently(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session], clear_metadata: bool
) -> None:
    token = redeem(client, code_from(approve(client, world))).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 200
    if clear_metadata:
        with db_session_factory() as db:
            key_id = db.scalars(select(APIKey.id).where(APIKey.user_id == str(world.users["alpha_member"]))).one()
        changed = client.patch(
            f"{API_ROOT}/organizations/me/keys/{key_id}",
            json={"metadata": {}},
            headers=_session(world, "alpha_member"),
        )
        assert changed.status_code == 200, changed.text
    deactivated = _deactivate(client, world)
    assert deactivated.status_code == 200, deactivated.text
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 401
    refused = client.post(
        f"{API_ROOT}/chat/completions",
        headers=headers,
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert refused.status_code == 401
    with db_session_factory() as db:
        key = db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).one()
        assert not key.is_active and key.operator_disabled
    reactivated = client.patch(
        f"{API_ROOT}/admin/users/{world.users['alpha_member']}",
        json={"is_active": True},
        headers=_session(world, "alpha_operator"),
    )
    assert reactivated.status_code == 200
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 401


@pytest.mark.parametrize("operation", ["deactivate", "delete_organization", "suspend_member", "remove_member"])
def test_exchange_serializes_with_authority_changes(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    code = code_from(approve(client, world))
    client.cookies.clear()
    membership_id = None
    if operation != "deactivate":
        with db_session_factory() as db:
            moved_user = "alpha_owner" if operation == "delete_organization" else "alpha_member"
            db.add(OrganizationMember(organization_id=world.beta, user_id=world.users[moved_user], role="member"))
            db.commit()
            membership_id = db.scalars(
                select(OrganizationMember.id).where(
                    OrganizationMember.organization_id == world.alpha,
                    OrganizationMember.user_id == world.users["alpha_member"],
                )
            ).one()
    held, release, mutation_started = Event(), Event(), Event()
    original_lock = UserRepository.lock
    original_get_locked = UserRepository.get_locked
    original_pointed = UserRepository.get_ids_by_active_organization

    async def lock(self: UserRepository, user_id: uuid.UUID) -> None:
        if user_id == world.users["alpha_member"]:
            mutation_started.set()
        await original_lock(self, user_id)

    async def get_locked(self: UserRepository, user_id: uuid.UUID) -> User | None:
        result = await original_get_locked(self, user_id)
        if user_id == world.users["alpha_member"]:
            held.set()
            assert await asyncio.to_thread(release.wait, 10), "test did not release identity lock"
        return result

    async def pointed(self: UserRepository, organization_id: uuid.UUID) -> list[uuid.UUID]:
        result = await original_pointed(self, organization_id)
        mutation_started.set()
        return result

    monkeypatch.setattr(UserRepository, "lock", lock)
    monkeypatch.setattr(UserRepository, "get_locked", get_locked)
    monkeypatch.setattr(UserRepository, "get_ids_by_active_organization", pointed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        exchange = executor.submit(redeem, client, code)
        try:
            assert held.wait(5), "exchange did not lock the approving identity"
            if operation == "deactivate":
                mutation = executor.submit(_deactivate, client, world)
            elif operation == "delete_organization":
                mutation = executor.submit(
                    client.delete, f"{API_ROOT}/organizations/me", headers=_session(world, "alpha_owner")
                )
            else:
                mutation = executor.submit(
                    client.request,
                    "PATCH" if operation == "suspend_member" else "DELETE",
                    f"{API_ROOT}/organizations/me/members/{membership_id}",
                    headers=_session(world, "alpha_owner"),
                    json={"status": "suspended"} if operation == "suspend_member" else None,
                )
            assert mutation_started.wait(5), "authority change did not reach its serialization point"
        finally:
            release.set()
        result = exchange.result(timeout=10)
        changed = mutation.result(timeout=10)
    assert result.status_code == 200, result.text
    assert changed.status_code == (409 if operation == "delete_organization" else 200), changed.text
    with db_session_factory() as db:
        key = db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).one()
        assert key.is_active is (operation == "delete_organization")
        if operation in {"suspend_member", "remove_member"}:
            identity = db.get(User, world.users["alpha_member"])
            assert identity is not None and identity.active_organization_id == world.beta


def test_exchange_identity_lock_allows_workspace_membership_foreign_keys(
    client: TestClient, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = code_from(approve(client, world))
    client.cookies.clear()
    held, release = Event(), Event()
    original = UserRepository.get_locked

    async def get_locked(self: UserRepository, user_id: uuid.UUID) -> User | None:
        identity = await original(self, user_id)
        if user_id == world.users["alpha_member"]:
            held.set()
            assert await asyncio.to_thread(release.wait, 10), "test did not release identity lock"
        return identity

    monkeypatch.setattr(UserRepository, "get_locked", get_locked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(redeem, client, code)
        try:
            assert held.wait(5), "exchange did not lock the approving identity"
            joining = executor.submit(
                client.post,
                f"{API_ROOT}/workspaces/{world.workspaces['alpha_two']}/members/{world.users['alpha_member']}",
                headers=_session(world, "alpha_owner"),
            )
            joined = joining.result(timeout=5)
            assert joined.status_code == 201, joined.text
        finally:
            release.set()
        issued = pending.result(timeout=10)
    assert issued.status_code == 200, issued.text


@pytest.mark.parametrize("revocation", ["deleted", "blocked"])
def test_owner_created_and_revoked_between_lookups_is_not_revived(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    revocation: str,
) -> None:
    code = code_from(approve(client, world))
    read, release = Event(), Event()
    original = AsyncSession.execute

    async def execute(self: AsyncSession, statement: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, statement, *args, **kwargs)
        descriptions = getattr(statement, "column_descriptions", ())
        if not read.is_set() and descriptions and descriptions[0].get("expr") is BillingUser:
            read.set()
            assert await asyncio.to_thread(release.wait, 10), "test did not release owner lookup"
        return result

    monkeypatch.setattr(AsyncSession, "execute", execute)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(redeem, client, code)
        try:
            assert read.wait(5), "exchange did not read its attribution owner"
            with db_session_factory() as db:
                db.add(
                    BillingUser(
                        user_id=str(world.users["alpha_member"]),
                        deleted_at=datetime.now(UTC) if revocation == "deleted" else None,
                        blocked=revocation == "blocked",
                    )
                )
                db.commit()
        finally:
            release.set()
        refused = pending.result(timeout=10)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"] == "invalid_grant"
    with db_session_factory() as db:
        owner = db.get(BillingUser, str(world.users["alpha_member"]))
        assert owner is not None and (owner.deleted_at is not None or owner.blocked)
        assert db.scalars(select(APIKey).where(APIKey.user_id == owner.user_id)).first() is None


def test_owner_deletion_waits_for_issuance_then_revokes_the_new_key(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with db_session_factory() as db:
        db.add(BillingUser(user_id=str(world.users["alpha_member"])))
        db.commit()
    code = code_from(approve(client, world))
    held, release, deletion_started = Event(), Event(), Event()
    original_issue = ApiKeyRepository.issue_app_key
    original_load = user_routes._load_user_in_organization

    async def issue(self: ApiKeyRepository, **kwargs: Any) -> None:
        held.set()
        assert await asyncio.to_thread(release.wait, 10), "test did not release issuance"
        await original_issue(self, **kwargs)

    async def load(*args: Any, **kwargs: Any) -> BillingUser:
        deletion_started.set()
        return await original_load(*args, **kwargs)

    monkeypatch.setattr(ApiKeyRepository, "issue_app_key", issue)
    monkeypatch.setattr(user_routes, "_load_user_in_organization", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(redeem, client, code)
        try:
            assert held.wait(5), "exchange did not validate its owner"
            deletion = executor.submit(
                client.delete,
                f"{API_ROOT}/users/{world.users['alpha_member']}",
                headers=_session(world, "alpha_operator"),
            )
            assert deletion_started.wait(5), "deletion did not try to load its owner"
        finally:
            release.set()
        issued = pending.result(timeout=10)
        deleted = deletion.result(timeout=10)
    assert issued.status_code == 200, issued.text
    assert deleted.status_code == 204, deleted.text
    with db_session_factory() as db:
        owner = db.get(BillingUser, str(world.users["alpha_member"]))
        assert owner is not None and owner.deleted_at is not None
        key = db.scalars(select(APIKey).where(APIKey.user_id == owner.user_id)).one()
        assert not key.is_active


def test_issued_app_key_cannot_bypass_owner_request_budget(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    token = redeem(client, code_from(approve(client, world))).json()["access_token"]
    with db_session_factory() as db:
        budget = Budget(request_limit=1)
        db.add(budget)
        db.flush()
        owner = db.get(BillingUser, str(world.users["alpha_member"]))
        assert owner is not None
        owner.budget_id, owner.current_requests = budget.budget_id, 1
        db.commit()
    refused = client.post(
        f"{API_ROOT}/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert refused.status_code == 403, refused.text
    assert refused.json()["code"] == "budget_exceeded"
