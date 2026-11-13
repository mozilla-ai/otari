"""The OSS database preserves OAuth grant bindings and one-use consumption."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401 (registers the shared metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository


@pytest.mark.asyncio
async def test_sqlite_grants_round_trip_and_expire_at_the_boundary() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            grants = OAuthAppGrantRepository(uow)
            now = datetime.now(UTC)
            async with uow:
                organization = Organization(name="OAuth", slug="oauth")
                session.add(organization)
                await session.flush()
                user = User(active_organization_id=organization.id)
                workspace = Workspace(name="App", organization_id=organization.id)
                session.add_all([user, workspace])
                await session.flush()
                await grants.issue(
                    code_hash="c" * 64,
                    client_id="app",
                    redirect_uri="https://app.example/callback",
                    challenge="v" * 43,
                    user_id=user.id,
                    workspace_id=workspace.id,
                    scopes=["inference:run"],
                    expires_at=now + timedelta(minutes=2),
                )
            async with uow:
                bound = await grants.find_bound(
                    code_hash="c" * 64,
                    client_id="app",
                    redirect_uri="https://app.example/callback",
                    challenge="v" * 43,
                    now=now,
                )
                assert bound is not None
                assert (bound.user_id, bound.workspace_id) == (user.id, workspace.id)
                assert bound.expires_at == now + timedelta(minutes=2)
                assert await grants.consume(bound.code_hash, now)
                assert not await grants.consume(bound.code_hash, now)
                await grants.issue(
                    code_hash="e" * 64,
                    client_id="app",
                    redirect_uri="https://app.example/callback",
                    challenge="v" * 43,
                    user_id=user.id,
                    workspace_id=workspace.id,
                    scopes=["inference:run"],
                    expires_at=now,
                )
                assert not await grants.consume("e" * 64, now)
                await grants.prune_expired(now)
                assert await grants.get("e" * 64) is None
    finally:
        await engine.dispose()
