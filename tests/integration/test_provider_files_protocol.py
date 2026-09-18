"""A composed core Files authority derives ownership and returns transient credentials."""

import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from gateway.api.deps import get_config, get_unit_of_work
from gateway.api.routes.provider_files import create_provider_files_router
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_keys import OrgProviderKey
from gateway.services.provider_files.contracts import FileAccount, FileScope, FilesError, OutputPrepare
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.secret_box import encrypt_secret

from .test_provider_file_lifecycle import files_setup as files_setup
from .test_provider_file_lifecycle import metadata

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_authenticated_protocol_prepares_and_finalizes(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
    provider: str,
) -> None:
    service, scope, account = files_setup
    async with service.uow:
        generation = await service.repo.account(account.generation_id)
        assert generation is not None
        generation.provider = provider
        key = OrgProviderKey(
            id=uuid.UUID(generation.credential_ref),
            organization_id=scope.organization_id,
            provider=provider,
            name="Files",
            encrypted_api_key=encrypt_secret("upstream-key"),
        )
        await service.repo.save(key)

    async def authenticate(request: Request, uow: UnitOfWork) -> FileScope:
        if request.headers.get("X-Gateway-Token") != "gateway" or request.headers.get("X-User-Token") != "user":
            raise FilesError(401, "Invalid authentication")
        return scope

    async def authorize(scope: FileScope, body: OutputPrepare, uow: UnitOfWork) -> FileAccount:
        return account

    app = FastAPI()
    app.dependency_overrides[get_unit_of_work] = lambda: service.uow
    app.dependency_overrides[get_config] = lambda: GatewayConfig(
        mode="hosted",
        files_provider_native_enabled=True,
        files_max_count=10,
        files_max_bytes=1024,
        files_max_outstanding_bytes=10240,
    )
    app.include_router(
        create_provider_files_router(
            authenticate=authenticate, authenticate_gateway=authenticate, authorize_attempt=authorize
        ),
        prefix=API_ROOT,
    )
    headers = {"X-Gateway-Token": "gateway", "X-User-Token": "user", "X-Otari-Files-Protocol": "2"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://control") as client:
        denied = await client.post(
            f"{API_ROOT}/gateway/files/uploads/prepare", json={"operation_id": str(uuid.uuid4()), "size_bytes": 100}
        )
        assert denied.status_code == 401 and denied.headers["X-Otari-Files-Protocol"] == "2"
        request: dict[str, Any] = {
            "operation_id": str(uuid.uuid4()),
            "provider": provider,
            "size_bytes": 100,
            "user_id": "foreign",
            "workspace_id": str(uuid.uuid4()),
        }
        prepared = await client.post(f"{API_ROOT}/gateway/files/uploads/prepare", json=request, headers=headers)
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["account"]["api_key"] == "upstream-key"
        assert prepared.json()["account"]["provider"] == provider
        assert prepared.headers["Cache-Control"] == "private, no-store"
        data = metadata()
        finalized = await client.post(
            f"{API_ROOT}/gateway/files/uploads/{prepared.json()['id']}/finalize",
            json={"metadata": data.model_dump(mode="json")},
            headers=headers,
        )
        assert finalized.status_code == 200, finalized.text
        listing = await client.post(f"{API_ROOT}/gateway/files/list", json={"provider": provider}, headers=headers)
        assert listing.json()["data"][0]["id"] == data.id
        async with service.uow:
            stored = await service.repo.get(uuid.UUID(prepared.json()["id"]))
        assert stored is not None and stored.user_id == scope.user_id and stored.workspace_id == scope.workspace_id
