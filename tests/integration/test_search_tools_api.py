"""Integration tests for the /api/v1/search-tools CRUD endpoints.

A search tool used to be declarable only in a config file, so a deployment
configured through the dashboard could not use POST /api/v1/search at all (issue
#601). These cover the route in: keys are write-only, the same rules startup
validation applies are applied here, config-file tools stay honored and
read-only, and a tool added at runtime is immediately dispatchable.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import any_fetch
import any_search
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.schemas.tools import SearchProviderOptionSchema
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.search_backend import SearchHit, SearchOutcome
from gateway.services.search_tool_store_service import reset_search_tool_cache
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """Override the shared config with one config-file search tool."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        search_tools={"from-file": {"provider": "exa", "api_key": "file-key"}},
    )


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_search_tool_cache()
    yield
    reset_search_tool_cache()


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _create(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    payload = {"name": "local", "provider": "searxng", "api_base": "http://searxng:8080", **body}
    return client.post(f"{API_ROOT}/search-tools", json=payload, headers=headers)


def test_requires_master_key(client: TestClient) -> None:
    assert client.get(f"{API_ROOT}/search-tools").status_code == 401
    assert client.post(f"{API_ROOT}/search-tools", json={"name": "x", "provider": "searxng"}).status_code == 401
    assert client.delete(f"{API_ROOT}/search-tools/x").status_code == 401


def test_create_lists_and_never_returns_the_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-live-9876")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "local"
    assert body["provider"] == "exa"
    assert body["last4"] == "9876"
    assert "api_key" not in body
    assert "exa-live-9876" not in resp.text

    listed = client.get(f"{API_ROOT}/search-tools", headers=master_key_header)
    assert listed.status_code == 200
    assert [tool["name"] for tool in listed.json()["stored"]] == ["local"]
    assert "exa-live-9876" not in listed.text


def test_keyless_searxng_tool_needs_no_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header)
    assert resp.status_code == 201, resp.text
    assert resp.json()["last4"] is None


def test_storing_a_key_requires_the_secret_key(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET_KEY", raising=False)
    resp = _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-live")
    assert resp.status_code == 400
    assert "OTARI_SECRET_KEY" in resp.json()["detail"]


def test_unsupported_provider_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header, provider="bing")
    assert resp.status_code == 422
    assert "not a supported search provider" in resp.json()["detail"]


def test_provider_requiring_a_key_is_refused_without_one(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header, provider="exa", api_base=None)
    assert resp.status_code == 422
    assert "api_key is required" in resp.json()["detail"]


def test_name_used_as_a_path_segment_may_not_contain_a_slash(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = _create(client, master_key_header, name="a/b")
    assert resp.status_code == 422
    assert "must not contain '/'" in resp.json()["detail"]


def test_non_http_api_base_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header, api_base="file:///etc/passwd")
    assert resp.status_code == 422
    assert "http or https" in resp.json()["detail"]


def test_private_api_base_is_allowed(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The bundled SearXNG sidecar lives on a private address; refusing it would
    reject the main thing this page configures."""
    assert _create(client, master_key_header, api_base="http://searxng:8080").status_code == 201


def test_credentialed_http_api_base_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = _create(client, master_key_header, api_key="adapter-secret")
    assert resp.status_code == 422
    assert "api_base must use https when api_key is set" in resp.json()["detail"]


def test_duplicate_name_conflicts(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _create(client, master_key_header).status_code == 201
    dup = _create(client, master_key_header)
    assert dup.status_code == 409
    assert "already exists" in dup.json()["detail"]


def test_patch_updates_base_keeps_key_then_rotates(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-orig-1111")

    patched = client.patch(
        f"{API_ROOT}/search-tools/local",
        json={"api_base": "https://proxy.internal"},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["api_base"] == "https://proxy.internal"
    assert patched.json()["last4"] == "1111"

    rotated = client.patch(
        f"{API_ROOT}/search-tools/local", json={"api_key": "exa-new-2222"}, headers=master_key_header
    )
    assert rotated.status_code == 200
    assert rotated.json()["last4"] == "2222"
    assert "exa-new-2222" not in rotated.text


def test_patch_refuses_to_clear_a_key_the_provider_needs(client: TestClient, master_key_header: dict[str, str]) -> None:
    """The tool as it would be after the patch is validated, not the patch alone."""
    _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-orig")
    resp = client.patch(f"{API_ROOT}/search-tools/local", json={"api_key": None}, headers=master_key_header)
    assert resp.status_code == 422
    assert "api_key is required" in resp.json()["detail"]
    assert client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()["stored"][0]["last4"] == "orig"


def test_patch_optimistic_precondition(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header)
    stale = client.patch(
        f"{API_ROOT}/search-tools/local",
        json={"api_base": "http://other:8080", "expected_updated_at": "1999-01-01T00:00:00+00:00"},
        headers=master_key_header,
    )
    assert stale.status_code == 412


def test_patch_unknown_tool_is_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.patch(f"{API_ROOT}/search-tools/nope", json={"api_base": "http://x"}, headers=master_key_header)
    assert resp.status_code == 404


def test_delete_removes_the_tool(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header)
    assert client.delete(f"{API_ROOT}/search-tools/local", headers=master_key_header).status_code == 204
    assert client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()["stored"] == []
    assert client.delete(f"{API_ROOT}/search-tools/local", headers=master_key_header).status_code == 404


def test_delete_of_a_config_tool_explains_why_it_cannot(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.delete(f"{API_ROOT}/search-tools/from-file", headers=master_key_header)
    assert resp.status_code == 404
    assert "defined in the config file" in resp.json()["detail"]


def test_list_reports_config_tools_and_shadowing(client: TestClient, master_key_header: dict[str, str]) -> None:
    listed = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()
    assert [tool["name"] for tool in listed["config"]] == ["from-file"]
    assert listed["config"][0]["has_api_key"] is True
    assert listed["config"][0]["shadowed"] is False
    # The config entry's key is never echoed, only the fact that one is set.
    assert "file-key" not in str(listed)

    assert _create(client, master_key_header, name="from-file", provider="exa", api_base=None, api_key="k").json()[
        "shadows_config"
    ]
    after = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()
    assert after["config"][0]["shadowed"] is True
    assert after["stored"][0]["shadows_config"] is True


def test_provider_catalog_reports_what_each_provider_needs(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.get(f"{API_ROOT}/search-tools/providers", headers=master_key_header)
    catalog = {entry["id"]: entry for entry in resp.json()}
    assert catalog["exa"]["requires_api_key"] is True
    assert catalog["exa"]["requires_api_base"] is False
    assert catalog["exa"]["default_api_base"] == "https://api.exa.ai"
    assert catalog["searxng"]["requires_api_key"] is False
    assert catalog["searxng"]["requires_api_base"] is True
    # Nothing supplies one on this config, so the form must ask for it.
    assert catalog["searxng"]["default_api_base"] is None


_INHERITED_URL = "http://searxng.internal:8080"


@pytest.fixture
def catalog_client(test_config: GatewayConfig, clean_database: None) -> Iterator[TestClient]:
    """The shared config, plus a URL a searxng tool inherits and one fetch tool."""
    config = test_config.model_copy(
        update={
            "web_search_url": _INHERITED_URL,
            "fetch_tools": {"exa-fetch": {"provider": "exa", "api_key": "file-key"}},
        },
        deep=True,
    )
    yield from build_test_client(config)


def _catalog(client: TestClient, kind: str | None = None, **request: Any) -> dict[str, dict[str, Any]]:
    resp = client.get(f"{API_ROOT}/search-tools/providers", params={"kind": kind} if kind else None, **request)
    assert resp.status_code == 200, resp.text
    return {entry["id"]: entry for entry in resp.json()}


def test_search_catalog_serves_the_library_metadata(
    catalog_client: TestClient, master_key_header: dict[str, str]
) -> None:
    catalog = _catalog(catalog_client, headers=master_key_header)
    # any-search's fake provider exists for tests: accepted in configuration, never offered.
    assert list(catalog) == ["exa", "searxng"]
    exa = catalog["exa"]
    metadata = any_search.AnySearch.get_provider_metadata("exa")
    assert exa["kind"] == "search"
    assert exa["doc_url"] == metadata.doc_url
    assert exa["tier"] == "production"
    assert exa["max_results"] == metadata.max_results
    assert exa["query_in_url"] is False
    assert exa["options"] == [option.model_dump() for option in metadata.options]
    assert exa["instances"] == ["from-file"]
    # Fetch's own fields stay null on a search entry, and no environment key is reported.
    assert exa["max_urls_per_call"] is None
    assert exa["formats"] is None
    assert "env_key" not in exa


def test_searxng_keeps_its_entry_without_a_schema(
    catalog_client: TestClient, master_key_header: dict[str, str]
) -> None:
    searxng = _catalog(catalog_client, headers=master_key_header)["searxng"]
    assert searxng["requires_api_key"] is False
    assert searxng["requires_api_base"] is True
    assert searxng["default_api_base"] == _INHERITED_URL
    # Null, not empty: its options are passed unchecked, not refused.
    assert searxng["options"] is None
    assert searxng["instances"] == []

    assert _create(catalog_client, master_key_header).status_code == 201
    assert _catalog(catalog_client, headers=master_key_header)["searxng"]["instances"] == ["local"]


def test_fetch_catalog_serves_any_fetch(catalog_client: TestClient, master_key_header: dict[str, str]) -> None:
    catalog = _catalog(catalog_client, "fetch", headers=master_key_header)
    # builtin serves only the implicit builtin_fetch tool, and fake exists for tests.
    assert list(catalog) == ["exa"]
    exa = catalog["exa"]
    metadata = any_fetch.AnyFetch.get_provider_metadata("exa")
    assert exa["kind"] == "fetch"
    assert exa["requires_api_key"] is True
    assert exa["default_api_base"] == metadata.default_api_base
    assert exa["max_urls_per_call"] == metadata.max_urls_per_call
    assert exa["renders_javascript"] is False
    assert exa["formats"] == metadata.formats
    assert exa["options"] == [option.model_dump() for option in metadata.options]
    assert exa["instances"] == ["exa-fetch"]
    assert exa["max_results"] is None


@pytest.mark.parametrize("library", [any_search.AnySearch, any_fetch.AnyFetch], ids=["any-search", "any-fetch"])
def test_every_option_a_library_declares_fits_the_catalog_schema(
    library: type[any_search.AnySearch] | type[any_fetch.AnyFetch],
) -> None:
    """A new option type or field in a library fails here, in the library's own change, not as a 500."""
    for provider in library.get_supported_providers():
        for option in library.get_provider_metadata(provider).options:
            declared = option.model_dump()
            assert SearchProviderOptionSchema(**declared).model_dump() == declared, (provider, option.name)


def test_catalog_kind_is_search_or_fetch(catalog_client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = catalog_client.get(f"{API_ROOT}/search-tools/providers?kind=crawl", headers=master_key_header)
    assert resp.status_code == 422


def _session_token(db: Session, organization: Organization, *, email: str, is_superuser: bool) -> str:
    user = User(email=email, full_name="Reader", active_organization_id=organization.id, is_superuser=is_superuser)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="member", status="active"))
    token = f"otari-sess-{email}"
    db.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    db.commit()
    return token


@pytest.fixture
def sessions(
    catalog_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> dict[str, str]:
    assert catalog_client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    db = db_session_factory()
    try:
        organization = Organization(name="Alpha", slug="alpha")
        db.add(organization)
        db.commit()
        db.refresh(organization)
        return {
            "member": _session_token(db, organization, email="member@alpha.test", is_superuser=False),
            "operator": _session_token(db, organization, email="root@alpha.test", is_superuser=True),
        }
    finally:
        db.close()


def test_every_catalog_reader_lists_providers_and_only_an_operator_sees_the_deployment_in_them(
    catalog_client: TestClient, master_key_header: dict[str, str], sessions: dict[str, str]
) -> None:
    assert catalog_client.get(f"{API_ROOT}/search-tools/providers").status_code == 401

    minted = catalog_client.post(f"{API_ROOT}/keys", json={"key_name": "reader"}, headers=master_key_header)
    api_key = {API_KEY_HEADER: f"Bearer {minted.json()['key']}"}
    by_api_key = _catalog(catalog_client, headers=api_key)
    assert by_api_key["searxng"]["default_api_base"] is None
    assert by_api_key["exa"]["instances"] == []
    # The library's endpoints are the same for everyone, so they are not withheld.
    assert by_api_key["exa"]["default_api_base"] == "https://api.exa.ai"
    assert _catalog(catalog_client, "fetch", headers=api_key)["exa"]["instances"] == []

    by_master_key = _catalog(catalog_client, headers=master_key_header)
    assert by_master_key["searxng"]["default_api_base"] == _INHERITED_URL
    assert by_master_key["exa"]["instances"] == ["from-file"]

    for role, operates in (("member", False), ("operator", True)):
        catalog_client.cookies.set(SESSION_COOKIE_NAME, sessions[role])
        try:
            catalog = _catalog(catalog_client)
            fetch = _catalog(catalog_client, "fetch")
        finally:
            catalog_client.cookies.clear()
        assert catalog["searxng"]["default_api_base"] == (_INHERITED_URL if operates else None), role
        assert catalog["exa"]["instances"] == (["from-file"] if operates else []), role
        assert fetch["exa"]["instances"] == (["exa-fetch"] if operates else []), role


def test_stored_tool_is_immediately_dispatchable(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """The whole point of issue #601: a dashboard-added tool serves /api/v1/search."""
    assert _create(client, master_key_header).status_code == 201
    outcome = SearchOutcome(results=[SearchHit(url="https://example.com", title="Example")])
    mock = AsyncMock(return_value=outcome)
    with patch("gateway.api.routes.search.run_search", mock):
        resp = client.post(f"{API_ROOT}/search/local", json={"query": "otari"}, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert resp.json()["search_tool"] == "local"
    dispatched = mock.call_args.args[0]
    assert dispatched.provider == "searxng"
    assert dispatched.api_base == "http://searxng:8080"


def test_deleting_a_stored_tool_restores_the_config_one(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    _create(client, master_key_header, name="from-file", provider="exa", api_base=None, api_key="stored-key")
    mock = AsyncMock(return_value=SearchOutcome(results=[]))
    with patch("gateway.api.routes.search.run_search", mock):
        client.post(f"{API_ROOT}/search/from-file", json={"query": "q"}, headers=api_key_header)
    assert mock.call_args.args[0].api_key == "stored-key"

    assert client.delete(f"{API_ROOT}/search-tools/from-file", headers=master_key_header).status_code == 204
    with patch("gateway.api.routes.search.run_search", mock):
        client.post(f"{API_ROOT}/search/from-file", json={"query": "q"}, headers=api_key_header)
    assert mock.call_args.args[0].api_key == "file-key"


def test_reencrypt_allows_secret_key_retirement(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    old_key, new_key = generate_secret_key(), generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", old_key)
    _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-rotate")

    monkeypatch.setenv("OTARI_SECRET_KEY", f"{new_key},{old_key}")
    resp = client.post(f"{API_ROOT}/search-tools/reencrypt", headers=master_key_header)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"reencrypted": 1, "unreadable": 0, "skipped": 0}

    monkeypatch.setenv("OTARI_SECRET_KEY", new_key)
    listed = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()
    assert listed["stored"][0]["decryptable"] is True


def test_list_flags_a_key_that_can_no_longer_be_decrypted(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _create(client, master_key_header, provider="exa", api_base=None, api_key="exa-orig")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    listed = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()
    assert listed["stored"][0]["decryptable"] is False
