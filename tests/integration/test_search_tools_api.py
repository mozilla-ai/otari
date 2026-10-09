"""Integration tests for the /api/v1/search-tools CRUD endpoints.

A search tool used to be declarable only in a config file, so a deployment
configured through the dashboard could not use POST /api/v1/search at all (issue
#601). These cover the route in: keys are write-only, the same rules startup
validation applies are applied here, config-file tools stay honored and
read-only, and a tool added at runtime is immediately dispatchable.
"""

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

import any_fetch
import any_search
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.settings.tools import (
    ToolInstance,
    effective_fetch_instances,
    effective_search_instances,
    in_loop_default,
)
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.models.tools import SearchToolCredential
from gateway.schemas.tools import SearchProviderOptionSchema
from gateway.services import search_backend
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.search_backend import SearchHit, SearchOutcome
from gateway.services.search_tool_store_service import refresh_tool_instances, reset_search_tool_cache
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


@pytest.fixture(autouse=True)
def _no_legacy_search_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy in-loop settings decide the in-loop default before a single
    # instance does, so one left in the shell would stop every pin below.
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)


def _create(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    payload = {"name": "local", "provider": "searxng", "api_base": "http://searxng:8080", **body}
    return client.post(f"{API_ROOT}/search-tools", json=payload, headers=headers)


def _store_row(db_session: Session, **fields: Any) -> None:
    """Store a row as a release before the write rules could have, bypassing the route."""
    db_session.add(SearchToolCredential(**{"options": {}, **fields}))
    db_session.commit()


def _config(client: TestClient) -> GatewayConfig:
    config: GatewayConfig = client.app.state.config  # type: ignore[attr-defined]
    return config


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


@pytest.mark.parametrize("name", ["a/b", "exa:main"])
def test_name_used_as_a_path_segment_or_pricing_key_carries_no_slash_or_colon(
    client: TestClient, master_key_header: dict[str, str], name: str
) -> None:
    """#1231: the OpenAPI document states the rule, so the request model refuses the name."""
    resp = _create(client, master_key_header, name=name)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "name"]


@pytest.mark.parametrize("name", ["builtin_fetch", "None", "BUILTIN_FETCH"])
def test_reserved_names_are_refused_in_any_case(
    client: TestClient, master_key_header: dict[str, str], name: str
) -> None:
    resp = _create(client, master_key_header, name=name)
    assert resp.status_code == 422
    assert "its name is reserved" in resp.json()["detail"]


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


def test_a_stored_fetch_instance_joins_the_fetch_catalog_only(
    catalog_client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _create(
        catalog_client, master_key_header, name="stored-fetch", kind="fetch", provider="exa", api_base=None, api_key="k"
    )
    assert created.status_code == 201, created.text
    assert _catalog(catalog_client, "fetch", headers=master_key_header)["exa"]["instances"] == [
        "exa-fetch",
        "stored-fetch",
    ]
    assert _catalog(catalog_client, headers=master_key_header)["exa"]["instances"] == ["from-file"]


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


# --------------------------------------------------------------------------- #
# Fetch instances and the write rules
# --------------------------------------------------------------------------- #


def test_a_fetch_instance_is_stored_listed_by_kind_and_overlaid(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _create(client, master_key_header, name="fake-fetch", kind="fetch", provider="fake", api_base=None)
    assert created.status_code == 201, created.text
    assert created.json()["kind"] == "fetch"
    assert created.json()["pinned_web_search_default_tool"] is None

    search = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()
    assert search["stored"] == []
    assert [(tool["name"], tool["kind"]) for tool in search["config"]] == [("from-file", "search")]

    fetch = client.get(f"{API_ROOT}/search-tools", params={"kind": "fetch"}, headers=master_key_header).json()
    assert [(tool["name"], tool["kind"]) for tool in fetch["stored"]] == [("fake-fetch", "fetch")]
    assert [(tool["name"], tool["provider"]) for tool in fetch["config"]] == [("builtin_fetch", "builtin")]

    config = _config(client)
    assert "fake-fetch" in effective_fetch_instances(config)
    assert "fake-fetch" not in config.search_tools


def test_an_instances_kind_cannot_change(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header, name="fake-fetch", kind="fetch", provider="fake", api_base=None)
    resp = client.patch(f"{API_ROOT}/search-tools/fake-fetch", json={"kind": "search"}, headers=master_key_header)
    assert resp.status_code == 422
    assert "kind cannot change" in resp.json()["detail"]
    same = client.patch(
        f"{API_ROOT}/search-tools/fake-fetch", json={"kind": "fetch", "timeout": 5}, headers=master_key_header
    )
    assert same.status_code == 200, same.text


def test_names_are_unique_across_search_and_fetch_instances(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    clash = _create(client, master_key_header, name="from-file", kind="fetch", provider="fake", api_base=None)
    assert clash.status_code == 422
    assert "A search instance named 'from-file' exists" in clash.json()["detail"]

    _create(client, master_key_header, name="fake-fetch", kind="fetch", provider="fake", api_base=None)
    clash = _create(client, master_key_header, name="fake-fetch")
    assert clash.status_code == 422
    assert "A fetch instance named 'fake-fetch' exists" in clash.json()["detail"]


def test_options_are_checked_against_the_providers_schema(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    unknown = _create(client, master_key_header, name="f", provider="fake", api_base=None, options={"bogus": 1})
    assert unknown.status_code == 422
    assert "option 'bogus' is not one the provider knows" in unknown.json()["detail"]
    refused = _create(
        client, master_key_header, name="e", provider="exa", api_base=None, api_key="k", options={"type": "neural"}
    )
    assert refused.status_code == 422
    assert "option 'type' has a value the provider refuses" in refused.json()["detail"]
    # Until any-search has its SearXNG adapter, a searxng instance's options are passed as today.
    assert _create(client, master_key_header, options={"engines": "brave", "anything": 1}).status_code == 201


def test_a_key_rotation_leaves_options_that_predate_the_rules_alone(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _store_row(db_session, name="old", provider="fake", options={"bogus": 1})
    rotated = client.patch(f"{API_ROOT}/search-tools/old", json={"api_key": "rotated"}, headers=master_key_header)
    assert rotated.status_code == 200, rotated.text
    assert rotated.json()["last4"] == "ated"

    resent = client.patch(f"{API_ROOT}/search-tools/old", json={"options": {"bogus": 2}}, headers=master_key_header)
    assert resent.status_code == 422
    # A new provider is checked against the options the row keeps.
    moved = client.patch(
        f"{API_ROOT}/search-tools/old", json={"provider": "exa", "api_base": None}, headers=master_key_header
    )
    assert moved.status_code == 422
    assert "option 'bogus' is not one the provider knows" in moved.json()["detail"]


def test_fetch_tool_names_a_fetch_instance_and_only_a_search_instance_has_one(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    nowhere = _create(client, master_key_header, name="s", provider="fake", api_base=None, fetch_tool="nope")
    assert nowhere.status_code == 422
    assert "fetch_tool must name a fetch instance" in nowhere.json()["detail"]
    on_fetch = _create(
        client, master_key_header, name="f", kind="fetch", provider="fake", api_base=None, fetch_tool="builtin_fetch"
    )
    assert on_fetch.status_code == 422
    assert "only a search instance has one" in on_fetch.json()["detail"]

    created = _create(client, master_key_header, name="s", provider="fake", api_base=None, fetch_tool="builtin_fetch")
    assert created.status_code == 201, created.text
    assert created.json()["fetch_tool"] == "builtin_fetch"
    assert effective_search_instances(_config(client))["s"].fetch_tool == "builtin_fetch"

    _create(client, master_key_header, name="fake-fetch", kind="fetch", provider="fake", api_base=None)
    url = f"{API_ROOT}/search-tools/s"
    assert client.patch(url, json={"fetch_tool": "fake-fetch"}, headers=master_key_header).json()["fetch_tool"] == (
        "fake-fetch"
    )
    assert client.patch(url, json={"fetch_tool": None}, headers=master_key_header).json()["fetch_tool"] is None


# --------------------------------------------------------------------------- #
# The pin: a second search instance names the first as the default
# --------------------------------------------------------------------------- #


def test_a_second_search_instance_pins_the_first_as_the_default(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    body = _create(client, master_key_header).json()
    assert body["pinned_web_search_default_tool"] == "from-file"
    assert "wins over the configuration file until it is cleared" in body["notice"]

    config = _config(client)
    assert config.web_search_default_tool == "from-file"
    default = in_loop_default(config)
    assert isinstance(default, ToolInstance) and default.name == "from-file"
    fields = {
        field["key"]: field
        for field in client.get(f"{API_ROOT}/tool-settings", headers=master_key_header).json()["fields"]
    }
    assert fields["web_search_default_tool"]["value"] == "from-file"

    third = _create(client, master_key_header, name="third").json()
    assert third["pinned_web_search_default_tool"] is None
    assert third["notice"] is None


def test_no_pin_when_the_count_stays_at_one_or_a_default_is_named(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    # Overrides the configured entry rather than adding an instance.
    shadow = _create(client, master_key_header, name="from-file", provider="exa", api_base=None, api_key="k")
    assert shadow.json()["pinned_web_search_default_tool"] is None
    client.patch(f"{API_ROOT}/tool-settings", json={"web_search_default_tool": "none"}, headers=master_key_header)
    assert _create(client, master_key_header).json()["pinned_web_search_default_tool"] is None
    assert _config(client).web_search_default_tool == "none"


def test_a_default_that_names_nothing_is_replaced_by_the_pin(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Treated as unset (rule 7), so the default was the one instance, and the pin keeps it so."""
    _config(client).web_search_default_tool = "gone"
    assert _create(client, master_key_header).json()["pinned_web_search_default_tool"] == "from-file"


def test_the_pin_refuses_a_stored_instance_whose_name_is_reserved(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _config(client)._search_tool_baseline = {}
    _store_row(db_session, name="none", provider="fake")
    resp = _create(client, master_key_header)
    assert resp.status_code == 422
    assert "delete 'none' and create it again under another name" in resp.json()["detail"]
    stored = client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()["stored"]
    assert [tool["name"] for tool in stored] == ["none"]


def test_the_pin_refuses_a_configured_instance_whose_name_is_reserved(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _config(client)._search_tool_baseline = {"None": {"provider": "fake"}}
    resp = _create(client, master_key_header)
    assert resp.status_code == 422
    assert "rename 'None' in the configuration file" in resp.json()["detail"]


async def _refresh_replica(database_url: str, config: GatewayConfig) -> None:
    url = database_url.replace("postgresql+psycopg2://", "postgresql://", 1).replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    engine = create_async_engine(url)
    try:
        async with AsyncSession(engine) as session:
            await refresh_tool_instances(session, config)
    finally:
        await engine.dispose()


def test_a_refresh_brings_the_pinned_default_with_the_rows(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig
) -> None:
    """Another replica gets the second instance and the default naming the first in one refresh."""
    _create(client, master_key_header)
    replica = GatewayConfig(search_tools={"from-file": {"provider": "exa", "api_key": "file-key"}})
    asyncio.run(_refresh_replica(test_config.database_url, replica))
    assert set(replica.search_tools) == {"from-file", "local"}
    assert replica.web_search_default_tool == "from-file"
    default = in_loop_default(replica)
    assert isinstance(default, ToolInstance) and default.name == "from-file"


# --------------------------------------------------------------------------- #
# Connection tests
# --------------------------------------------------------------------------- #


def _test(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    payload = {"name": "probe", "provider": "fake", **body}
    return client.post(f"{API_ROOT}/search-tools/test", json=payload, headers=headers)


def test_an_unsaved_search_instance_is_tested_with_one_query(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    with patch("gateway.api.routes.search_tools.track_request") as track:
        resp = _test(client, master_key_header, query="otari")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "error": None, "hits": 3, "characters": None}
    # A provider call, so /api/v1/usage/in-flight sees it while it runs.
    assert track.call_args.kwargs == {"endpoint": f"{API_ROOT}/search-tools/test", "model": "probe", "provider": "fake"}
    # Nothing was stored.
    assert client.get(f"{API_ROOT}/search-tools", headers=master_key_header).json()["stored"] == []


def test_a_failed_test_reports_the_tag_never_the_content(client: TestClient, master_key_header: dict[str, str]) -> None:
    failed = _test(client, master_key_header, query="q", options={"error": "rate_limited"})
    assert failed.json() == {"ok": False, "error": "rate_limited", "hits": None, "characters": None}
    in_body = _test(client, master_key_header, query="q", options={"in_body_error": "no_results"})
    assert in_body.json()["error"] == "no_results"


def test_an_unsaved_fetch_instance_is_tested_with_one_fetch(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    body = _test(client, master_key_header, kind="fetch", url="https://example.com", options={"text": "hello"}).json()
    assert body == {"ok": True, "error": None, "hits": None, "characters": 5}


def test_a_test_needs_a_query_or_a_url_as_its_kind_says(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _test(client, master_key_header).status_code == 422
    assert _test(client, master_key_header, kind="fetch", query="q").status_code == 422
    # And the create's checks apply to what it tests, so a test never approves what cannot be saved.
    assert _test(client, master_key_header, name="a:b", query="q").status_code == 422
    assert _test(client, master_key_header, query="q", fetch_tool="nope").status_code == 422
    url = "https://example.com"
    assert _test(client, master_key_header, kind="fetch", url=url, fetch_tool="builtin_fetch").status_code == 422
    taken = _test(client, master_key_header, name="from-file", kind="fetch", url=url)
    assert taken.status_code == 422
    assert "A search instance named 'from-file' exists" in taken.json()["detail"]


def test_a_stored_instance_is_tested_by_name(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header, name="fake-search", provider="fake", api_base=None, options={"error": "timeout"})
    resp = client.post(f"{API_ROOT}/search-tools/fake-search/test", json={"query": "q"}, headers=master_key_header)
    assert resp.json() == {"ok": False, "error": "timeout", "hits": None, "characters": None}


def test_builtin_fetch_has_no_test_yet(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/search-tools/builtin_fetch/test", json={"url": "https://a"}, headers=master_key_header
    )
    assert resp.status_code == 400
    assert "no connection test yet" in resp.json()["detail"]
    assert client.delete(f"{API_ROOT}/search-tools/builtin_fetch", headers=master_key_header).status_code == 404


def test_testing_an_unknown_instance_is_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/search-tools/nope/test", json={"query": "q"}, headers=master_key_header)
    assert resp.status_code == 404


def test_a_searxng_instance_is_tested_through_the_direct_endpoints_client(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params["q"] == "slow":
            raise httpx.ReadTimeout("slow", request=request)
        if request.url.params["q"] == "broken":
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"results": [{"url": "https://a.example"}, {"url": "https://b.example"}]})

    monkeypatch.setattr(search_backend, "_client", httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    searxng = {"provider": "searxng", "api_base": "http://searxng:8080"}
    assert _test(client, master_key_header, query="otari", **searxng).json() == {
        "ok": True,
        "error": None,
        "hits": 2,
        "characters": None,
    }
    assert str(requests[0].url).startswith("http://searxng:8080/search")
    assert _test(client, master_key_header, query="broken", **searxng).json()["error"] == "http_error"
    assert _test(client, master_key_header, query="slow", **searxng).json()["error"] == "timeout"
    # A port that does not parse passes the shape check, and fails the call, not the route.
    bad_port = {"provider": "searxng", "api_base": "http://searxng:8o8o"}
    assert _test(client, master_key_header, query="q", **bad_port).json()["error"] == "network"
