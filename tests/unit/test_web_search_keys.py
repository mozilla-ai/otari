"""Which web search and fetch keys a workspace uses, how they reach its calls, and which keys may be written."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pytest

from gateway.api.routes._tools import _build_web_retrieval_backend
from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import ToolKind
from gateway.exceptions.tools_exceptions import (
    WebSearchCredentialRefusedError,
    WebSearchKeyIncompleteError,
    WebSearchKeySettingRefusedError,
    WebSearchKeyUnknownProviderError,
)
from gateway.models.tools import (
    OrgWebSearchKey,
    ResolvedWebSearchConfig,
    WebFetchCredential,
    WebSearchCredential,
    WebTool,
    WebToolCredentialSource,
    WorkspaceWebSearchKeyOverride,
)
from gateway.repositories.tools import resolve_web_search_key
from gateway.services import search_backend
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tools import SearchQuery, _web_search_keys, apply_web_access_policy, resolve_search
from gateway.services.web_search_providers import WebSearchProviderError

_START = datetime(2026, 1, 1, tzinfo=UTC)
_ORG = WebToolCredentialSource.ORG
_NO_POLICY = ResolvedWebSearchConfig(
    enabled=True,
    max_results=None,
    purpose_hint=None,
    allowed_domains=None,
    blocked_domains=None,
    provider_options=None,
    authorized_tools=None,
)


def _key(
    name: str, *, provider: str = "tavily", default: bool = False, age: int = 0, kind: str = "search"
) -> OrgWebSearchKey:
    """A key created ``age`` days after the first, so a smaller age is older."""
    return OrgWebSearchKey(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        kind=kind,
        provider=provider,
        name=name,
        encrypted_api_key="ciphertext",
        is_org_default=default,
        created_at=_START + timedelta(days=age),
    )


def _override(key: OrgWebSearchKey, *, pinned: bool = False, disabled: bool = False) -> WorkspaceWebSearchKeyOverride:
    return WorkspaceWebSearchKeyOverride(
        workspace_id=uuid.uuid4(),
        organization_id=key.organization_id,
        org_web_search_key_id=key.id,
        is_default=pinned,
        disabled=disabled,
    )


def _chosen(
    *candidates: tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride | None], unusable: str = ""
) -> str | None:
    chosen = resolve_web_search_key(candidates, lambda key: key.name != unusable)
    return chosen.name if chosen is not None else None


def test_a_workspace_pin_wins_over_every_default() -> None:
    old_default = _key("default", default=True)
    pinned = _key("pinned", provider="brave", age=1)

    assert _chosen((old_default, None), (pinned, _override(pinned, pinned=True))) == "pinned"


def test_without_a_pin_the_default_wins_over_an_older_key() -> None:
    oldest = _key("oldest")
    default = _key("default", provider="brave", default=True, age=1)

    assert _chosen((oldest, None), (default, None)) == "default"


def test_without_a_pin_or_default_the_oldest_key() -> None:
    assert _chosen((_key("oldest"), None), (_key("newer", age=1), None)) == "oldest"


def test_a_key_the_workspace_turned_off_is_passed_over() -> None:
    default = _key("default", default=True)
    other = _key("other", age=1)

    assert _chosen((default, _override(default, disabled=True)), (other, None)) == "other"


def test_a_key_this_deployment_cannot_decrypt_is_passed_over() -> None:
    pinned = _key("pinned")
    other = _key("other", age=1)

    assert _chosen((pinned, _override(pinned, pinned=True)), (other, None), unusable="pinned") == "other"


def test_no_usable_key_means_the_deployments_search() -> None:
    only = _key("only")

    assert _chosen((only, _override(only, disabled=True))) is None
    assert _chosen() is None


def test_a_credential_never_prints_its_key_endpoint_or_options() -> None:
    credential = WebSearchCredential(
        provider="tavily",
        source=_ORG,
        api_key="tvly-secret",
        api_base="https://user:pass@search.internal",
        options={"topic": "news-secret"},
    )

    for printed in (repr(credential), str(credential)):
        assert "tvly-secret" not in printed
        assert "search.internal" not in printed
        assert "news-secret" not in printed


def test_an_organization_key_is_a_credential_from_the_organization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = _key("ours", provider="brave")
    key.encrypted_api_key = encrypt_secret("brv-own")

    assert _web_search_keys.key_credential(key) == WebSearchCredential(
        provider="brave", source=WebToolCredentialSource.ORG, api_key="brv-own"
    )


def test_a_fetch_key_is_a_fetch_credential_with_its_endpoint_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = _key("pages", provider="tabstack", kind="fetch")
    key.encrypted_api_key = encrypt_secret("tbs-own")
    key.api_base = "https://tabstack.example/v1"
    key.options = {"effort": "standard"}

    assert _web_search_keys.key_credential(key) == WebFetchCredential(
        provider="tabstack",
        source=WebToolCredentialSource.ORG,
        api_key="tbs-own",
        api_base="https://tabstack.example/v1",
        options={"effort": "standard"},
    )


def test_a_key_stored_without_a_key_is_never_completed_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "the-deployments-own-key")
    key = _key("keyless", provider="exa")
    key.encrypted_api_key = None
    credential = _web_search_keys.key_credential(key)
    assert isinstance(credential, WebSearchCredential)
    assert credential.api_key is None
    policy = replace(_NO_POLICY, credential=credential)

    with pytest.raises(WebSearchCredentialRefusedError, match="carries no API key"):
        resolve_search(GatewayConfig(), policy, requested=None, request_options=None)


def test_the_workspace_key_wins_over_the_deployments_provider() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="deployment-key")

    backend = _build_web_retrieval_backend(
        base_url="http://searxng:8080",
        search_tool_entry={},
        credential=WebSearchCredential(provider="tavily", source=_ORG, api_key="workspace-key"),
        config=config,
    )

    assert (backend._provider, backend._provider_api_key) == ("tavily", "workspace-key")


def test_without_a_workspace_key_the_deployments_provider_is_used() -> None:
    config = GatewayConfig(web_search_provider="brave", web_search_provider_api_key="deployment-key")

    backend = _build_web_retrieval_backend(base_url=None, search_tool_entry={}, config=config)

    assert (backend._provider, backend._provider_api_key) == ("brave", "deployment-key")


@pytest.mark.parametrize(
    ("search_entry", "carried"),
    [({"type": "otari_web_search"}, True), (None, False)],
    ids=["search-declared", "fetch-only"],
)
def test_admission_carries_the_key_only_for_a_declared_search(
    search_entry: dict[str, str] | None, carried: bool
) -> None:
    credential = WebSearchCredential(provider="tavily", source=_ORG, api_key="k")
    policy = ResolvedWebSearchConfig(
        enabled=True,
        max_results=None,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=None,
        provider_options=None,
        authorized_tools=None,
        credential=credential,
    )
    tools = [WebTool.SEARCH] if search_entry is not None else [WebTool.FETCH]

    grant = apply_web_access_policy(
        policy, requested_tools=tools, search_tool_entry=search_entry, config=GatewayConfig()
    )

    assert grant.search_credential is (credential if carried else None)


def test_the_choice_decrypts_only_until_a_usable_key() -> None:
    pinned, default, oldest = _key("pinned", age=2), _key("default", default=True, age=1), _key("oldest")
    asked: list[str] = []

    def usable(key: OrgWebSearchKey) -> bool:
        asked.append(key.name)
        return key.name != "pinned"

    chosen = resolve_web_search_key([(oldest, None), (default, None), (pinned, _override(pinned, pinned=True))], usable)

    assert chosen is default
    assert asked == ["pinned", "default"]


# The provider call is stubbed above the client, which the keyed search only passes on.
_NO_NETWORK = httpx.MockTransport(lambda _request: httpx.Response(500))


@pytest.mark.asyncio
async def test_direct_search_on_a_workspace_key_reads_the_providers_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_provider_search(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return [
            {"url": "https://kept.example/a", "title": "A", "content": "first"},
            {"url": "https://dropped.example/b", "title": "B", "content": "second"},
            {"url": "https://kept.example/c", "title": "C", "content": "third"},
        ]

    monkeypatch.setattr(search_backend, "provider_search", fake_provider_search)
    credential = WebSearchCredential(provider="tavily", source=_ORG, api_key="workspace-key")

    async with httpx.AsyncClient(transport=_NO_NETWORK) as client:
        outcome = await search_backend.run_keyed_search(
            credential,
            SearchQuery(query="q", max_results=2, domain_filter=("-dropped.example",)),
            timeout_s=3.0,
            client=client,
        )

    assert (seen["provider"], seen["api_key"], seen["query"]) == ("tavily", "workspace-key", "q")
    assert seen["options"] == {"max_results": 2}
    assert seen["timeout_s"] == 3.0
    # The caller's client, the app's web provider client in the route.
    assert seen["client"] is client
    assert [hit.url for hit in outcome.results] == ["https://kept.example/a", "https://kept.example/c"]
    assert outcome.cost_usd is None


@pytest.mark.asyncio
async def test_direct_search_on_a_workspace_key_reports_a_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_provider_search(**_kwargs: Any) -> list[dict[str, Any]]:
        raise WebSearchProviderError("tavily search returned HTTP 401", tag="http_error", status=401)

    monkeypatch.setattr(search_backend, "provider_search", failing_provider_search)

    with pytest.raises(search_backend.SearchProviderError) as raised:
        async with httpx.AsyncClient(transport=_NO_NETWORK) as client:
            await search_backend.run_keyed_search(
                WebSearchCredential(provider="tavily", source=_ORG, api_key="workspace-key"),
                SearchQuery(query="q"),
                timeout_s=3.0,
                client=client,
            )

    assert "workspace-key" not in str(raised.value)
    # Kept for the 502's headers.
    assert (raised.value.tag, raised.value.status) == ("http_error", 401)


@pytest.mark.asyncio
async def test_direct_search_on_a_workspace_key_localizes_to_the_requested_country(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_provider_search(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(search_backend, "provider_search", fake_provider_search)

    async with httpx.AsyncClient(transport=_NO_NETWORK) as client:
        await search_backend.run_keyed_search(
            WebSearchCredential(provider="brave", source=_ORG, api_key="workspace-key"),
            SearchQuery(query="q", max_results=2, country="de"),
            timeout_s=3.0,
            client=client,
        )

    assert seen["options"] == {"max_results": 2, "country": "de"}


class _Overrides:
    """Serves a fixed set of keys as the repository would, recording each read."""

    def __init__(self, keys: list[OrgWebSearchKey]) -> None:
        self.keys = keys
        self.reads: list[str] = []

    def _of(self, kind: str | None) -> list[OrgWebSearchKey]:
        return [key for key in self.keys if kind is None or key.kind == kind]

    async def preferred_candidates(self, *, kind: str | None, **_kwargs: Any) -> list[tuple[OrgWebSearchKey, None]]:
        self.reads.append(f"preferred {kind or 'every kind'}")
        return [(key, None) for key in self._of(kind) if key.is_org_default]

    async def candidate_page(
        self, *, skip: int, limit: int, kind: str | None, **_kwargs: Any
    ) -> list[tuple[OrgWebSearchKey, None]]:
        self.reads.append(f"page {skip} {kind or 'every kind'}")
        return [(key, None) for key in self._of(kind)[skip : skip + limit]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usable_name", "reads"),
    [
        ("default", ["preferred search"]),
        ("k3", ["preferred search", "page 0 search"]),
        ("k45", ["preferred search", "page 0 search", "page 20 search", "page 40 search"]),
    ],
)
async def test_the_choice_reads_only_the_keys_it_reaches(
    monkeypatch: pytest.MonkeyPatch, usable_name: str, reads: list[str]
) -> None:
    keys = [_key(f"k{index}", age=index) for index in range(50)]
    keys.append(_key("default", default=True, age=60))

    def credential(key: OrgWebSearchKey) -> WebSearchCredential | None:
        return WebSearchCredential(provider=key.provider, source=_ORG, api_key="k") if key.name == usable_name else None

    monkeypatch.setattr(_web_search_keys, "key_credential", credential)
    overrides = _Overrides(keys)

    chosen = (
        await _web_search_keys._choose_keys(
            cast(Any, overrides), uuid.uuid4(), uuid.uuid4(), ["search"], _web_search_keys._Decrypted()
        )
    )["search"]

    assert chosen is not None
    assert chosen.name == usable_name
    assert overrides.reads == reads


@pytest.mark.parametrize(
    ("kind", "provider"),
    [
        ("search", "exa"),
        ("search", "mojeek"),
        ("search", "tavily"),
        ("search", "brave"),
        ("fetch", "exa"),
        ("fetch", "tabstack"),
    ],
)
def test_a_key_may_name_a_provider_the_libraries_serve_for_its_kind(kind: ToolKind, provider: str) -> None:
    assert _web_search_keys._validated_provider(kind, f" {provider.upper()} ") == provider


@pytest.mark.parametrize(
    ("kind", "provider"),
    [
        ("search", "fake"),
        ("fetch", "fake"),
        ("fetch", "builtin"),
        ("search", "tabstack"),
        ("fetch", "mojeek"),
        ("fetch", "tavily"),
        ("search", "searxng"),
        ("search", "nowhere"),
    ],
)
def test_a_key_may_not_name_a_provider_its_kind_does_not_offer(kind: ToolKind, provider: str) -> None:
    with pytest.raises(WebSearchKeyUnknownProviderError) as refused:
        _web_search_keys._validated_provider(kind, provider)

    offered = refused.value.message.split("Use one of: ")[1].split(", ")
    assert provider not in offered
    assert not {"fake", "builtin"} & set(offered)


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "provider"), [("search", "exa"), ("search", "tavily"), ("fetch", "tabstack")])
async def test_a_keyed_provider_needs_a_key(kind: ToolKind, provider: str) -> None:
    with pytest.raises(WebSearchKeyIncompleteError, match="needs an API key"):
        await _web_search_keys._check_settings(kind, provider, has_key=False, api_base=None, options=None)


@pytest.mark.asyncio
async def test_an_endpoint_takes_a_key_only_over_https() -> None:
    with pytest.raises(WebSearchKeySettingRefusedError, match="must use https"):
        await _web_search_keys._check_settings(
            "search", "exa", has_key=True, api_base="http://exa.example", options=None
        )
    with pytest.raises(WebSearchKeySettingRefusedError, match="http or https"):
        await _web_search_keys._check_settings("search", "exa", has_key=True, api_base="exa.example", options=None)

    await _web_search_keys._check_settings("search", "exa", has_key=True, api_base="https://exa.example", options=None)


@pytest.mark.asyncio
async def test_an_internal_endpoint_is_refused_where_the_deployment_gates_them(monkeypatch: pytest.MonkeyPatch) -> None:
    internal = "https://10.0.0.7/exa"
    await _web_search_keys._check_settings("search", "exa", has_key=True, api_base=internal, options=None)

    monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
    with pytest.raises(WebSearchKeySettingRefusedError, match="api_base"):
        await _web_search_keys._check_settings("search", "exa", has_key=True, api_base=internal, options=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["tavily", "brave"])
async def test_a_key_on_the_old_search_code_takes_no_endpoint_or_options(provider: str) -> None:
    with pytest.raises(WebSearchKeySettingRefusedError, match="neither api_base nor options"):
        await _web_search_keys._check_settings(
            "search", provider, has_key=True, api_base="https://search.example", options=None
        )
    with pytest.raises(WebSearchKeySettingRefusedError, match="neither api_base nor options"):
        await _web_search_keys._check_settings(
            "search", provider, has_key=True, api_base=None, options={"topic": "news"}
        )


@pytest.mark.asyncio
async def test_a_keys_options_are_checked_against_its_provider() -> None:
    await _web_search_keys._check_settings("search", "exa", has_key=True, api_base=None, options={"type": "fast"})
    await _web_search_keys._check_settings("fetch", "tabstack", has_key=True, api_base=None, options={"effort": "max"})

    with pytest.raises(WebSearchKeySettingRefusedError, match="not one the provider knows"):
        await _web_search_keys._check_settings("search", "exa", has_key=True, api_base=None, options={"effort": "max"})
    with pytest.raises(WebSearchKeySettingRefusedError, match="a value the provider refuses"):
        await _web_search_keys._check_settings("search", "exa", has_key=True, api_base=None, options={"type": "neural"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("defaults", "reads"),
    [
        ({"search", "fetch"}, ["preferred every kind"]),
        ({"search"}, ["preferred every kind", "page 0 fetch"]),
        (set(), ["preferred every kind", "page 0 every kind"]),
    ],
    ids=["both-defaults", "search-default", "no-default"],
)
async def test_both_kinds_are_chosen_from_one_read_of_the_pins_and_defaults(
    monkeypatch: pytest.MonkeyPatch, defaults: set[str], reads: list[str]
) -> None:
    keys = [
        _key(f"{kind}-key", provider=provider, kind=kind, default=kind in defaults)
        for kind, provider in (("search", "exa"), ("fetch", "tabstack"))
    ]
    monkeypatch.setattr(
        _web_search_keys, "key_credential", lambda key: WebSearchCredential(provider=key.provider, source=_ORG)
    )
    overrides = _Overrides(keys)

    chosen = await _web_search_keys._choose_keys(
        cast(Any, overrides), uuid.uuid4(), uuid.uuid4(), ["search", "fetch"], _web_search_keys._Decrypted()
    )

    assert {kind: key.name if key else None for kind, key in chosen.items()} == {
        "search": "search-key",
        "fetch": "fetch-key",
    }
    assert overrides.reads == reads
