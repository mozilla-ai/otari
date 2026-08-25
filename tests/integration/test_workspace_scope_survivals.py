"""The gateway survivals are scoped to a workspace (otari-ai#1643).

Aliases, routing policies, routing memory, router preferences, files and batches
stay in the gateway rather than moving to the platform, and each is now managed
and resolved inside one workspace. ``user_id`` is untouched on every one of
them: the workspace is a second axis, so a user who holds keys in two workspaces
keeps their per-user scoping within each and reaches neither from the other.

Two properties recur and are what most of these tests are about. A write that
names no workspace lands in the deployment's default one, which is what keeps a
single-workspace deployment working with nothing to change. And the same name in
two workspaces resolves to each workspace's own row, which is the behavior the
widened uniqueness constraint exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from gateway.services.file_store import LocalDirFileStore

MASTER = "master"
ALIASES = "/v1/aliases"
POLICIES = "/v1/routing/policies"
NOWHERE = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    """Point the app's blob store at a temp dir (default writes to cwd)."""
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


def _default_workspace(client: TestClient, headers: dict[str, str]) -> str:
    context = client.get("/v1/organizations/me", headers=headers).json()
    return str(context["workspace_memberships"][0]["workspace_id"])


def _make_workspace(client: TestClient, headers: dict[str, str], name: str) -> str:
    created = client.post("/v1/workspaces", json={"name": name}, headers=headers)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    return str(created.json()["id"])


def _key_header(
    client: TestClient, master: dict[str, str], template: dict[str, str], **body: Any
) -> dict[str, str]:
    """Mint a key and return an auth header for it.

    The gateway requires a "Bearer " prefix on every header form, so the header
    name is taken from the fixture's own rather than hardcoded.
    """
    created = client.post("/v1/keys", json={"key_name": "k", **body}, headers=master)
    assert created.status_code == status.HTTP_200_OK, created.text
    return {next(iter(template)): f"Bearer {created.json()['key']}"}


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_two_workspaces_can_each_define_the_same_alias(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The row the pre-widening constraint refused outright.

    Storing both was a 409 while the resolution cache was keyed on name alone,
    because the second would have silently shadowed the first at request time.
    """
    default = _default_workspace(client, master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")

    for workspace, target in ((default, "anthropic:claude-haiku-4"), (platform, "openai:gpt-4o-mini")):
        created = client.post(
            ALIASES,
            json={"name": "fast", "target": target, "workspace_id": workspace},
            headers=master_key_header,
        )
        assert created.status_code == status.HTTP_200_OK, created.text
        assert created.json()["workspace_id"] == workspace

    listed = client.get(f"{ALIASES}?workspace_id={platform}", headers=master_key_header).json()
    stored = [row for row in listed if row["source"] == "stored"]
    assert [(row["name"], row["target"]) for row in stored] == [("fast", "openai:gpt-4o-mini")]


def test_an_alias_resolves_only_for_a_key_in_its_own_workspace(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """Resolution, not just listing: the catalog reads the caller's workspace.

    ``owned_by`` is the observable. An alias resolves to an otari-owned entry;
    a name that is not an alias in the caller's workspace is not a model either,
    so it 404s exactly as an unknown one does.
    """
    platform = _make_workspace(client, master_key_header, "Platform team")
    client.post(
        ALIASES,
        json={"name": "fast", "target": "anthropic:claude-haiku-4", "workspace_id": platform},
        headers=master_key_header,
    )
    in_platform = _key_header(client, master_key_header, api_key_header, workspace_id=platform)
    in_default = _key_header(client, master_key_header, api_key_header, key_name="elsewhere")

    resolved = client.get("/v1/models/fast", headers=in_platform)
    assert resolved.status_code == status.HTTP_200_OK, resolved.text
    assert resolved.json()["owned_by"] == "otari"

    unresolved = client.get("/v1/models/fast", headers=in_default)
    assert unresolved.status_code == status.HTTP_404_NOT_FOUND


def test_an_alias_written_without_a_workspace_lands_in_the_default_one(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """What a single-workspace deployment does, unchanged by any of this."""
    default = _default_workspace(client, master_key_header)

    created = client.post(
        ALIASES, json={"name": "fast", "target": "anthropic:claude-haiku-4"}, headers=master_key_header
    )

    assert created.json()["workspace_id"] == default


def test_deleting_an_alias_leaves_the_other_workspaces_copy(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    default = _default_workspace(client, master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")
    for workspace in (default, platform):
        client.post(
            ALIASES,
            json={"name": "fast", "target": "anthropic:claude-haiku-4", "workspace_id": workspace},
            headers=master_key_header,
        )

    removed = client.delete(f"{ALIASES}/fast?workspace_id={platform}", headers=master_key_header)
    assert removed.status_code == status.HTTP_204_NO_CONTENT

    remaining = client.get(ALIASES, headers=master_key_header).json()
    assert [row["workspace_id"] for row in remaining if row["source"] == "stored"] == [default]


def test_deleting_an_alias_in_the_wrong_workspace_is_a_404(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The error path of the scoped delete: right name, wrong tenant."""
    platform = _make_workspace(client, master_key_header, "Platform team")
    client.post(
        ALIASES, json={"name": "fast", "target": "anthropic:claude-haiku-4"}, headers=master_key_header
    )

    missing = client.delete(f"{ALIASES}/fast?workspace_id={platform}", headers=master_key_header)

    assert missing.status_code == status.HTTP_404_NOT_FOUND
    assert platform in missing.json()["detail"]


def test_an_alias_cannot_be_written_into_a_workspace_that_does_not_exist(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Checked rather than left to the foreign key, matching POST /v1/keys.

    The id comes from the caller, so an unknown one is a bad request; reaching
    the constraint would answer 500 "Database error" for a value they can fix.
    """
    refused = client.post(
        ALIASES,
        json={"name": "fast", "target": "anthropic:claude-haiku-4", "workspace_id": NOWHERE},
        headers=master_key_header,
    )

    assert refused.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in refused.json()["detail"]


# ---------------------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------------------


def _policy(target: str) -> dict[str, Any]:
    return {"select": [{"default": target}]}


def test_two_workspaces_can_each_define_the_same_policy(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    default = _default_workspace(client, master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")

    for workspace, target in ((default, "anthropic:claude-haiku-4"), (platform, "openai:gpt-4o-mini")):
        created = client.post(
            POLICIES,
            json={"name": "cheap", "spec": _policy(target), "workspace_id": workspace},
            headers=master_key_header,
        )
        assert created.status_code == status.HTTP_200_OK, created.text
        assert created.json()["workspace_id"] == workspace

    scoped = client.get(f"{POLICIES}?workspace_id={platform}", headers=master_key_header).json()
    stored = [row for row in scoped if row["source"] == "stored"]
    assert [row["spec"]["select"][0]["default"] for row in stored] == ["openai:gpt-4o-mini"]


def test_one_workspace_still_cannot_hold_two_policies_of_a_name(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Widening admits the cross-workspace row, not the duplicate within one.

    Sent as a rename onto a taken name, which is the path that checks rather
    than upserting; a plain second POST of the same name is an edit by design.
    """
    client.post(POLICIES, json={"name": "cheap", "spec": _policy("openai:gpt-4o-mini")}, headers=master_key_header)
    client.post(POLICIES, json={"name": "fast", "spec": _policy("openai:gpt-4o")}, headers=master_key_header)

    clash = client.post(
        POLICIES,
        json={"name": "cheap", "spec": _policy("openai:gpt-4o"), "rename_from": "fast"},
        headers=master_key_header,
    )

    assert clash.status_code == status.HTTP_409_CONFLICT


def test_explain_resolves_the_policy_in_the_workspace_it_is_asked_about(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Resolution again, on the surface that reports what a policy would do."""
    platform = _make_workspace(client, master_key_header, "Platform team")
    client.post(
        POLICIES,
        json={"name": "cheap", "spec": _policy("openai:gpt-4o-mini"), "workspace_id": platform},
        headers=master_key_header,
    )

    there = client.post(
        f"{POLICIES}/explain", json={"name": "cheap", "workspace_id": platform}, headers=master_key_header
    )
    assert there.status_code == status.HTTP_200_OK, there.text
    assert [c["model"] for c in there.json()["candidates"]] == ["gpt-4o-mini"]

    # The default workspace holds no such policy, so there is nothing to explain.
    here = client.post(f"{POLICIES}/explain", json={"name": "cheap"}, headers=master_key_header)
    assert here.status_code == status.HTTP_404_NOT_FOUND


def test_deleting_a_policy_in_the_wrong_workspace_is_a_404(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    platform = _make_workspace(client, master_key_header, "Platform team")
    client.post(POLICIES, json={"name": "cheap", "spec": _policy("openai:gpt-4o")}, headers=master_key_header)

    missing = client.delete(f"{POLICIES}/cheap?workspace_id={platform}", headers=master_key_header)

    assert missing.status_code == status.HTTP_404_NOT_FOUND
    assert platform in missing.json()["detail"]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_a_file_is_stamped_with_the_uploading_keys_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    """Read off the key, never a header: the caller controls one and not the other."""
    platform = _make_workspace(client, master_key_header, "Platform team")
    scoped = _key_header(
        client, master_key_header, api_key_header, workspace_id=platform, user_id="ada"
    )

    uploaded = client.post("/v1/files", headers=scoped, files={"file": ("a.txt", b"hi", "text/plain")})
    assert uploaded.status_code == status.HTTP_200_OK, uploaded.text

    listed = client.get(f"/v1/files?user=ada&workspace_id={platform}", headers=master_key_header).json()
    assert [row["id"] for row in listed["data"]] == [uploaded.json()["id"]]


def test_the_same_user_cannot_reach_their_file_from_another_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    """The isolation the user check alone does not give.

    Both keys belong to Ada, so per-user scoping says yes; the workspace is what
    says no. 404 rather than 403, matching the cross-user answer, so the two are
    indistinguishable from a missing file.
    """
    platform = _make_workspace(client, master_key_header, "Platform team")
    there = _key_header(client, master_key_header, api_key_header, workspace_id=platform, user_id="ada")
    here = _key_header(client, master_key_header, api_key_header, user_id="ada", key_name="here")

    file_id = client.post(
        "/v1/files", headers=there, files={"file": ("a.txt", b"hi", "text/plain")}
    ).json()["id"]

    assert client.get(f"/v1/files/{file_id}", headers=here).status_code == status.HTTP_404_NOT_FOUND
    assert client.get(f"/v1/files/{file_id}/content", headers=here).status_code == status.HTTP_404_NOT_FOUND
    assert client.delete(f"/v1/files/{file_id}", headers=here).status_code == status.HTTP_404_NOT_FOUND
    assert client.get("/v1/files", headers=here).json()["data"] == []
    # The owning workspace still has it, so nothing was lost, only partitioned.
    assert client.get(f"/v1/files/{file_id}", headers=there).status_code == status.HTTP_200_OK


def test_the_master_key_still_sees_every_workspaces_files(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    """The operator acting deployment-wide, which is what keeps their tooling working.

    Narrowable with ``workspace_id`` rather than narrowed by default, matching
    ``GET /v1/keys``.
    """
    platform = _make_workspace(client, master_key_header, "Platform team")
    there = _key_header(client, master_key_header, api_key_header, workspace_id=platform, user_id="ada")
    here = _key_header(client, master_key_header, api_key_header, user_id="ada", key_name="here")
    for header in (there, here):
        client.post("/v1/files", headers=header, files={"file": ("a.txt", b"hi", "text/plain")})

    everything = client.get("/v1/files?user=ada", headers=master_key_header).json()
    assert len(everything["data"]) == 2

    narrowed = client.get(f"/v1/files?user=ada&workspace_id={platform}", headers=master_key_header).json()
    assert len(narrowed["data"]) == 1


# ---------------------------------------------------------------------------
# Routing memory and router preferences
# ---------------------------------------------------------------------------


def _rank(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    return client.post("/v1/routing/preferences/rank", json=body, headers=headers)


def test_routing_memory_is_counted_per_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmth is per partition, and the partition now includes the workspace.

    A total spanning workspaces would report a pool warm that no single request
    ever votes over, because a decision loads one workspace's records.
    """
    from gateway.services.routing import knn

    async def _embed(self: Any, text: str) -> list[float]:
        return [1.0, 0.0]

    monkeypatch.setattr(knn.KnnRoutingMemory, "_embed", _embed)

    client.post("/v1/users", json={"user_id": "ada"}, headers=master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")

    written = _rank(
        client,
        master_key_header,
        user_id="ada",
        workspace_id=platform,
        examples=[{"prompt": "sum this", "scores": {"openai:gpt-4o-mini": 1.0}}],
    )
    assert written.status_code == status.HTTP_200_OK, written.text

    there = client.get(
        f"/v1/routing/status?user_id=ada&workspace_id={platform}", headers=master_key_header
    ).json()
    assert there["workspace_id"] == platform
    assert there["default_pool"]["records"] == 1

    # The default workspace was taught nothing, so its pool is empty even though
    # the same user has an example elsewhere.
    here = client.get("/v1/routing/status?user_id=ada", headers=master_key_header).json()
    assert here["default_pool"]["records"] == 0


def test_teaching_a_workspace_that_does_not_exist_is_a_404(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    client.post("/v1/users", json={"user_id": "ada"}, headers=master_key_header)

    refused = _rank(
        client,
        master_key_header,
        user_id="ada",
        workspace_id=NOWHERE,
        examples=[{"prompt": "sum this", "scores": {"openai:gpt-4o-mini": 1.0}}],
    )

    assert refused.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in refused.json()["detail"]
