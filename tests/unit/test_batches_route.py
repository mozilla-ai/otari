"""Unit tests for batch route Pydantic request models and helpers."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.api.routes.batches import (
    BatchRequestItem,
    CreateBatchRequest,
    _authorize_legacy_batch,
    _authorize_record,
    _lifecycle_workspace_id,
)


class TestBatchRequestItem:
    def test_valid_item(self) -> None:
        item = BatchRequestItem(
            custom_id="req-1",
            body={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
        )
        assert item.custom_id == "req-1"
        assert "messages" in item.body

    def test_missing_custom_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="custom_id"):
            BatchRequestItem(body={"messages": []})  # type: ignore[call-arg]


class TestCreateBatchRequest:
    def test_valid_request(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[
                BatchRequestItem(
                    custom_id="req-1",
                    body={"messages": [{"role": "user", "content": "Hello"}]},
                ),
            ],
        )
        assert request.model == "openai:gpt-4o-mini"
        assert len(request.requests) == 1
        assert request.completion_window == "24h"
        assert request.metadata is None

    def test_empty_requests_rejected(self) -> None:
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=[])

    def test_too_many_requests_rejected(self) -> None:
        items = [BatchRequestItem(custom_id=f"req-{i}", body={}) for i in range(10_001)]
        with pytest.raises(ValidationError, match="List should have at most 10000 items"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=items)

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            CreateBatchRequest(
                requests=[BatchRequestItem(custom_id="req-1", body={})],
            )  # type: ignore[call-arg]

    def test_optional_metadata(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            metadata={"team": "ml-ops"},
        )
        assert request.metadata == {"team": "ml-ops"}

    def test_custom_completion_window(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            completion_window="48h",
        )
        assert request.completion_window == "48h"


class TestLifecycleWorkspaceId:
    """`_lifecycle_workspace_id` picks the batch's own workspace over the
    caller's, per the CodeRabbit finding on otari#643: a master-key or
    cross-workspace retrieve/cancel/results must resolve credentials from the
    organization that created the batch, not the retriever's own workspace.
    """

    @pytest.mark.asyncio
    async def test_uses_the_batch_records_own_workspace_when_present(self) -> None:
        creating_workspace_id = uuid.uuid4()
        record = SimpleNamespace(workspace_id=creating_workspace_id)

        resolved = await _lifecycle_workspace_id(db=AsyncMock(), record=record, api_key=None)  # type: ignore[arg-type]

        assert resolved == creating_workspace_id

    @pytest.mark.asyncio
    async def test_falls_back_to_the_callers_workspace_for_a_legacy_batch_with_no_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        caller_workspace_id = uuid.uuid4()
        monkeypatch.setattr(
            "gateway.api.routes.batches.resolve_workspace_id",
            AsyncMock(return_value=caller_workspace_id),
        )

        resolved = await _lifecycle_workspace_id(db=AsyncMock(), record=None, api_key=None)

        assert resolved == caller_workspace_id

    @pytest.mark.asyncio
    async def test_falls_back_to_the_callers_workspace_when_the_record_predates_the_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record persisted before this migration has ``workspace_id is None``,
        the same as a legacy record-less batch, not the creating workspace.
        """
        caller_workspace_id = uuid.uuid4()
        monkeypatch.setattr(
            "gateway.api.routes.batches.resolve_workspace_id",
            AsyncMock(return_value=caller_workspace_id),
        )
        record = SimpleNamespace(workspace_id=None)

        resolved = await _lifecycle_workspace_id(db=AsyncMock(), record=record, api_key=None)  # type: ignore[arg-type]

        assert resolved == caller_workspace_id


class TestAuthorizeRecord:
    """`_authorize_record` runs *before* the batch's originating-organization
    credential is used to call the provider (see `_lifecycle_workspace_id`),
    so an unauthorized caller for another organization's batch is refused
    before that organization's credential is ever spent. It therefore takes
    no `Batch`, deliberately: needing one would force the credential-bearing
    dispatch to happen first."""

    def test_master_key_is_always_authorized(self) -> None:
        record = SimpleNamespace(user_id="someone-else")
        _authorize_record(record, "batch-1", api_key=None, is_master_key=True)  # type: ignore[arg-type]

    def test_the_records_own_user_is_authorized(self) -> None:
        record = SimpleNamespace(user_id="user-1")
        api_key = SimpleNamespace(user_id="user-1")
        _authorize_record(record, "batch-1", api_key=api_key, is_master_key=False)  # type: ignore[arg-type]

    def test_a_different_user_is_refused_with_404_not_403(self) -> None:
        """404, not 403: a foreign key must not be able to probe which batch
        ids exist by distinguishing "not yours" from "no such batch"."""
        record = SimpleNamespace(user_id="the-owner")
        api_key = SimpleNamespace(user_id="someone-else")

        with pytest.raises(HTTPException) as exc_info:
            _authorize_record(record, "batch-1", api_key=api_key, is_master_key=False)  # type: ignore[arg-type]

        assert exc_info.value.status_code == 404

    def test_no_api_key_is_refused(self) -> None:
        record = SimpleNamespace(user_id="the-owner")

        with pytest.raises(HTTPException) as exc_info:
            _authorize_record(record, "batch-1", api_key=None, is_master_key=False)  # type: ignore[arg-type]

        assert exc_info.value.status_code == 404


class TestAuthorizeLegacyBatch:
    """The metadata-anchored fallback for a record-less batch. Only reachable
    once the batch has already been fetched from the provider, which
    `_authorize_record`'s docstring explains is safe only in this case: a
    record-less batch has no stored workspace to leak, so credentials there
    already resolved to the caller's own."""

    def test_master_key_is_always_authorized(self) -> None:
        batch = SimpleNamespace(metadata={"otari_user_id": "someone-else"})
        _authorize_legacy_batch(batch, "batch-1", api_key=None, is_master_key=True)  # type: ignore[arg-type]

    def test_a_batch_with_no_marker_is_authorized_for_any_caller(self) -> None:
        """Batches predating the ownership marker, or from a provider that
        does not round-trip metadata, stay reachable by any authenticated key."""
        batch = SimpleNamespace(metadata=None)
        api_key = SimpleNamespace(user_id="anyone")
        _authorize_legacy_batch(batch, "batch-1", api_key=api_key, is_master_key=False)  # type: ignore[arg-type]

    def test_a_mismatched_marker_is_refused_with_404(self) -> None:
        batch = SimpleNamespace(metadata={"otari_user_id": "the-owner"})
        api_key = SimpleNamespace(user_id="someone-else")

        with pytest.raises(HTTPException) as exc_info:
            _authorize_legacy_batch(batch, "batch-1", api_key=api_key, is_master_key=False)  # type: ignore[arg-type]

        assert exc_info.value.status_code == 404
