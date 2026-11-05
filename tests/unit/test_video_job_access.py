"""Who may read a video job: its owner, in the workspace it was started in, and the master key."""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.exceptions.inference_exceptions import VideoJobNotFoundError
from gateway.services.inference import VideoJobService

_WORKSPACE = uuid.uuid4()


def _record(**overrides: Any) -> Any:
    fields: dict[str, Any] = {"user_id": "owner", "workspace_id": _WORKSPACE}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _read(record: Any, *, user_id: str | None = "owner", workspace_id: uuid.UUID | None = _WORKSPACE) -> Any:
    return VideoJobService.authorize(record, "video_1", user_id=user_id, workspace_id=workspace_id, is_master_key=False)


def test_the_owner_reads_a_job_from_the_workspace_it_was_started_in() -> None:
    record = _record()

    assert _read(record) is record


def test_the_master_key_reads_any_job() -> None:
    record = _record(workspace_id=None)

    assert VideoJobService.authorize(record, "video_1", user_id=None, workspace_id=None, is_master_key=True) is record


@pytest.mark.parametrize(
    ("record", "reader"),
    [
        (None, {}),
        (_record(), {"user_id": "someone-else"}),
        (_record(), {"workspace_id": uuid.uuid4()}),
        (_record(workspace_id=None), {}),
        (_record(workspace_id=None), {"workspace_id": None}),
    ],
    ids=["unknown id", "another user", "another workspace", "workspace deleted", "no workspace on either side"],
)
def test_an_unknown_foreign_or_orphaned_job_is_not_found(record: Any, reader: dict[str, Any]) -> None:
    with pytest.raises(VideoJobNotFoundError) as refused:
        _read(record, **reader)

    assert refused.value.status_code == 404
    assert refused.value.message == "Video 'video_1' not found"
