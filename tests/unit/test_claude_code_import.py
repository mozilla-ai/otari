"""Unit tests for reading Claude Code transcripts into importable usage events."""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from gateway.cli import cli
from gateway.services.claude_code_import import (
    normalize_model,
    parse_since,
    provider_for_model,
    scan_transcripts,
    session_label,
)


def _assistant_line(
    response_id: str,
    *,
    model: str = "claude-sonnet-4-5",
    timestamp: str = "2026-07-22T12:34:56.000Z",
    usage: dict[str, Any] | None = None,
) -> str:
    """One assistant line as Claude Code writes it, with only the fields read back."""
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": timestamp,
            "message": {
                "id": response_id,
                "model": model,
                "usage": usage
                if usage is not None
                else {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0},
            },
        }
    )


def _write_transcript(projects_dir: Path, project: str, session: str, lines: list[str]) -> Path:
    path = projects_dir / project / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_repeated_response_ids_are_counted_once(tmp_path: Path) -> None:
    """A reply spans several transcript lines under one response id; that is one API call."""
    _write_transcript(
        tmp_path,
        "-Users-alice-Projects-otari",
        "session-a",
        [_assistant_line("msg_01"), _assistant_line("msg_01"), _assistant_line("msg_02")],
    )

    result = scan_transcripts(tmp_path, label_prefix="host")

    assert [event.source_event_id for event in result.events] == ["msg_01", "msg_02"]
    assert result.duplicates_skipped == 1


def test_a_response_id_repeated_across_sessions_is_still_one_event(tmp_path: Path) -> None:
    """A resumed session re-records earlier replies; the scan is deduplicated globally."""
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-a", [_assistant_line("msg_01")])
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-b", [_assistant_line("msg_01")])

    result = scan_transcripts(tmp_path, label_prefix="host")

    assert len(result.events) == 1
    assert result.files_scanned == 2


def test_cache_writes_split_by_ttl(tmp_path: Path) -> None:
    """The 1h share is priced separately, so it is reported apart from the 5m share."""
    _write_transcript(
        tmp_path,
        "-Users-alice-Projects-otari",
        "session-a",
        [
            _assistant_line(
                "msg_01",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 500,
                    "cache_creation": {"ephemeral_1h_input_tokens": 200, "ephemeral_5m_input_tokens": 300},
                },
            )
        ],
    )

    event = scan_transcripts(tmp_path, label_prefix="host").events[0]

    assert (event.cache_write_tokens, event.cache_write_1h_tokens) == (300, 200)
    assert (event.input_tokens, event.output_tokens, event.cache_read_tokens) == (100, 20, 900)
    assert event.as_payload()["cache_tokens_in_prompt"] is False


def test_a_cache_total_below_its_1h_share_never_goes_negative(tmp_path: Path) -> None:
    """Token counts are unsigned at the endpoint, so an inconsistent transcript clamps to zero."""
    _write_transcript(
        tmp_path,
        "-Users-alice-Projects-otari",
        "session-a",
        [
            _assistant_line(
                "msg_01",
                usage={
                    "cache_creation_input_tokens": 100,
                    "cache_creation": {"ephemeral_1h_input_tokens": 400},
                },
            )
        ],
    )

    event = scan_transcripts(tmp_path, label_prefix="host").events[0]

    assert event.cache_write_tokens == 0


def test_locally_generated_messages_are_not_imported(tmp_path: Path) -> None:
    """Claude Code writes an API error notice itself, with a usage block and no request behind it."""
    _write_transcript(
        tmp_path,
        "-Users-alice-Projects-otari",
        "session-a",
        [
            _assistant_line("msg_01"),
            _assistant_line("msg_syn", model="<synthetic>", usage={"input_tokens": 0, "output_tokens": 0}),
        ],
    )

    result = scan_transcripts(tmp_path, label_prefix="host")

    assert [event.source_event_id for event in result.events] == ["msg_01"]
    assert result.synthetic_skipped == 1
    assert all(event.model != "synthetic" for event in result.events)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-sonnet-4-5[1m]", "claude-sonnet-4-5"),
        ("claude-opus-4-6", "claude-opus-4-6"),
        ("openai/gpt-5-mini", "openai/gpt-5-mini"),
        ("weird model!name", "weird-model-name"),
        ("[1m]", "unknown"),
        (None, "unknown"),
    ],
)
def test_model_ids_are_folded_into_the_endpoint_grammar(raw: str | None, expected: str) -> None:
    """A bracketed context tag names a window, not a priced model, and the grammar rejects it anyway."""
    assert normalize_model(raw) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-6", "anthropic"),
        ("gpt-5-mini", "openai"),
        ("gemini-2-5-pro", "google"),
        ("kimi-k2", "moonshot"),
        ("something-local", "unknown"),
    ],
)
def test_provider_is_admitted_unknown_rather_than_guessed(model: str, expected: str) -> None:
    """An unpriced row is visibly unpriced; one priced as the wrong provider is silently wrong."""
    assert provider_for_model(model) == expected


@pytest.mark.parametrize(
    ("project_dir", "expected"),
    [
        ("-Users-alice-Projects-otari", "box:otari"),
        ("-home-alice-src-my-app", "box:app"),
        ("plain", "box:plain"),
        ("", "box:unknown"),
    ],
)
def test_session_label_uses_the_last_path_segment(project_dir: str, expected: str) -> None:
    """The directory name is a mangled path; only its last segment survives as a label."""
    assert session_label(project_dir, "box") == expected


def test_since_skips_transcripts_by_modification_time(tmp_path: Path) -> None:
    """A run can skip whole files without opening them."""
    old = _write_transcript(tmp_path, "-Users-alice-Projects-old", "session-old", [_assistant_line("msg_old")])
    _write_transcript(tmp_path, "-Users-alice-Projects-new", "session-new", [_assistant_line("msg_new")])
    stale = time.time() - timedelta(days=30).total_seconds()
    os.utime(old, (stale, stale))

    result = scan_transcripts(tmp_path, label_prefix="host", since=datetime.now(timezone.utc) - timedelta(days=1))

    assert [event.source_event_id for event in result.events] == ["msg_new"]
    assert result.files_scanned == 1


def test_lines_without_usage_are_ignored_and_bad_json_is_counted(tmp_path: Path) -> None:
    """A transcript carries user turns and tool results too, and can be truncated mid-write."""
    _write_transcript(
        tmp_path,
        "-Users-alice-Projects-otari",
        "session-a",
        [
            json.dumps({"type": "user", "message": {"role": "user"}}),
            '{"message": {"usage": {"input_tokens": 1}',
            _assistant_line("msg_01"),
        ],
    )

    result = scan_transcripts(tmp_path, label_prefix="host")

    assert len(result.events) == 1
    assert result.unparsable_lines == 1


def test_a_missing_projects_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert scan_transcripts(tmp_path / "absent", label_prefix="host").events == []


@pytest.mark.parametrize(
    ("value", "expected_delta"),
    [("24h", timedelta(hours=24)), ("7d", timedelta(days=7)), ("2w", timedelta(weeks=2))],
)
def test_since_accepts_durations(value: str, expected_delta: timedelta) -> None:
    parsed = parse_since(value)
    assert abs((datetime.now(timezone.utc) - parsed) - expected_delta) < timedelta(seconds=5)


def test_since_accepts_a_bare_date_as_utc_midnight() -> None:
    assert parse_since("2026-07-22") == datetime(2026, 7, 22, tzinfo=timezone.utc)


def test_since_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="neither an ISO-8601"):
        parse_since("last tuesday")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Stands in for httpx.Client, recording what a run would have sent."""

    posted: list[dict[str, Any]] = []
    response = _FakeResponse(200, {"accepted": 1, "duplicate": 0, "rejected": 0, "errors": []})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _FakeResponse:
        type(self).posted.append({"url": url, "body": json, "headers": headers})
        return type(self).response


@pytest.fixture
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    import httpx

    _FakeClient.posted = []
    _FakeClient.response = _FakeResponse(200, {"accepted": 1, "duplicate": 0, "rejected": 0, "errors": []})
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return _FakeClient


def test_dry_run_reports_totals_and_sends_nothing(tmp_path: Path, fake_httpx: type[_FakeClient]) -> None:
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-a", [_assistant_line("msg_01")])

    result = CliRunner().invoke(
        cli,
        ["import", "claude-code", "--projects-dir", str(tmp_path), "--api-key", "gw-test", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "1 event(s)" in result.output
    assert "Dry run: nothing was sent." in result.output
    assert fake_httpx.posted == []


def test_a_run_posts_the_batch_to_the_external_events_endpoint(
    tmp_path: Path, fake_httpx: type[_FakeClient]
) -> None:
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-a", [_assistant_line("msg_01")])

    result = CliRunner().invoke(
        cli,
        [
            "import",
            "claude-code",
            "--projects-dir",
            str(tmp_path),
            "--url",
            "http://gateway.test/",
            "--api-key",
            "gw-test",
            "--user-id",
            "alice",
            "--label-prefix",
            "box",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(fake_httpx.posted) == 1
    sent = fake_httpx.posted[0]
    assert sent["url"] == "http://gateway.test/v1/usage/external-events"
    assert sent["headers"]["Authorization"] == "Bearer gw-test"
    assert sent["body"]["source"] == "claude_code"
    assert sent["body"]["user_id"] == "alice"
    assert sent["body"]["events"][0]["session_label"] == "box:otari"
    assert "prompt" not in sent["body"]["events"][0]


def test_an_empty_scan_sends_nothing(tmp_path: Path, fake_httpx: type[_FakeClient]) -> None:
    result = CliRunner().invoke(
        cli, ["import", "claude-code", "--projects-dir", str(tmp_path), "--api-key", "gw-test"]
    )

    assert result.exit_code == 0
    assert "Nothing to import." in result.output
    assert fake_httpx.posted == []


def test_a_refused_batch_exits_non_zero(tmp_path: Path, fake_httpx: type[_FakeClient]) -> None:
    """A silent success on a refused import would leave a gap nobody goes looking for."""
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-a", [_assistant_line("msg_01")])
    fake_httpx.response = _FakeResponse(403, {"detail": "key is not budget-exempt"})

    result = CliRunner().invoke(
        cli, ["import", "claude-code", "--projects-dir", str(tmp_path), "--api-key", "gw-test"]
    )

    assert result.exit_code == 1
    assert "403" in result.output
    assert "key is not budget-exempt" in result.output


def test_events_rejected_individually_also_exit_non_zero(
    tmp_path: Path, fake_httpx: type[_FakeClient]
) -> None:
    _write_transcript(tmp_path, "-Users-alice-Projects-otari", "session-a", [_assistant_line("msg_01")])
    fake_httpx.response = _FakeResponse(
        200,
        {"accepted": 0, "duplicate": 0, "rejected": 1, "errors": [{"index": 0, "detail": "no price"}]},
    )

    result = CliRunner().invoke(
        cli, ["import", "claude-code", "--projects-dir", str(tmp_path), "--api-key", "gw-test"]
    )

    assert result.exit_code == 1
    assert "no price" in result.output


def test_a_bad_since_is_a_usage_error(tmp_path: Path, fake_httpx: type[_FakeClient]) -> None:
    result = CliRunner().invoke(
        cli,
        ["import", "claude-code", "--projects-dir", str(tmp_path), "--api-key", "gw-test", "--since", "yesterday"],
    )

    assert result.exit_code == 2
    assert "--since" in result.output
