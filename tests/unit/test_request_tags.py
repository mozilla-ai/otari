"""Request tags read from a request's ``metadata`` and its ``Otari-Tags`` header."""

import pytest
from fastapi.exceptions import RequestValidationError

from gateway.api.routes._request_tags import (
    ANTHROPIC_USER_KEY,
    MAX_REQUEST_TAGS,
    MAX_TAG_KEY_LENGTH,
    MAX_TAG_VALUE_LENGTH,
    forwarded_metadata,
    header_tags,
    merge_tags,
    request_tags,
    resolved_tags,
)


def test_flat_metadata_is_the_tags() -> None:
    assert request_tags({"purpose": "chat", "country": "DE"}) == {"purpose": "chat", "country": "DE"}


def test_litellm_spend_logs_metadata_is_read_and_wins_over_a_flat_key() -> None:
    metadata = {"purpose": "flat", "spend_logs_metadata": {"purpose": "chat", "country_code": "DE"}}
    assert request_tags(metadata) == {"purpose": "chat", "country_code": "DE"}


@pytest.mark.parametrize("metadata", [None, {}, {"purpose": None}, {"spend_logs_metadata": None}])
def test_no_tags_reads_as_none(metadata: dict[str, object] | None) -> None:
    assert request_tags(metadata) is None


def test_null_values_are_dropped() -> None:
    assert request_tags({"spend_logs_metadata": {"purpose": "chat", "country_code": None}}) == {"purpose": "chat"}


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"purpose": 1}, "must be a string"),
        ({"purpose": ["chat"]}, "must be a string"),
        ({"spend_logs_metadata": {"purpose": {"nested": "deeper"}}}, "must be a string"),
        ({"spend_logs_metadata": "chat"}, "must be an object"),
        ({"": "chat"}, "tag key must be"),
        ({"k" * (MAX_TAG_KEY_LENGTH + 1): "chat"}, "tag key must be"),
        ({"purpose": "v" * (MAX_TAG_VALUE_LENGTH + 1)}, f"at most {MAX_TAG_VALUE_LENGTH} characters"),
        ({f"k{i}": "v" for i in range(MAX_REQUEST_TAGS + 1)}, f"at most {MAX_REQUEST_TAGS} tags"),
    ],
)
def test_anything_but_bounded_string_tags_is_refused(metadata: dict[str, object], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        request_tags(metadata)


def test_the_bound_counts_tags_after_merging() -> None:
    flat = {f"k{i}": "v" for i in range(MAX_REQUEST_TAGS)}
    assert request_tags({**flat, "spend_logs_metadata": {"k0": "w"}}) is not None
    with pytest.raises(ValueError, match=f"at most {MAX_REQUEST_TAGS} tags"):
        request_tags({**flat, "spend_logs_metadata": {"extra": "w"}})


def test_ignored_keys_are_neither_read_nor_checked() -> None:
    ignore = frozenset({ANTHROPIC_USER_KEY})
    assert request_tags({"user_id": 42, "purpose": "chat"}, ignore=ignore) == {"purpose": "chat"}
    assert request_tags({"user_id": "u1"}, ignore=ignore) is None


def test_forwarded_metadata_drops_only_the_litellm_key() -> None:
    fields = {"metadata": {"user_id": "u1", "spend_logs_metadata": {"purpose": "chat"}}}
    assert forwarded_metadata(fields) == {"metadata": {"user_id": "u1"}}


def test_forwarded_metadata_removes_a_field_left_empty() -> None:
    assert forwarded_metadata({"metadata": {"spend_logs_metadata": {"purpose": "chat"}}, "model": "m"}) == {
        "model": "m"
    }


def test_forwarded_metadata_leaves_plain_metadata_alone() -> None:
    fields = {"metadata": {"purpose": "chat"}}
    assert forwarded_metadata(fields) == {"metadata": {"purpose": "chat"}}


def test_header_pairs_are_trimmed_and_percent_decoded() -> None:
    assert header_tags(" agent=bug-fix , run=1234,note=a%2Cb%3Dc ,") == {
        "agent": "bug-fix",
        "run": "1234",
        "note": "a,b=c",
    }


def test_a_non_ascii_value_is_read_percent_encoded() -> None:
    assert header_tags("agent=caf%C3%A9") == {"agent": "caf\u00e9"}


def test_a_header_value_keeps_everything_after_the_first_equals() -> None:
    assert header_tags("expr=a=b") == {"expr": "a=b"}
    assert header_tags("empty=") == {"empty": ""}


@pytest.mark.parametrize("value", [None, "", "  ", ",", " , "])
def test_an_empty_header_has_no_tags(value: str | None) -> None:
    assert header_tags(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "agent",
        "agent=a,run",
        "=value",
        "agent=a,agent=b",
        "agent=%FF",
        "agent=bug%ZZ",
        "agent=50%",
        "ag%2=v",
        "agent=caf\u00c3\u00a9",
        f"{'k' * (MAX_TAG_KEY_LENGTH + 1)}=v",
        f"agent={'v' * (MAX_TAG_VALUE_LENGTH + 1)}",
        ",".join(f"k{i}=v" for i in range(MAX_REQUEST_TAGS + 1)),
    ],
)
def test_a_malformed_header_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="Otari-Tags"):
        header_tags(value)


def test_a_later_layer_wins_on_a_key() -> None:
    assert merge_tags({"agent": "header", "run": "1"}, None, {"agent": "body"}) == {"agent": "body", "run": "1"}
    assert merge_tags(None, None) is None


def test_body_tags_override_header_tags() -> None:
    tags = resolved_tags("agent=bug-fix,run=1234", {"agent": "triage", "user_id": "u1"}, ignore=frozenset({"user_id"}))
    assert tags == {"agent": "triage", "run": "1234"}


def test_the_bound_counts_header_and_body_together() -> None:
    header = ",".join(f"h{i}=v" for i in range(MAX_REQUEST_TAGS // 2))
    body = {f"b{i}": "v" for i in range(MAX_REQUEST_TAGS // 2)}
    assert resolved_tags(header, body) is not None
    # An overlapping key counts once.
    assert resolved_tags(header, {**body, "h0": "w"}) is not None
    with pytest.raises(RequestValidationError) as exc_info:
        resolved_tags(header, {**body, "extra": "w"})
    assert exc_info.value.errors()[0]["loc"] == ("header", "Otari-Tags")
