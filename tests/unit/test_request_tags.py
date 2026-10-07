"""Request tags read from a request's ``metadata``."""

import pytest

from gateway.api.routes._request_tags import (
    ANTHROPIC_USER_KEY,
    MAX_REQUEST_TAGS,
    MAX_TAG_KEY_LENGTH,
    MAX_TAG_VALUE_LENGTH,
    anthropic_metadata,
    forwarded_metadata,
    request_tags,
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
    "metadata",
    [
        {"purpose": 1},
        {"purpose": ["chat"]},
        {"spend_logs_metadata": {"purpose": {"nested": "deeper"}}},
        {"spend_logs_metadata": "chat"},
        {"": "chat"},
        {"k" * (MAX_TAG_KEY_LENGTH + 1): "chat"},
        {"purpose": "v" * (MAX_TAG_VALUE_LENGTH + 1)},
        {f"k{i}": "v" for i in range(MAX_REQUEST_TAGS + 1)},
    ],
)
def test_anything_but_bounded_string_tags_is_refused(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        request_tags(metadata)


def test_the_bound_counts_tags_after_merging() -> None:
    flat = {f"k{i}": "v" for i in range(MAX_REQUEST_TAGS)}
    assert request_tags({**flat, "spend_logs_metadata": {"k0": "w"}}) is not None
    with pytest.raises(ValueError):
        request_tags({**flat, "spend_logs_metadata": {"extra": "w"}})


def test_ignored_keys_are_neither_read_nor_checked() -> None:
    ignore = frozenset({ANTHROPIC_USER_KEY})
    assert request_tags({"user_id": 42, "purpose": "chat"}, ignore=ignore) == {"purpose": "chat"}
    assert request_tags({"user_id": "u1"}, ignore=ignore) is None


def test_anthropic_metadata_keeps_only_user_id() -> None:
    fields = {"metadata": {"user_id": "u1", "purpose": "chat", "spend_logs_metadata": {"country": "DE"}}}
    assert anthropic_metadata(fields) == {"metadata": {"user_id": "u1"}}
    assert anthropic_metadata({"metadata": {"purpose": "chat"}, "model": "m"}) == {"model": "m"}


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
