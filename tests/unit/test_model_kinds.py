"""Paging a provider's offered models by kind."""

from dataclasses import dataclass

from gateway.services.providers._model_kinds import page_by_kind
from gateway.types.model_type import ModelType


@dataclass
class _Row:
    model: str


ROWS = [_Row("dall-e-3"), _Row("gpt-4o"), _Row("gpt-4o-mini"), _Row("mystery")]
KINDS: dict[str, ModelType | None] = {"dall-e-3": "image", "gpt-4o": "chat", "gpt-4o-mini": "chat", "mystery": None}


def test_every_row_counts_toward_its_kind_and_a_kind_nothing_names_is_unknown() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type=None, skip=0, limit=10)

    assert [row.model for row in page.rows] == ["dall-e-3", "gpt-4o", "gpt-4o-mini", "mystery"]
    assert page.count == 4
    assert page.type_counts == {"image": 1, "chat": 2, "unknown": 1}


def test_a_kind_narrows_the_page_and_its_total_but_not_the_counts() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type="chat", skip=1, limit=10)

    assert [row.model for row in page.rows] == ["gpt-4o-mini"]
    assert page.count == 2
    assert page.type_counts == {"image": 1, "chat": 2, "unknown": 1}


def test_unknown_is_a_kind_a_client_can_ask_for() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type="unknown", skip=0, limit=10)

    assert [row.model for row in page.rows] == ["mystery"]


def test_a_kind_no_row_has_is_an_empty_page() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type="video", skip=0, limit=10)

    assert (page.rows, page.count) == ([], 0)


def test_a_search_narrows_the_rows_and_the_counts_by_name_case_insensitively() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type=None, skip=0, limit=10, search=" GPT-4O ")

    assert [row.model for row in page.rows] == ["gpt-4o", "gpt-4o-mini"]
    assert page.count == 2
    assert page.type_counts == {"chat": 2}


def test_a_search_and_a_kind_narrow_together() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type="image", skip=0, limit=10, search="gpt")

    assert (page.rows, page.count) == ([], 0)
    assert page.type_counts == {"chat": 2}


def test_a_blank_search_keeps_every_row() -> None:
    page = page_by_kind(ROWS, KINDS, "model", model_type=None, skip=0, limit=10, search="  ")

    assert page.count == 4
