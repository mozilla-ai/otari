"""Memory's pure rules: how it scores, what it refuses to keep, and how it reads a consolidation answer."""

import pytest

from gateway.services.memory._models import Decision, parse_decision
from gateway.services.memory._scoring import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    INITIAL_CONFIDENCE,
    confidence,
    cosine,
    relevance,
)
from gateway.services.memory._screening import screen


def test_confidence_follows_cqs_rules() -> None:
    assert confidence(0, 0) == INITIAL_CONFIDENCE
    assert confidence(1, 0) == pytest.approx(0.6)
    assert confidence(0, 1) == pytest.approx(0.35)
    assert confidence(2, 1) == pytest.approx(0.55)


def test_confidence_stays_between_its_floor_and_ceiling() -> None:
    assert confidence(50, 0) == CONFIDENCE_CEILING
    assert confidence(0, 50) == CONFIDENCE_FLOOR


def test_cosine_of_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(("a", "b"), [([1.0], [1.0, 2.0]), ([], []), ([0.0, 0.0], [1.0, 1.0])])
def test_cosine_of_vectors_it_cannot_compare_is_zero(a: list[float], b: list[float]) -> None:
    assert cosine(a, b) == 0.0


def test_a_dissimilar_entry_ranks_low_however_trusted() -> None:
    assert relevance(-0.5, 1.0) == 0.0
    assert relevance(0.9, 0.5) > relevance(0.2, 1.0)


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("Use the key sk-proj-abcdefghijklmnop1234 for staging", "credential"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload", "credential"),
        ("set password=hunter2hunter2 in the env file", "credential"),
        ('{"api_key": "abcdef123456"}', "credential"),
        ("AKIAIOSFODNN7EXAMPLE is the deploy user's key", "credential"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nabc", "private_key"),
        ("Ask maria.lopez@example.com before deploying", "email_address"),
        ("The test card 4111 1111 1111 1111 always passes", "card_number"),
    ],
)
def test_what_memory_does_not_keep_is_found(text: str, finding: str) -> None:
    assert screen(text) == finding


@pytest.mark.parametrize(
    "text",
    [
        "Run migrations with make migrate before the integration tests.",
        "The staging database is reached through the bastion host, never directly.",
        "Order 1234 5678 9012 3456 is not a card number because it fails Luhn.",
        "Key rotation happens on the first Monday of each month.",
        "password rotation is handled by the platform team",
    ],
)
def test_ordinary_knowledge_passes(text: str) -> None:
    assert screen(text) is None


@pytest.mark.parametrize(
    ("answer", "decision"),
    [
        ('{"action": "add", "target": null}', Decision("add")),
        ('{"action": "update", "target": 2}', Decision("update", 1)),
        ('Sure: {"action": "no_change", "target": 1} done', Decision("no_change", 0)),
        ('{"action": "conflict", "target": 3}', Decision("conflict", 2)),
    ],
)
def test_a_consolidation_answer_is_read(answer: str, decision: Decision) -> None:
    assert parse_decision(answer, existing=3) == decision


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "add it",
        '{"action": "delete_everything"}',
        '{"action": "update"}',
        '{"action": "update", "target": 0}',
        '{"action": "update", "target": 9}',
        '{"action": "update", "target": true}',
        '{"action": "update", "target": "1"}',
        "[1, 2]",
    ],
)
def test_an_answer_this_cannot_act_on_waits_for_an_admin(answer: str) -> None:
    assert parse_decision(answer, existing=3) == Decision("conflict")
