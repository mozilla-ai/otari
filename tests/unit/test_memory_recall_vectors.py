"""The embeddings automatic recall holds: by workspace, for a while, within a cap, and dropped on erasure."""

import uuid

from gateway.services.memory import RecallVectors

WORKSPACE = uuid.UUID("00000000-0000-4000-8000-000000000001")
OTHER = uuid.UUID("00000000-0000-4000-8000-000000000002")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_what_is_held_is_given_back_for_the_ids_asked_and_the_workspace_that_holds_it() -> None:
    held = RecallVectors()
    first, second = uuid.uuid4(), uuid.uuid4()
    held.keep(WORKSPACE, {first: (0.5, 0.25), second: (1.0, 0.0)})

    taken = held.take(WORKSPACE, [first, uuid.uuid4()])

    assert list(taken) == [first] and list(taken[first]) == [0.5, 0.25]
    assert held.take(OTHER, [first]) == {}, "another workspace never reads what one holds"
    assert len(held) == 2


def test_an_embedding_expires_after_its_time_however_often_it_is_used() -> None:
    clock = Clock()
    held = RecallVectors(ttl_seconds=600, clock=clock)
    entry = uuid.uuid4()
    held.keep(WORKSPACE, {entry: (1.0,)})

    clock.now = 599
    assert entry in held.take(WORKSPACE, [entry])
    clock.now = 600
    assert held.take(WORKSPACE, [entry]) == {}
    assert len(held) == 0


def test_past_the_cap_the_least_recently_used_workspace_is_dropped() -> None:
    held = RecallVectors(max_vectors=2)
    old, recent = uuid.uuid4(), uuid.uuid4()
    held.keep(OTHER, {old: (1.0,)})
    held.keep(WORKSPACE, {recent: (1.0,)})
    held.take(OTHER, [old])  # OTHER is now the most recently used

    held.keep(WORKSPACE, {uuid.uuid4(): (1.0,)})

    assert held.take(OTHER, [old]) == {}, "WORKSPACE was used after, so OTHER is dropped first"
    assert len(held) == 2


def test_erasing_drops_everything_held_of_the_workspace() -> None:
    held = RecallVectors()
    entry = uuid.uuid4()
    held.keep(WORKSPACE, {entry: (1.0,)})
    held.keep(OTHER, {uuid.uuid4(): (1.0,)})

    held.forget(WORKSPACE)

    assert held.take(WORKSPACE, [entry]) == {}
    assert len(held) == 1
