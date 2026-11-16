"""The embeddings automatic recall holds: by workspace, for a while, within a cap, and dropped on erasure."""

import uuid

import numpy as np

from gateway.services.memory import RecallVectors

WORKSPACE = uuid.UUID("00000000-0000-4000-8000-000000000001")
OTHER = uuid.UUID("00000000-0000-4000-8000-000000000002")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _held_ids(held: RecallVectors, workspace_id: uuid.UUID, ids: list[uuid.UUID]) -> list[uuid.UUID]:
    view = held.view(workspace_id, ids)
    return view.ids if view is not None else []


def test_what_is_held_is_given_back_scaled_to_length_one_for_the_workspace_that_holds_it() -> None:
    held = RecallVectors()
    first, second = uuid.uuid4(), uuid.uuid4()
    held.keep(WORKSPACE, {first: (3.0, 4.0), second: (1.0, 0.0)})

    view = held.view(WORKSPACE, [second, uuid.uuid4(), first])

    assert view is not None and view.ids == [second, first], "in the order asked, what is not held left out"
    assert np.allclose(view.matrix[view.rows], [[1.0, 0.0], [0.6, 0.8]])
    assert held.view(OTHER, [first]) is None, "another workspace never reads what one holds"
    assert held.missing(WORKSPACE, [first, (new := uuid.uuid4())]) == [new]
    assert len(held) == 2


def test_an_all_zero_embedding_is_held_as_zeros() -> None:
    held = RecallVectors()
    entry = uuid.uuid4()
    held.keep(WORKSPACE, {entry: (0.0, 0.0)})

    view = held.view(WORKSPACE, [entry])

    assert view is not None and view.matrix[view.rows].tolist() == [[0.0, 0.0]]


def test_an_embedding_expires_after_its_time_however_often_it_is_used() -> None:
    clock = Clock()
    held = RecallVectors(ttl_seconds=600, clock=clock)
    entry = uuid.uuid4()
    held.keep(WORKSPACE, {entry: (1.0,)})

    clock.now = 599
    assert _held_ids(held, WORKSPACE, [entry]) == [entry]
    clock.now = 600
    assert held.missing(WORKSPACE, [entry]) == [entry]
    assert len(held) == 0


def test_what_is_kept_later_expires_later() -> None:
    clock = Clock()
    held = RecallVectors(ttl_seconds=600, clock=clock)
    early, late = uuid.uuid4(), uuid.uuid4()
    held.keep(WORKSPACE, {early: (1.0, 0.0)})
    clock.now = 300
    held.keep(WORKSPACE, {late: (0.0, 1.0)})

    clock.now = 650

    assert _held_ids(held, WORKSPACE, [early, late]) == [late]


def test_a_matrix_already_handed_out_is_never_changed_by_what_is_kept_after() -> None:
    held = RecallVectors()
    first = uuid.uuid4()
    held.keep(WORKSPACE, {first: (1.0, 0.0)})
    view = held.view(WORKSPACE, [first])
    assert view is not None
    before = view.matrix.copy()

    held.keep(WORKSPACE, {uuid.uuid4(): (0.0, 1.0)})

    assert np.array_equal(view.matrix, before), "a ranking on a worker thread reads what it was handed"


def test_dead_rows_are_compacted_away_and_what_lives_is_kept() -> None:
    clock = Clock()
    held = RecallVectors(ttl_seconds=10, clock=clock)
    old = [uuid.uuid4() for _ in range(4)]
    held.keep(WORKSPACE, {entry: (1.0, float(index)) for index, entry in enumerate(old)})
    clock.now = 5
    young = uuid.uuid4()
    held.keep(WORKSPACE, {young: (0.0, 1.0)})
    clock.now = 12  # the four old ones expire

    held.keep(WORKSPACE, {(newest := uuid.uuid4()): (1.0, 1.0)})
    view = held.view(WORKSPACE, [young, newest, *old])

    assert view is not None and view.ids == [young, newest]
    assert len(view.matrix) == 2, "the expired rows are gone from the matrix"
    assert np.allclose(view.matrix[view.rows], [[0.0, 1.0], [2**-0.5, 2**-0.5]])


def test_an_embedding_of_another_size_replaces_what_the_workspace_held() -> None:
    held = RecallVectors()
    old, new = uuid.uuid4(), uuid.uuid4()
    held.keep(WORKSPACE, {old: (1.0, 0.0)})

    held.keep(WORKSPACE, {new: (1.0, 0.0, 0.0)})

    assert _held_ids(held, WORKSPACE, [old, new]) == [new]
    assert len(held) == 1


def test_past_the_cap_the_least_recently_used_workspace_is_dropped() -> None:
    held = RecallVectors(max_vectors=2)
    old, recent = uuid.uuid4(), uuid.uuid4()
    held.keep(OTHER, {old: (1.0,)})
    held.keep(WORKSPACE, {recent: (1.0,)})
    held.view(OTHER, [old])  # OTHER is now the most recently used

    held.keep(WORKSPACE, {uuid.uuid4(): (1.0,)})

    assert held.view(OTHER, [old]) is None, "WORKSPACE was used after, so OTHER is dropped first"
    assert len(held) == 2


def test_erasing_drops_everything_held_of_the_workspace() -> None:
    held = RecallVectors()
    entry = uuid.uuid4()
    held.keep(WORKSPACE, {entry: (1.0,)})
    held.keep(OTHER, {uuid.uuid4(): (1.0,)})

    held.forget(WORKSPACE)

    assert held.view(WORKSPACE, [entry]) is None
    assert len(held) == 1
