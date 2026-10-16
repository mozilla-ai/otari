"""The process-wide list of price generations lookups resolve against.

Holds the accepted models.dev snapshots (the active one and the history window)
and answers from the packaged snapshot while none has been accepted. Kept
current by the startup load, an accept on this worker, and the refresher that
picks up an accept on another. Reads are synchronous and take no lock: the
tuple is replaced whole, never mutated.
"""

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime

from gateway.services.pricing.bundled import bundled_generation
from gateway.services.pricing.models_dev_index import ModelsDevPriceIndex, PriceGeneration, PriceTimeline

# Each generation is a parsed index of several thousand entries, so the window
# held in memory is smaller than the history the database keeps.
MAX_RESIDENT_GENERATIONS = 10

_EARLIEST = datetime.min.replace(tzinfo=UTC)

_accepted: tuple[PriceGeneration, ...] = ()
# Whether ``_accepted`` starts at the first snapshot ever accepted. Only then
# does the packaged snapshot price the dates before it; once older history was
# pruned or left out of the window, those dates are unpriced rather than
# answered with a later rate.
_complete = True
_timeline: PriceTimeline | None = None


def set_accepted_generations(generations: Iterable[PriceGeneration], *, complete: bool = True) -> None:
    """Replace the accepted generations, keeping the newest ``MAX_RESIDENT_GENERATIONS``.

    ``complete`` says the oldest given one is the first ever accepted; dropping
    any to fit the window clears it.
    """
    global _accepted, _complete, _timeline
    ordered = sorted(generations, key=lambda generation: generation.effective_at)
    if len(ordered) > MAX_RESIDENT_GENERATIONS:
        ordered = ordered[-MAX_RESIDENT_GENERATIONS:]
        complete = False
    _accepted = tuple(ordered)
    _complete = complete
    _timeline = None


def add_accepted_generation(generation: PriceGeneration, *, history_pruned: bool = False) -> None:
    """Make ``generation`` the newest, as an accept on this worker does."""
    set_accepted_generations((*_accepted, generation), complete=_complete and not history_pruned)


def active_timeline() -> PriceTimeline:
    """The generations in effect, oldest first, ready for as-of lookups."""
    global _timeline
    if _timeline is None:
        if not _accepted:
            _timeline = PriceTimeline((bundled_generation(),))
        elif _complete:
            baseline = replace(bundled_generation(), effective_at=_EARLIEST)
            _timeline = PriceTimeline((baseline, *_accepted))
        else:
            _timeline = PriceTimeline(_accepted)
    return _timeline


def active_generations() -> tuple[PriceGeneration, ...]:
    """Every generation in effect, oldest first; the packaged snapshot when none was accepted."""
    return active_timeline().generations


def current_index() -> ModelsDevPriceIndex:
    """The index of the newest generation."""
    return active_generations()[-1].index


def reset_generations() -> None:
    """Forget every accepted generation (tests)."""
    global _accepted, _complete, _timeline
    _accepted = ()
    _complete = True
    _timeline = None
