"""The process-wide list of price generations lookups resolve against.

Holds the accepted models.dev snapshots (the active one and the history window)
and answers from the packaged snapshot while none has been accepted. Kept
current by the startup load, an accept on this worker, and the refresher that
picks up an accept on another. Reads are synchronous and take no lock: the
tuple is replaced whole, never mutated.
"""

from collections.abc import Iterable

from gateway.services.pricing.bundled import bundled_generation
from gateway.services.pricing.models_dev_index import ModelsDevPriceIndex, PriceGeneration

# Each generation is a parsed index of several thousand entries, so the window
# held in memory is smaller than the history the database keeps.
MAX_RESIDENT_GENERATIONS = 10

_accepted: tuple[PriceGeneration, ...] = ()


def set_accepted_generations(generations: Iterable[PriceGeneration]) -> None:
    """Replace the accepted generations, keeping the newest ``MAX_RESIDENT_GENERATIONS``."""
    global _accepted
    ordered = sorted(generations, key=lambda generation: generation.effective_at)
    _accepted = tuple(ordered[-MAX_RESIDENT_GENERATIONS:])


def add_accepted_generation(generation: PriceGeneration) -> None:
    """Make ``generation`` the newest, as an accept on this worker does."""
    set_accepted_generations((*_accepted, generation))


def active_generations() -> tuple[PriceGeneration, ...]:
    """Every generation in effect, oldest first; the packaged snapshot when none was accepted."""
    return _accepted or (bundled_generation(),)


def current_index() -> ModelsDevPriceIndex:
    """The index of the newest generation."""
    return active_generations()[-1].index


def reset_generations() -> None:
    """Forget every accepted generation (tests)."""
    global _accepted
    _accepted = ()
