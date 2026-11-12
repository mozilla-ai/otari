"""How memory ranks what it recalls: similarity to the query times the entry's confidence.

Confidence follows cq's rules (``cq-schema``), so workspace memory and cq's
organization lessons rank alike: it starts at ``INITIAL_CONFIDENCE``, each
member who confirms an entry raises it by ``CONFIRMATION_BOOST``, each who flags
it lowers it by ``FLAG_PENALTY``, and it stays between the floor and the
ceiling. A member counts once per entry, whichever they said last.
"""

import math
from collections.abc import Sequence

INITIAL_CONFIDENCE = 0.5
CONFIRMATION_BOOST = 0.1
FLAG_PENALTY = 0.15
CONFIDENCE_FLOOR = 0.0
CONFIDENCE_CEILING = 1.0
# Members who must flag an entry before it leaves recall and waits for an admin.
QUARANTINE_FLAGS = 3


def confidence(confirmations: int, flags: int) -> float:
    """An entry's confidence after ``confirmations`` members confirmed it and ``flags`` flagged it."""
    value = INITIAL_CONFIDENCE + confirmations * CONFIRMATION_BOOST - flags * FLAG_PENALTY
    return min(CONFIDENCE_CEILING, max(CONFIDENCE_FLOOR, value))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when their sizes differ or either is all zeros."""
    if len(a) != len(b) or not a:
        return 0.0
    # sumprod runs in C: recall compares every entry of a workspace on each turn, ten times faster than a generator.
    norm = math.sqrt(math.sumprod(a, a) * math.sumprod(b, b))
    return math.sumprod(a, b) / norm if norm else 0.0


def relevance(similarity: float, entry_confidence: float) -> float:
    """What a search orders by. A dissimilar entry ranks low however trusted it is."""
    return max(similarity, 0.0) * entry_confidence
