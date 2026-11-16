"""How long each stage of a recall or a search takes, to tell which one to make cheaper.

Stages: ``embed`` (the turn's embedding), ``list`` (the entries a member may recall, from the database),
``rank`` (the whole ranking, here or on a memory service), ``open`` and ``score`` (the ranking's two halves,
wherever it runs), ``rerank`` (the relevance check) and ``hand_over`` (opening what is handed over).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from gateway.metrics import REGISTRY, Counter, Histogram

_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

RECALL_STAGE_SECONDS = Histogram(
    "gateway_memory_recall_stage_seconds",
    "Seconds one stage of an automatic recall or a memory search took",
    ["stage"],
    buckets=_BUCKETS,
    registry=REGISTRY,
)

RERANK_SKIPPED = Counter(
    "gateway_memory_rerank_skipped",
    "Automatic recalls that handed over by similarity alone, the most alike entry being alike enough to skip the "
    "relevance check",
    registry=REGISTRY,
)


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Observe how long the block takes as ``stage``, whether it returns or raises."""
    started = perf_counter()
    try:
        yield
    finally:
        RECALL_STAGE_SECONDS.labels(stage=stage).observe(perf_counter() - started)
