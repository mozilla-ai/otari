"""One page of a provider's offered models, narrowed to one kind of model, to a search and to whether each is served."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gateway.types.model_type import UNKNOWN_MODEL_TYPE, ModelType, OfferedModelType


@dataclass(frozen=True)
class KindPage[Row]:
    """The rows one page shows, how many match the kind asked for, and how many of each kind there are."""

    rows: list[Row]
    count: int
    type_counts: dict[OfferedModelType, int]


def page_by_kind[Row](
    rows: Sequence[Row],
    kinds: Mapping[str, ModelType | None],
    name_of: str,
    *,
    model_type: OfferedModelType | None,
    skip: int,
    limit: int,
    search: str | None = None,
    enabled: bool | None = None,
) -> KindPage[Row]:
    """Narrow ``rows`` to ``search``, ``enabled`` and ``model_type``, and cut one page from what is left.

    The kind of a row comes from ``kinds``, keyed by the row's ``name_of``
    attribute, and a row with none is ``unknown``. ``search`` keeps the rows
    whose name contains it, case-insensitively, and ``enabled`` the rows whose
    ``enabled`` attribute equals it. The counts cover every row those two keep,
    so a client can name each kind beside its share whichever one is shown.
    """

    def kind_of(row: Row) -> OfferedModelType:
        return kinds.get(getattr(row, name_of)) or UNKNOWN_MODEL_TYPE

    needle = (search or "").strip().casefold()
    if needle:
        rows = [row for row in rows if needle in getattr(row, name_of).casefold()]
    if enabled is not None:
        rows = [row for row in rows if getattr(row, "enabled") is enabled]
    type_counts: dict[OfferedModelType, int] = dict(Counter(kind_of(row) for row in rows))
    matching = [row for row in rows if model_type is None or kind_of(row) == model_type]
    return KindPage(rows=matching[skip : skip + limit], count=len(matching), type_counts=type_counts)
