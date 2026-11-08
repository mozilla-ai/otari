"""One page of a provider's offered models, narrowed to one kind of model."""

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
) -> KindPage[Row]:
    """Narrow ``rows`` to ``model_type`` and cut one page from what is left.

    The kind of a row comes from ``kinds``, keyed by the row's ``name_of``
    attribute, and a row with none is ``unknown``. The counts cover every row,
    so a client can name each kind beside its share whichever one is shown.
    """

    def kind_of(row: Row) -> OfferedModelType:
        return kinds.get(getattr(row, name_of)) or UNKNOWN_MODEL_TYPE

    type_counts: dict[OfferedModelType, int] = dict(Counter(kind_of(row) for row in rows))
    matching = [row for row in rows if model_type is None or kind_of(row) == model_type]
    return KindPage(rows=matching[skip : skip + limit], count=len(matching), type_counts=type_counts)
