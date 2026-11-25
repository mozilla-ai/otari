"""Record on the request's span what a commit changed in an audited model.

A model opts in by naming the columns to record in ``__audited__``. A column it
does not name is never recorded, so a credential column added later stays out
until someone lists it. Changes are collected at flush and added to the current
span as one ``setting.changed`` event per row once the transaction commits, so a
rolled-back change records nothing. Who made it is the span's ``enduser.id``,
set where the request's identity is resolved.

Only writes through the ORM unit of work are seen: a bulk ``update()`` or
``delete()`` statement never loads the row it changes, and nothing outside the
application (a migration, a manual query) passes through here.
"""

from typing import Any

from opentelemetry import trace
from sqlalchemy import event
from sqlalchemy.orm import Session, UOWTransaction
from sqlalchemy.orm.attributes import instance_state

SETTING_CHANGED_EVENT = "setting.changed"
_PENDING = "otari_audited_changes"


def _row_change(obj: object, operation: str) -> dict[str, str] | None:
    columns: tuple[str, ...] | None = getattr(type(obj), "__audited__", None)
    if not columns:
        return None
    state = instance_state(obj)
    changed: dict[str, str] = {}
    if operation != "delete":
        for name in columns:
            history = state.attrs[name].history
            if history.has_changes():
                old = history.deleted[0] if history.deleted else None
                new = history.added[0] if history.added else None
                changed[f"change.{name}"] = f"{old!r} -> {new!r}"
        if not changed:
            return None
    return {
        "change.table": state.mapper.local_table.description,
        "change.operation": operation,
        "change.row": ",".join(str(part) for part in state.mapper.primary_key_from_instance(obj)),
        **changed,
    }


def _collect(session: Session, _flush_context: UOWTransaction) -> None:
    # After the flush, so a new row has its key, and before its history resets.
    pending: list[dict[str, str]] = session.info.setdefault(_PENDING, [])
    for operation, objects in (("insert", session.new), ("update", session.dirty), ("delete", session.deleted)):
        for obj in objects:
            change = _row_change(obj, operation)
            if change is not None:
                pending.append(change)


def _emit(session: Session) -> None:
    changes = session.info.pop(_PENDING, None)
    if changes:
        span = trace.get_current_span()
        for attributes in changes:
            span.add_event(SETTING_CHANGED_EVENT, attributes)


def _discard(session: Session, *_args: Any) -> None:
    session.info.pop(_PENDING, None)


def install_change_audit() -> None:
    """Listen on every ORM session. Idempotent."""
    if not event.contains(Session, "after_flush", _collect):
        event.listen(Session, "after_flush", _collect)
        event.listen(Session, "after_commit", _emit)
        event.listen(Session, "after_rollback", _discard)
