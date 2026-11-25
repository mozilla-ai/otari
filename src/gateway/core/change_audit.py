"""Record on the request's span what a commit changed in an audited model.

A model opts in by naming the columns to record in ``__audited__``. A column it
does not name is never recorded, so a credential column added later stays out
until someone lists it. Changes are collected at flush and added to the current
span as one ``setting.changed`` event per row once the outermost transaction
commits. A released savepoint hands its changes to its parent; a savepoint or
transaction that rolls back, or a session closed without committing, drops only
its own. Who made the change is the span's ``enduser.id``, set where the
request's identity is resolved.

Only writes through the ORM unit of work are seen: a bulk ``update()`` or
``delete()`` statement never loads the row it changes, and nothing outside the
application (a migration, a manual query) passes through here.
"""

from opentelemetry import trace
from sqlalchemy import event
from sqlalchemy.orm import Session, SessionTransaction, UOWTransaction
from sqlalchemy.orm.attributes import instance_state

SETTING_CHANGED_EVENT = "setting.changed"
# Old value of a column the session never loaded, so the change cannot say what it replaced.
UNLOADED = "<unloaded>"
_PENDING = "otari_audited_changes"
_COMMITTED = "otari_audited_committed"

_Change = dict[str, str]


def _row_change(obj: object, operation: str) -> _Change | None:
    columns: tuple[str, ...] | None = getattr(type(obj), "__audited__", None)
    if not columns:
        return None
    state = instance_state(obj)
    changed: _Change = {}
    if operation != "delete":
        for name in columns:
            history = state.attrs[name].history
            if history.has_changes():
                if history.deleted:
                    old = repr(history.deleted[0])
                else:
                    old = repr(None) if operation == "insert" else UNLOADED
                new = history.added[0] if history.added else None
                changed[f"change.{name}"] = f"{old} -> {new!r}"
        if not changed:
            return None
    return {
        "change.table": state.mapper.local_table.description,
        "change.operation": operation,
        "change.row": ",".join(str(part) for part in state.mapper.primary_key_from_instance(obj)),
        **changed,
    }


def _innermost(session: Session) -> SessionTransaction | None:
    return session.get_nested_transaction() or session.get_transaction()


def _collect(session: Session, _flush_context: UOWTransaction) -> None:
    # After the flush, so a new row has its key, and before its history resets.
    transaction = _innermost(session)
    if transaction is None:
        return
    changes = [
        change
        for operation, objects in (("insert", session.new), ("update", session.dirty), ("delete", session.deleted))
        for obj in objects
        if (change := _row_change(obj, operation)) is not None
    ]
    if changes:
        pending: dict[SessionTransaction, list[_Change]] = session.info.setdefault(_PENDING, {})
        pending.setdefault(transaction, []).extend(changes)


def _mark_committed(session: Session) -> None:
    # Fires for a released savepoint as well as the outermost commit, before either ends.
    transaction = _innermost(session)
    if transaction is not None and transaction in session.info.get(_PENDING, {}):
        session.info.setdefault(_COMMITTED, set()).add(transaction)


def _settle(session: Session, transaction: SessionTransaction) -> None:
    changes = session.info.get(_PENDING, {}).pop(transaction, None)
    committed = transaction in session.info.get(_COMMITTED, set())
    session.info.get(_COMMITTED, set()).discard(transaction)
    if not changes or not committed:
        return
    if transaction.parent is not None:
        session.info[_PENDING].setdefault(transaction.parent, []).extend(changes)
        return
    span = trace.get_current_span()
    for attributes in changes:
        span.add_event(SETTING_CHANGED_EVENT, attributes)


def install_change_audit() -> None:
    """Listen on every ORM session. Idempotent."""
    if not event.contains(Session, "after_flush", _collect):
        event.listen(Session, "after_flush", _collect)
        event.listen(Session, "after_commit", _mark_committed)
        event.listen(Session, "after_transaction_end", _settle)
