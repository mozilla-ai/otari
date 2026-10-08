---
applyTo: "alembic/versions/**/*.py,src/gateway/models/**/*.py"
---

# Schema review instructions

A schema change is the one kind of change in this repository that a review
cannot take back. Code ships wrong and is reverted; a migration that ran against
production has already rewritten rows, and the revert is a second migration
somebody has to write under time pressure with the damage already done. Budget
and usage tables make that worse: they decide refusals and they carry money,
so a column that means something slightly different after the migration than
before it is a wrong 403, or a wrong charge, that nothing reports.

So the rule here is about **who** signs off, not only what to look for.

## Say plainly that a human has to review this

When a diff adds, drops, renames or retypes a column, table, index or
constraint, or backfills or rewrites existing rows:

- Open the review of that file by stating that the change needs review by a
  person, and that an automated review is not sufficient sign-off for it.
- **Ask the author directly to confirm the schema is right**, as a question that
  expects an answer, naming the specific decisions you want confirmed: the
  column's type and nullability, what the default does to rows already in the
  table, the direction and lossiness of any data conversion, and whether
  `downgrade()` genuinely restores what `upgrade()` changed.
- Do not mark the change approved, resolved or satisfied on the strength of your
  own reading. Say what you checked, say what you could not check, and leave the
  decision with the reviewer.

An automated review reads the diff. It does not know which rows exist in
production, which of them a conversion loosens, or whether a value that looks
like a leftover is somebody's deliberate setting. Those are the questions that
make a schema change safe, and none of them is answerable from the patch.

## What to put the question on

- **A conversion that changes meaning.** A migration that maps an old value onto
  a new vocabulary is making a product decision per row. Name the mapping back to
  the author and ask whether each arm is intended, especially any arm that makes
  a limit, a window or a permission *more permissive* than it was.
- **A dropped or renamed column.** Ask what still reads it: another service, a
  generated client, a dashboard, an export, a downstream consumer outside this
  repository.
- **A new non-nullable column.** Ask what its `server_default` means for rows
  that predate it, and whether that value is correct for them rather than merely
  valid.
- **A constraint added to an existing table.** Ask whether rows already in the
  table satisfy it, and what happens to the migration if they do not.
- **Anything under `budgets`, `scoped_budgets`, `budget_reservations` or
  `usage_logs`.** These decide refusals and carry money.
  Treat a change to their shape or semantics as needing an explicit yes.

## What still applies

The ordinary gates in [backend-standards](../skills/backend-standards/SKILL.md)
are unchanged and this does not replace them: a matching migration chained to the
current head, a real reversible `downgrade()`, a dialect-neutral chain that runs
on SQLite and PostgreSQL, `op.batch_alter_table(..., copy_from=...)` rather than
`ALTER TABLE ... ADD CONSTRAINT`, an explicit `ondelete` on every foreign key,
and a `server_default` for every new non-nullable column. Flag those as usual.
The rule above is what you add on top of them, not instead of them.
