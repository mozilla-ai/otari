---
applyTo: "alembic/versions/**/*.py,src/gateway/models/**/*.py,src/gateway/api/routes/_platform.py,src/gateway/api/routes/hybrid_mode.py,docs/hybrid-mode-protocol.md,docs/code-execution-protocol.md"
---

# M4 Reconciliation Ledger

The M4 reconciliation ledger lives in [otari-ai#1587](https://github.com/mozilla-ai/otari-ai/issues/1587). It is the authoritative inventory of every capability built twice across this gateway and the otari.ai platform, or held deliberately in one of them. That spec ([otari-ai#1451](https://github.com/mozilla-ai/otari-ai/issues/1451)) is written and closed; the ledger now scopes what follows it, the rehome ([otari-ai#1452](https://github.com/mozilla-ai/otari-ai/issues/1452)) and the overlay split ([otari-ai#1455](https://github.com/mozilla-ai/otari-ai/issues/1455), [otari-ai#1456](https://github.com/mozilla-ai/otari-ai/issues/1456)). A surface missing from the ledger is a surface whose home nobody has decided.

One ledger covers both repositories. Entries from here go on that issue, not into a second list.

## When to append

**Only when the ledger would be wrong without it.** That is one of:

- **The target disposition changes** for a row: the surface re-parents somewhere else, dedupes to a different home, becomes overlay depth, or becomes gateway-only.
- **The surface has no row**, and you are proposing one, with its Current scope, Target and Tracking cells.
- **A row's Current scope cell became false**: a table you dropped, a constraint that moved, a capability that stopped existing.

Otherwise, do not append. A PR that adds a table or a cross-repo contract an existing row already covers, heading where that row already says, records that in its own description. The ledger holds where a surface is going, not the history of how it got there.

The reason this gate exists: sixteen of the ledger's first thirty-five entries reported that the target disposition was unchanged. Each was accurate, none changed what the ledger knows, and together they turned the authoritative table into something a reader had to reconstruct from a thread.

**One entry per change, not per PR.** A stack of PRs delivering one surface change gets one entry naming all of them.

These are the surfaces to weigh against the gate: a persistent table (a new entity in `src/gateway/models/` or a new Alembic migration), a control-plane contract the platform consumes or serves (`src/gateway/api/routes/_platform.py`, `src/gateway/api/routes/hybrid_mode.py`, and the wire contracts in [docs/hybrid-mode-protocol.md](../../docs/hybrid-mode-protocol.md) and [docs/code-execution-protocol.md](../../docs/code-execution-protocol.md)), a capability otari.ai also has, and a mode-specific surface.

## What to write

Comment on [otari-ai#1587](https://github.com/mozilla-ai/otari-ai/issues/1587) with:

1. **The affected row**, by name, from the ledger table. If no row covers it, propose a new row with its Current scope, Target, and Tracking cells.
2. **A link to the PR.**
3. **What the ledger did not already know.** Which of the three triggers above fired, and what the row should say now. If the target changes (re-parent under the default org/workspace, dedupe to one home, seam to enterprise, survive as gateway-only), say which and why.

Keep it to a few sentences. The ledger records where a surface is heading, not how it gets there; the migration spec (otari-ai#1451) owns schema, cutover, and rollback.

Note that "gateway-only" is a disposition about where behavior runs, not a claim that the feature is out of scope for the ledger. A gateway-only capability whose management data gets re-parented to an organization or workspace still belongs on the ledger.
