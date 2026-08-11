---
applyTo: "alembic/versions/**/*.py,src/gateway/models/**/*.py,src/gateway/api/routes/_platform.py,src/gateway/api/routes/hybrid_mode.py,docs/hybrid-mode-protocol.md,docs/code-execution-protocol.md"
---

# M4 Reconciliation Ledger

The M4 reconciliation ledger lives in [otari-ai#1587](https://github.com/mozilla-ai/otari-ai/issues/1587). It is the authoritative inventory of every capability built twice across this gateway and the otari.ai platform, or held deliberately in one of them. It is the input the M4 migration spec ([otari-ai#1451](https://github.com/mozilla-ai/otari-ai/issues/1451)) starts from, so a surface missing from the ledger is a surface the spec will not reconcile.

One ledger covers both repositories. Entries from here go on that issue, not into a second list.

## When to append

Append an entry when your PR does any of the following:

- **Adds or changes a persistent table.** A new entity in `src/gateway/models/`, a new Alembic migration, or a change to what an existing table means (a column that turns derived state into explicit state counts).
- **Adds or changes a control-plane contract the platform consumes or serves.** `src/gateway/api/routes/_platform.py`, `src/gateway/api/routes/hybrid_mode.py`, and the wire contracts in [docs/hybrid-mode-protocol.md](../../docs/hybrid-mode-protocol.md) and [docs/code-execution-protocol.md](../../docs/code-execution-protocol.md).
- **Adds a capability the platform also has.** Anything that becomes a second implementation of something otari.ai already does (pricing, budgets, usage accounting, tenancy, provider management, tool configuration).
- **Adds a mode-specific surface.** Anything that exists in standalone and not hybrid, or the reverse.

Refactors that move code without changing a table, a cross-repo contract, or a duplicated capability do not need an entry.

## What to write

Comment on [otari-ai#1587](https://github.com/mozilla-ai/otari-ai/issues/1587) with:

1. **The affected row**, by name, from the ledger table. If no row covers it, propose a new row with its Current scope, Target, and Tracking cells.
2. **A link to the PR.**
3. **Whether the target disposition changes.** Most entries confirm the existing target and add detail; say so explicitly when that is the case. If the target changes (re-parent under the default org/workspace, dedupe to one home, seam to enterprise, survive as gateway-only), say which and why.

Keep it to a few sentences. The ledger records where a surface is heading, not how it gets there; the migration spec (otari-ai#1451) owns schema, cutover, and rollback.

Note that "gateway-only" is a disposition about where behavior runs, not a claim that the feature is out of scope for the ledger. A gateway-only capability whose management data gets re-parented to an organization or workspace still belongs on the ledger.
