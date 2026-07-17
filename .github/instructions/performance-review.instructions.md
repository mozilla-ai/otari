---
applyTo: "src/gateway/**/*.py,web/src/**/*.{ts,tsx}"
---

# Performance Review Instructions

Apply this checklist when reviewing performance-sensitive gateway code (database access on the
request path, list endpoints, budget/usage services) and dashboard code (`web/`). The gateway
is async SQLAlchemy 2.0 throughout; examples use its real entities (`User`, `APIKey`, `Budget`,
`UsageLog`, `ModelPricing`, `ModelAlias`).

## Database performance

### N+1 queries: CWE-driven, but mostly a latency bug

```python
# BAD: a query per row inside a loop
for model_key in model_keys:
    row = (await db.execute(
        select(ModelPricing).where(ModelPricing.model_key == model_key)
    )).scalar_one_or_none()

# GOOD: one batched query
rows = (await db.execute(
    select(ModelPricing).where(ModelPricing.model_key.in_(model_keys))
)).scalars().all()
```

**Checklist:**
- ✅ No `await db.execute(select(...))` inside a `for` loop, batch with `.in_()`, or eager-load
  a relationship with `selectinload`/`joinedload` instead of touching it lazily per row.
- ✅ No one-by-one deletes in a loop, use a bulk `delete().where(... .in_(...))` or a cascade.
- ✅ Endpoints returning nested/derived data are checked (log SQL or `echo=True`) to confirm
  they don't fan out into N+1.
- **Severity:** High on the request/billing path.

### Missing indexes

Foreign keys and hot filter/sort columns must be indexed. otari already does this (e.g.
`APIKey.user_id` is `ForeignKey(..., ondelete="CASCADE"), index=True`; `User.deleted_at` is
`index=True`), keep it up when you add columns.

```python
# BAD: FK with no index: slow joins and slow CASCADE deletes
user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"))

# GOOD
user_id: Mapped[str | None] = mapped_column(
    ForeignKey("users.user_id", ondelete="CASCADE"), index=True,
)
```

**Checklist:**
- ✅ Every `ForeignKey` column has `index=True`.
- ✅ Columns used in `WHERE`, `ORDER BY`, or join conditions are indexed; frequently
  co-filtered columns get a composite `Index`.
- ✅ The Alembic migration creates the index alongside the column.
- **Severity:** Medium to High.

### Query optimization

**Push work into SQL; don't fetch rows to process in Python.**

```python
# BAD: load everything, filter/count in Python
logs = (await db.execute(select(UsageLog))).scalars().all()
recent = [l for l in logs if l.created_at > cutoff]
n = len(recent)

# GOOD: filter and count in SQL
recent = (await db.execute(
    select(UsageLog).where(UsageLog.created_at > cutoff)
)).scalars().all()
n = (await db.execute(
    select(func.count()).select_from(UsageLog).where(UsageLog.created_at > cutoff)
)).scalar_one()
```

**Every list endpoint has a hard upper bound.** Growing tables (`UsageLog`, `ModelPricing`)
must never be selected without a `limit`. The dashboard's pricing read already pages with a
server-side `limit` cap of 1000 and a client-side page cap, mirror that on both ends.

**Checklist:**
- ✅ `func.count()` for counts, not `len(all())`; an existence check selects one row / uses
  `.limit(1)`, not a full count.
- ✅ Pagination (`limit`/`skip`) with a sane default and maximum on every list endpoint,
  including admin/master-key ones.
- ✅ No `select(Model)` without `WHERE` or `LIMIT` on a table that grows over time.
- ✅ Filter, sort, count, and aggregate in SQL, not in Python.
- **Severity:** Medium.

### Transaction atomicity & batch writes

```python
# BAD: a commit per iteration
for user_id in user_ids:
    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one()
    user.blocked = True
    await db.commit()

# GOOD: one bulk update, one commit
await db.execute(update(User).where(User.user_id.in_(user_ids)).values(blocked=True))
await db.commit()
```

**Checklist:**
- ✅ Related writes are grouped in one transaction; commit once, not inside a loop.
- ✅ Multi-step writes that must succeed or fail together share a transaction.
- ✅ Budget/spend updates follow the atomic reservation pattern (a single conditional
  `UPDATE`), never read-modify-write across a provider call, see the security instructions
  (§0.2/§0.3).
- **Severity:** High on hot paths.

## Async efficiency

```python
# BAD: sequential awaits on independent work
pricing = await load_pricing()
providers = await load_providers()

# GOOD: run independent awaits together
pricing, providers = await asyncio.gather(load_pricing(), load_providers())
```

- ✅ Independent awaits use `asyncio.gather`; sequential `await` only when one result feeds the
  next.
- ✅ No blocking/sync I/O on the event loop in a request handler; sessions and file handles are
  closed via context managers / FastAPI dependencies.
- ✅ No unbounded in-memory accumulation of a growing table.
- **Severity:** Medium to High.

## Algorithm efficiency

```python
# BAD: O(n*m) membership scan          # GOOD: O(n) set lookup
for a in list_a:                        seen = set(list_b)
    if a in list_b:                     for a in list_a:
        ...                                 if a in seen:
                                                ...
```

- ✅ Watch for O(n²) nested loops / repeated linear scans; use `set`/`dict` for membership.
- ✅ Sort once and reuse; prefer generators for large sequences.

## Frontend (`web/`)

The dashboard is small, but the same principles apply:

- ✅ **Server does the shaping.** No client-side filtering/sorting/pagination of large server
  datasets when the endpoint can do it. Don't assemble a view from several requests and join in
  the browser, prefer one endpoint that returns what the page needs.
- ✅ **TanStack Query owns server-state caching**, never duplicate it in `useState`. Pick
  `staleTime` per how fast the data moves; invalidate only the keys a mutation actually changes
  (see the frontend `data-fetching` guide).
- ✅ **Bounded "fetch all" loops** (`fetchAllPricing` caps pages) so a misbehaving backend can't
  spin an unbounded request loop.
- ✅ **Effect cleanup**: remove listeners/intervals/subscriptions on unmount; correct
  dependency arrays.
- ✅ Memoize (`useMemo`/`useCallback`) only when a re-render is a measured problem, not by
  reflex.

## Severity guidelines

- **Critical**: request-path query that overloads the DB; unbounded memory growth on a hot
  path.
- **High**: endpoint >1s; N+1 on a frequently used route; unbounded list fetch on a growing
  table; per-iteration commits on a billing path.
- **Medium**: missing index on a moderately used query; suboptimal algorithm on small data;
  avoidable re-renders.
- **Low**: marginal caching wins; readability-only tweaks.

## Finding format

```markdown
## [Severity]: Performance: [brief description]
**File:** `src/gateway/.../file.py:line`
**Issue:** what is slow and why.
**Impact:** affected route(s), how it degrades as data grows.
**Recommendation:** the fix, ideally with a snippet.
```

## References
- [SQLAlchemy performance](https://docs.sqlalchemy.org/en/20/faq/performance.html) ·
  [React performance](https://react.dev/learn/render-and-commit#optimizing-performance)
