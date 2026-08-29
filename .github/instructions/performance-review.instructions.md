---
applyTo: "src/gateway/**/*.py,web/src/**/*.{ts,tsx}"
---

# Performance review instructions

Focus on request paths, list and aggregation APIs, budget and usage services,
and dashboard work that runs on every navigation or poll.

## Database

- Batch lookups and writes. An `await db.execute` inside a loop needs a clear
  reason; prefer `IN`, eager loading, bulk statements, or a database cascade.
- Filter, sort, count, and aggregate in SQL. Do not load a growing table to
  process it in Python.
- Every list endpoint has a server-enforced limit, including operator endpoints.
  Existence checks stop after one row.
- Foreign keys and common filter, join, and sort columns need indexes. Add the
  matching index in the migration. Use a composite index for common combined
  predicates, based on the query's order and selectivity.
- Related writes share one transaction and commit once. Repositories do not
  commit. Preserve the atomic conditional updates and ordering used by budget
  reservations.
- Check query count and plans for nested response shapes and new usage
  aggregations. Watch for a query per row and unnecessary full-table sorts.
- Avoid loading large result sets into memory. Stream or page when the caller
  does not need the complete set.

Treat an N+1 or unbounded query on inference, billing, or a frequently refreshed
dashboard route as High severity. A missing index or avoidable full scan on a
bounded administrative table is usually Medium.

## Async work

- Do not run blocking network, file, subprocess, or database work on the event
  loop.
- Run independent I/O concurrently only when doing so does not violate
  transaction ordering, provider limits, or connection-pool bounds.
- Bound queues, retries, fan-out, and in-memory caches. Clean up sessions, files,
  tasks, streams, and subscriptions on every exit path.
- Do not create one database session per helper when the request already owns
  the transaction.

## Algorithms

Use sets or dictionaries for repeated membership checks. Avoid repeated sorting,
serialization, model validation, and parsing inside hot loops. A candidate set
is currently small only when a validator enforces that limit.

## Dashboard

- TanStack Query owns server state. Do not mirror it in component state.
- Let the server filter, sort, aggregate, and paginate growing datasets.
- Paginated or filtered queries keep previous data while fetching when the UX
  calls for continuity. Do not replace usable cached content with a full-page
  skeleton.
- Route files export only `Route`, preserving automatic code splitting.
- Lazy-load heavy, non-critical charts, editors, and dialogs. Check the build
  output when adding a dependency or changing imports.
- The React Compiler handles ordinary memoization. Add `useMemo`,
  `useCallback`, or `React.memo` only for a measured or semantic need.
- Use query polling rather than hand-written intervals, and remove listeners,
  observers, and subscriptions on unmount.
- Bound any client loop that walks paginated endpoints.

The frontend standards skill owns component and data-fetching patterns. This
file only identifies performance regressions.

## Findings

A finding names the file and line, the affected route or interaction, how cost
grows with data or traffic, and a concrete fix. Use numbers from a query plan,
query count, bundle output, profiler, or benchmark when available.

Use Critical for an easily triggered path that can exhaust the service. Use High
for request-path N+1 queries, unbounded growing-table reads, or per-item commits
on billing paths. Use Medium for missing indexes and measurable avoidable
client work. Do not report readability-only changes as performance findings.
