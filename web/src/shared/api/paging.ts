/**
 * The bounded "fetch everything" walk, in one place.
 *
 * **Not the shape to copy.** Reading a whole collection is what
 * `performance.md` forbids, and the cap below only stops the walk looping: it
 * does not paginate the read, and everything downstream still sorts and filters
 * in the browser. This module serves the nineteen reads that predate that rule
 * (otari#1376) and exists to be deleted as they are worked down. A new caller
 * wants something from its endpoint instead: `skip` and `limit` for a table, a
 * search parameter for a picker, an embedded label or a batch lookup where a
 * page is resolving ids.
 *
 * Two shapes reach it because the gateway answers two. The tenancy routes wrap
 * their rows in a `{ data, count }` envelope; `/keys`, `/users`, `/budgets`,
 * `/scoped-budgets` and `/pricing` answer a bare array. That is the only thing
 * that differs, so it is the only thing the two exports below decide.
 *
 * The caps are what the walk is for. A backend or proxy that ignores `skip`
 * returns a full page every time, which turns "read everything" into a request
 * loop with no end. A hundred pages of a thousand rows is 100k, past any real
 * key list, roster or price history, so reaching the cap means something is
 * wrong rather than something is large.
 */

import { apiFetch } from "@/shared/api/client"

/**
 * The largest `limit` every route walked here accepts (`le=1000` on each). Above
 * it they answer 422 rather than clamping, so raising this alone turns every one
 * of these reads into a validation error: the gateway has to move first.
 */
const PAGE_SIZE = 1000
const MAX_PAGES = 100

interface Paged<T> {
  data: T[]
  count: number
}

async function walk<Page, T>(
  path: string,
  params: Record<string, string> | undefined,
  rowsOf: (page: Page) => T[],
): Promise<T[]> {
  // Appended after the paging pair rather than merged with it, so every
  // existing caller's URL is unchanged.
  const extra = params ? `&${new URLSearchParams(params)}` : ""
  const all: T[] = []
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const body = await apiFetch<Page>(
      `${path}?skip=${page * PAGE_SIZE}&limit=${PAGE_SIZE}${extra}`,
    )
    const rows = rowsOf(body)
    all.push(...rows)
    // A short page is the last page. Equality would loop forever against a
    // backend that answers a full page past the end.
    if (rows.length < PAGE_SIZE) break
  }
  return all
}

/** Every row behind a route that answers the `{ data, count }` envelope. */
export function fetchAllPaged<T>(
  path: string,
  params?: Record<string, string>,
): Promise<T[]> {
  return walk<Paged<T>, T>(path, params, (body) => body.data)
}

/** Every row behind a route that answers a bare array. */
export function fetchAllRows<T>(
  path: string,
  params?: Record<string, string>,
): Promise<T[]> {
  return walk<T[], T>(path, params, (rows) => rows)
}
