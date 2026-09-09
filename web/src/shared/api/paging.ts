/**
 * The bounded "fetch everything" walk.
 *
 * Shared because more than one tenancy read needs it; the page caps are what
 * stop a backend or proxy that ignores `skip` from turning a walk into an
 * unbounded request loop.
 */

import { apiFetch } from "@/shared/api/client"

const TENANCY_PAGE_SIZE = 1000
const TENANCY_MAX_PAGES = 100

interface Paged<T> {
  data: T[]
  count: number
}

export async function fetchAllPaged<T>(
  path: string,
  // Appended after the paging pair rather than merged with it, so every
  // existing caller's URL is unchanged.
  params?: Record<string, string>,
): Promise<T[]> {
  const extra = params ? `&${new URLSearchParams(params)}` : ""
  const all: T[] = []
  for (let page = 0; page < TENANCY_MAX_PAGES; page += 1) {
    const body = await apiFetch<Paged<T>>(
      `${path}?skip=${page * TENANCY_PAGE_SIZE}&limit=${TENANCY_PAGE_SIZE}${extra}`,
    )
    all.push(...body.data)
    if (body.data.length < TENANCY_PAGE_SIZE) break
  }
  return all
}

// The organization the caller's identity is pointed at, and their standing in
// it. Every tenancy page reads it first: it names the tenant on screen and
// decides whether the management controls are offered at all. Read often and
// changed rarely, so it is cached for a minute like the other management lists.
