import { afterEach, describe, expect, it, vi } from "vitest"

import * as apiClient from "@/shared/api/client"
import { fetchAllPaged, fetchAllRows } from "@/shared/api/paging"

const PAGE_SIZE = 1000

/** Spy on the transport and hand back pages of `total` rows. */
function serve(total: number, envelope: boolean) {
  const urls: string[] = []
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path) => {
    urls.push(String(path))
    const skip = Number(
      new URL(String(path), "http://x").searchParams.get("skip"),
    )
    const rows = Array.from(
      { length: Math.max(0, Math.min(PAGE_SIZE, total - skip)) },
      (_, i) => ({ id: skip + i }),
    )
    return (envelope ? { data: rows, count: total } : rows) as never
  })
  return urls
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("the bounded walk", () => {
  it("asks one page when the first one is short", async () => {
    const urls = serve(3, false)
    await expect(fetchAllRows("/users")).resolves.toHaveLength(3)
    expect(urls).toEqual(["/users?skip=0&limit=1000"])
  })

  it("keeps asking while a page comes back full", async () => {
    const urls = serve(2_500, false)
    await expect(fetchAllRows("/users")).resolves.toHaveLength(2_500)
    expect(urls).toEqual([
      "/users?skip=0&limit=1000",
      "/users?skip=1000&limit=1000",
      "/users?skip=2000&limit=1000",
    ])
  })

  it("asks a second page when the first is exactly full", async () => {
    // The end of the data and a full page look identical, so the walk cannot
    // stop on equality: it has to see a short page to know it is done.
    const urls = serve(PAGE_SIZE, false)
    await expect(fetchAllRows("/users")).resolves.toHaveLength(PAGE_SIZE)
    expect(urls).toHaveLength(2)
  })

  it("stops at the cap rather than looping on a backend that ignores skip", async () => {
    // The case the cap exists for: every page comes back full, forever.
    const urls = serve(Number.MAX_SAFE_INTEGER, false)
    const rows = await fetchAllRows("/users")
    expect(urls).toHaveLength(100)
    expect(rows).toHaveLength(100 * PAGE_SIZE)
  })

  it("appends extra parameters after the paging pair", async () => {
    const urls = serve(1, false)
    await fetchAllRows("/keys", { workspace_id: "ws-1" })
    expect(urls).toEqual(["/keys?skip=0&limit=1000&workspace_id=ws-1"])
  })

  it("reads rows out of the envelope the tenancy routes answer", async () => {
    const urls = serve(2, true)
    await expect(fetchAllPaged("/organizations/me/members")).resolves.toEqual([
      { id: 0 },
      { id: 1 },
    ])
    expect(urls).toEqual(["/organizations/me/members?skip=0&limit=1000"])
  })
})
