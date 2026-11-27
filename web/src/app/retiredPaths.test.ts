import { createMemoryHistory, createRouter } from "@tanstack/react-router"
import { describe, expect, it } from "vitest"

import { routeTree } from "@/routeTree.gen"
import { parseSearch, stringifySearch } from "@/shared/helpers/search"

/** The real route tree opened at `url`, once its redirects have run. */
async function open(url: string) {
  const history = createMemoryHistory({ initialEntries: [url] })
  const probe = createRouter({
    routeTree,
    history,
    parseSearch,
    stringifySearch,
  })
  await probe.load()
  return {
    path: probe.state.location.pathname,
    search: probe.state.location.search,
    // A replace leaves the one entry the router started with.
    entries: history.length,
  }
}

describe("retired paths", () => {
  it("opens Traces on the old Sessions path, keeping the query and leaving no history entry", async () => {
    const landed = await open("/sessions?range=7d&status=error&peek=t1")

    expect(landed.path).toBe("/traces")
    expect(landed.search).toEqual({
      range: "7d",
      status: "error",
      peek: "t1",
    })
    expect(landed.entries).toBe(1)
  })

  it("opens Traces on /traces itself", async () => {
    expect((await open("/traces?range=7d")).path).toBe("/traces")
  })
})
