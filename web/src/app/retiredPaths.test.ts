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

  it("opens the MCP servers list on its old Tools path, keeping the query", async () => {
    for (const url of ["/tools/mcp-servers", "/tools/mcp-servers/"]) {
      const landed = await open(`${url}?connected=1`)

      expect(landed.path, url).toBe("/mcp-servers")
      expect(landed.search, url).toEqual({ connected: "1" })
      expect(landed.entries, url).toBe(1)
    }
  })

  it("opens a server's page on its old Tools path, with the workspace it was opened for", async () => {
    const landed = await open("/tools/mcp-servers/srv-1?workspace=ws-9")

    expect(landed.path).toBe("/mcp-servers/srv-1")
    expect(landed.search).toEqual({ workspace: "ws-9" })
    expect(landed.entries).toBe(1)
  })

  it("follows a sign-in's return_to that still names the old path", async () => {
    // What the callback page does with the path a sign-in began from: it
    // navigates to it, replacing the callback's own entry.
    const history = createMemoryHistory({ initialEntries: ["/"] })
    const probe = createRouter({
      routeTree,
      history,
      parseSearch,
      stringifySearch,
    })
    await probe.load()
    history.replace("/tools/mcp-servers/srv-1?workspace=ws-9")
    await probe.load()

    expect(probe.state.location.pathname).toBe("/mcp-servers/srv-1")
    expect(probe.state.location.search).toEqual({ workspace: "ws-9" })
    expect(history.length).toBe(1)
  })

  it("leaves the new MCP servers paths and the Tools pages where they are", async () => {
    expect((await open("/mcp-servers")).path).toBe("/mcp-servers")
    expect((await open("/mcp-servers/srv-1")).path).toBe("/mcp-servers/srv-1")
    expect((await open("/tools/web-search")).path).toBe("/tools/web-search")
  })

  it("opens Guardrails on the old Rules path, dropping the tab it named", async () => {
    for (const url of ["/rules", "/rules/"]) {
      const landed = await open(`${url}?type=data&q=ci`)

      expect(landed.path, url).toBe("/guardrails")
      expect(landed.search, url).toEqual({ q: "ci" })
      expect(landed.entries, url).toBe(1)
    }
  })

  it("opens a guardrail's page on its old Rules path", async () => {
    const landed = await open("/rules/spend-each-b1")

    expect(landed.path).toBe("/guardrails/spend-each-b1")
    expect(landed.entries).toBe(1)
  })

  it("opens Guardrails on the workspace's old guardrails page", async () => {
    const landed = await open("/tools/guardrails")

    expect(landed.path).toBe("/guardrails")
    expect(landed.entries).toBe(1)
  })

  it("leaves the new Guardrails paths where they are", async () => {
    expect((await open("/guardrails")).path).toBe("/guardrails")
    expect((await open("/guardrails/data-g1")).path).toBe("/guardrails/data-g1")
  })
})
