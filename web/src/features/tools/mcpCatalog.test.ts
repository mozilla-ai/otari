import { describe, expect, it } from "vitest"

import {
  describeAdded,
  describeRefusal,
  findFreeName,
  normalizeEndpoint,
  toCatalogPick,
} from "@/features/tools/mcpCatalog"
import {
  mcpCatalogFeatured,
  mcpCatalogServer,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"

describe("normalizeEndpoint", () => {
  it("reads two spellings of one endpoint as the same", () => {
    expect(normalizeEndpoint("https://MCP.Example.com/mcp/")).toBe(
      normalizeEndpoint(" https://mcp.example.com/mcp"),
    )
  })

  it("keeps different paths and queries apart", () => {
    expect(normalizeEndpoint("https://mcp.example.com/a")).not.toBe(
      normalizeEndpoint("https://mcp.example.com/b"),
    )
    expect(normalizeEndpoint("https://mcp.example.com/a?team=1")).not.toBe(
      normalizeEndpoint("https://mcp.example.com/a?team=2"),
    )
  })

  it("falls back to the text of something that is not a URL", () => {
    expect(normalizeEndpoint(" not a url ")).toBe("not a url")
  })
})

describe("findFreeName", () => {
  it("uses the title when no server has it", () => {
    expect(findFreeName(" Sentry ", new Set(["github"]))).toBe("Sentry")
  })

  it("numbers a title that is taken", () => {
    expect(findFreeName("Sentry", new Set(["Sentry", "Sentry 2"]))).toBe(
      "Sentry 3",
    )
  })

  it("stays within the gateway's 128 characters, suffix included", () => {
    const title = "x".repeat(200)
    const first = findFreeName(title, new Set())
    expect(first).toHaveLength(128)
    const second = findFreeName(title, new Set([first]))
    expect(second).toHaveLength(128)
    expect(second.endsWith(" 2")).toBe(true)
  })

  it("names a server whose title is blank", () => {
    expect(findFreeName("  ", new Set())).toBe("MCP server")
  })
})

describe("toCatalogPick", () => {
  it("carries a catalog server's own id", () => {
    expect(toCatalogPick(mcpCatalogServer({ id: "server-1" })).serverId).toBe(
      "server-1",
    )
  })

  it("carries the catalog id a featured entry was copied from, or none", () => {
    expect(
      toCatalogPick(mcpCatalogFeatured({ catalog_server_id: "server-2" }))
        .serverId,
    ).toBe("server-2")
    expect(
      toCatalogPick(mcpCatalogFeatured({ catalog_server_id: null })).serverId,
    ).toBeUndefined()
  })

  it("reads a missing description as none", () => {
    expect(
      toCatalogPick(mcpCatalogServer({ description: null })).description,
    ).toBe("")
  })
})

describe("describeAdded", () => {
  it("reports a passing check with what the server offers", () => {
    const outcome = describeAdded(
      workspaceMcpServer({
        health: workspaceMcpServerHealth({ tools: ["a", "b"], tool_count: 2 }),
      }),
    )
    expect(outcome.tone).toBe("success")
    expect(outcome.label).toBe("Connected")
  })

  it.each(["unauthorized", "forbidden"] as const)(
    "reads %s as a server that needs sign-in, not as a failure to add",
    (error) => {
      const outcome = describeAdded(
        workspaceMcpServer({
          health: workspaceMcpServerHealth({ status: "failing", error }),
        }),
      )
      expect(outcome.tone).toBe("warning")
      expect(outcome.label).toBe("Needs sign-in")
    },
  )

  it("says what failed when the check failed for another reason", () => {
    const outcome = describeAdded(
      workspaceMcpServer({
        health: workspaceMcpServerHealth({
          status: "failing",
          error: "timeout",
        }),
      }),
    )
    expect(outcome.label).toBe("Check failed")
    expect(outcome.detail).toContain("did not answer in time")
  })

  it("reports a server stored without a check", () => {
    expect(describeAdded(workspaceMcpServer({ health: null })).label).toBe(
      "Added",
    )
  })
})

describe("describeRefusal", () => {
  it("carries the gateway's reason", () => {
    expect(describeRefusal("A server named 'x' already exists")).toEqual({
      tone: "danger",
      label: "Not added",
      detail: "A server named 'x' already exists",
    })
  })
})
