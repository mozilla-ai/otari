import { describe, expect, it } from "vitest"

import {
  describeAdded,
  describeRefusal,
  findFreeName,
  hostOf,
  normalizeEndpoint,
  suggestMode,
  toCatalogPick,
} from "@/features/tools/mcpCatalog"
import {
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
  it("reads a missing description and a missing logo as none", () => {
    const pick = toCatalogPick(
      mcpCatalogServer({ description: null, logo_url: null }),
    )
    expect(pick.description).toBe("")
    expect(pick.logoUrl).toBe("")
  })

  it("carries the gateway's logo and whether the server is its maker's own", () => {
    const pick = toCatalogPick(
      mcpCatalogServer({
        logo_url: "/api/v1/mcp-catalog/logos/io.example/sentry?v=abc",
        logo_tone: "dark",
        publisher_kind: "vendor",
      }),
    )
    expect(pick.logoUrl).toBe(
      "/api/v1/mcp-catalog/logos/io.example/sentry?v=abc",
    )
    expect(pick.logoTone).toBe("dark")
    expect(pick.isOfficial).toBe(true)
    expect(hostOf(pick)).toBe("mcp.sentry.example")
    expect(
      toCatalogPick(mcpCatalogServer({ publisher_kind: "community" }))
        .isOfficial,
    ).toBe(false)
  })
})

describe("suggestMode", () => {
  const open = toCatalogPick(mcpCatalogServer())
  const tokened = toCatalogPick(mcpCatalogServer({ requires_token: true }))

  it("starts a server the registry gives a token on Shared, whatever the check says", () => {
    const probe = workspaceMcpServerHealth({
      status: "failing",
      error: "unauthorized",
    })
    expect(suggestMode(tokened, probe, false).mode).toBe("shared")
  })

  it.each(["unauthorized", "forbidden"] as const)(
    "suggests Personal for a server that answers %s to a caller with no account",
    (error) => {
      const probe = workspaceMcpServerHealth({ status: "failing", error })
      expect(suggestMode(open, probe, false)).toEqual({
        mode: "per_user",
        reason: "It asks each person to sign in.",
      })
    },
  )

  it("suggests Shared for a server that answers without a sign-in", () => {
    expect(suggestMode(open, workspaceMcpServerHealth(), false).mode).toBe(
      "shared",
    )
  })

  it("stays on Shared while the check runs and when it could not tell", () => {
    expect(suggestMode(open, undefined, true)).toEqual({
      mode: "shared",
      reason: "Checking how it signs in.",
    })
    const timeout = workspaceMcpServerHealth({
      status: "failing",
      error: "timeout",
    })
    expect(suggestMode(open, timeout, false)).toEqual({
      mode: "shared",
      reason: "",
    })
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
    "reads %s as a server that needs an account, not as a failure to add",
    (error) => {
      const health = workspaceMcpServerHealth({ status: "failing", error })
      const shared = describeAdded(workspaceMcpServer({ health }))
      expect(shared.tone).toBe("warning")
      expect(shared.label).toBe("Needs setup")
      // Where the account is set up, and both ways to do it.
      expect(shared.detail).toMatch(
        /on its page: sign in for everyone, or paste a token/,
      )
      const personal = describeAdded(
        workspaceMcpServer({ health, credential_mode: "per_user" }),
      )
      expect(personal.label).toBe("Needs sign-in")
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

  it("tells a Personal server's admin that members connect their own accounts", () => {
    const outcome = describeAdded(
      workspaceMcpServer({ credential_mode: "per_user", health: null }),
    )
    expect(outcome.label).toBe("Added")
    expect(outcome.detail).toContain("their own account")
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
