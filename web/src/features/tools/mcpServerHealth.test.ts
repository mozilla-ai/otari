import { describe, expect, it } from "vitest"

import {
  healthDetail,
  healthLabel,
  healthNeedsAttention,
} from "@/features/tools/mcpServerHealth"
import { workspaceMcpServerHealth as health } from "@/tests/fixtures"

describe("mcpServerHealth", () => {
  it("counts the tools a passing check found", () => {
    expect(healthLabel(health({ tool_count: 1 }))).toBe("1 TOOL")
    expect(healthLabel(health({ tool_count: 12 }))).toBe("12 TOOLS")
    expect(healthDetail(health({ tool_count: 2, tools: ["a", "b"] }))).toBe(
      "Connected. The server offers 2 tools.",
    )
    expect(
      healthDetail(health({ tool_count: 1, tools: ["a", "b", "c"] })),
    ).toBe(
      "Connected. The allow-list admits 1 of the 3 tools the server offers.",
    )
    expect(healthNeedsAttention(health())).toBe(false)
  })

  it("flags allow-listed names the server does not offer", () => {
    const missing = health({
      tool_count: 1,
      missing_allowed_tools: ["close_issue", "merge"],
    })

    expect(healthLabel(missing)).toBe("1 TOOL · 2 MISSING")
    expect(healthDetail(missing)).toMatch(/does not offer close_issue, merge/)
    expect(healthNeedsAttention(missing)).toBe(true)
  })

  it("names a failure and carries the status the server refused with", () => {
    const refused = health({
      status: "failing",
      error: "unauthorized",
      http_status: 401,
      tool_count: null,
    })

    expect(healthLabel(refused)).toBe("TOKEN REJECTED")
    expect(healthDetail(refused)).toMatch(/Check the authorization token/)
    expect(healthNeedsAttention(refused)).toBe(true)
  })

  it("labels an unclassified refusal by its status", () => {
    const refused = health({
      status: "failing",
      error: "http_error",
      http_status: 406,
      tool_count: null,
    })

    expect(healthLabel(refused)).toBe("HTTP 406")
    expect(healthDetail(refused)).toMatch(/HTTP 406/)
  })
})
