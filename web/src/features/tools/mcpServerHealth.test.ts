import { describe, expect, it } from "vitest"

import {
  checkedTools,
  healthDetail,
  healthNeedsAttention,
  healthState,
  healthStateDetail,
} from "@/features/tools/mcpServerHealth"
import { workspaceMcpServerHealth as health } from "@/tests/fixtures"

describe("mcpServerHealth", () => {
  it("says how many tools a passing check found", () => {
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

    expect(healthDetail(refused)).toMatch(/HTTP 406/)
  })
})

describe("healthState", () => {
  const passing = {
    status: "healthy" as const,
    checked_at: "2026-11-10T10:00:00Z",
    tool_count: 2,
    tools: ["a", "b"],
  }
  const now = Date.parse("2026-11-12T10:00:00Z")

  it("reads each of the four states", () => {
    expect(healthState(null, now)).toBe("unchecked")
    expect(
      healthState({ ...passing, status: "failing", error: "timeout" }, now),
    ).toBe("failing")
    expect(healthState(passing, now)).toBe("healthy")
    expect(
      healthState({ ...passing, missing_allowed_tools: ["gone"] }, now),
    ).toBe("attention")
  })

  it("claims no tool count for a server that answered without listing its tools", () => {
    // What a tool call records on a server no check ever listed.
    const reached = health({ tool_count: null, tools: [] })
    expect(healthDetail(reached)).toBe("Connected.")
    expect(checkedTools({ health: reached, allowed_tools: null }).isKnown).toBe(
      false,
    )
  })

  it("turns amber once a server has not answered in a week", () => {
    const later = now + 8 * 24 * 60 * 60 * 1000
    expect(healthState(passing, later)).toBe("attention")
    expect(healthStateDetail(passing, "attention")).toMatch(
      /No successful contact in the last 7 days/,
    )
  })
})
