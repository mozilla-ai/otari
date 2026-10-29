import { describe, expect, it } from "vitest"

import {
  accessMode,
  defaultPolicySummary,
  keyAccessLabel,
  patternMatches,
  ruleAction,
  ruleTools,
  toolsFor,
  verdictLabel,
} from "@/features/tools/mcpAccess"

describe("mcpAccess", () => {
  it("reads a rule's three answers", () => {
    expect(accessMode(["*"])).toBe("all")
    expect(accessMode([])).toBe("none")
    expect(accessMode(["get_issue"])).toBe("some")
    expect(toolsFor("all", ["x"])).toEqual(["*"])
    expect(toolsFor("none", ["x"])).toEqual([])
    expect(toolsFor("some", ["x"])).toEqual(["x"])
  })

  it("reads a rule as its tools and action, and the default policy for the list", () => {
    expect([ruleTools(["*"]), ruleAction(["*"])]).toEqual(["Any", "ALLOW"])
    expect([ruleTools([]), ruleAction([])]).toEqual(["Any", "DENY"])
    expect([ruleTools(["a", "b"]), ruleAction(["a", "b"])]).toEqual([
      "a, b",
      "ALLOW",
    ])
    expect(defaultPolicySummary(["*"])).toBe("DEFAULT: ALLOW")
    expect(defaultPolicySummary([])).toBe("DEFAULT: DENY")
    expect(defaultPolicySummary(["a"])).toBe("DEFAULT: ALLOW 1 TOOL")
  })

  it("matches names and prefix patterns", () => {
    expect(patternMatches("*", "anything")).toBe(true)
    expect(patternMatches("list_*", "list_issues")).toBe(true)
    expect(patternMatches("list_*", "get_issue")).toBe(false)
    expect(patternMatches("get_issue", "get_issue")).toBe(true)
  })

  it("cuts what a key reaches down to the server's ceiling", () => {
    const base = {
      server_id: "s",
      name: "github",
      enabled: true,
      decided_by: "everyone" as const,
      grant_id: null,
    }
    expect(keyAccessLabel({ ...base, tools: ["*"], allowed_tools: null })).toBe(
      "All tools",
    )
    expect(
      keyAccessLabel({ ...base, tools: ["*"], allowed_tools: ["a", "b"] }),
    ).toBe("a, b")
    expect(
      keyAccessLabel({ ...base, tools: ["c"], allowed_tools: ["a", "b"] }),
    ).toBe("No access")
    expect(keyAccessLabel({ ...base, tools: [], allowed_tools: null })).toBe(
      "No access",
    )
  })

  it("says which rule a trace matched", () => {
    const options = { ruleOwner: "Bo's laptop", serverEnabled: true }
    expect(
      verdictLabel(
        { allowed: false, decided_by: "everyone", grant_id: null, tools: [] },
        options,
      ),
    ).toBe("DENY, matched the default policy (deny)")
    expect(
      verdictLabel(
        { allowed: true, decided_by: "key", grant_id: "g", tools: ["*"] },
        options,
      ),
    ).toBe("ALLOW, matched the rule for Bo's laptop")
    expect(
      verdictLabel(
        { allowed: false, decided_by: "server", grant_id: null, tools: [] },
        options,
      ),
    ).toBe("DENY, the server does not expose this tool")
    expect(
      verdictLabel(
        { allowed: false, decided_by: "server", grant_id: null, tools: [] },
        { ...options, serverEnabled: false },
      ),
    ).toBe("DENY, the server is disabled")
  })
})
