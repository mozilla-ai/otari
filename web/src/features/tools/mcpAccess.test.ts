import { describe, expect, it } from "vitest"

import {
  accessMode,
  defaultPolicySummary,
  isLiteralToolName,
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

  it("reads a rule of the read preset as its own answer", () => {
    expect(accessMode(["@read"])).toBe("read")
    expect(accessMode(["@read", "create_issue"])).toBe("some")
    expect(toolsFor("read", ["x"])).toEqual(["@read"])
    expect(ruleTools(["@read"])).toBe("Read tools")
    expect(ruleTools(["@read", "create_issue"])).toBe(
      "read tools, create_issue",
    )
    expect(defaultPolicySummary(["@read"])).toBe("DEFAULT: READ TOOLS")
    // The preset is the server's labels, which a name cannot match by spelling.
    expect(patternMatches("@read", "@read")).toBe(false)
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

  it("expands the read preset into the tools labeled read", () => {
    const base = {
      server_id: "s",
      name: "github",
      enabled: true,
      decided_by: "key" as const,
      grant_id: "g",
      read_tools: ["list_issues", "get_issue"],
    }
    // The only tool the server exposes comes from the preset, not the named one.
    expect(
      keyAccessLabel({
        ...base,
        tools: ["@read", "create_issue"],
        allowed_tools: ["list_issues"],
      }),
    ).toBe("list_issues")
    expect(
      keyAccessLabel({ ...base, tools: ["@read"], allowed_tools: null }),
    ).toBe("Read tools: list_issues, get_issue")
    expect(
      keyAccessLabel({
        ...base,
        tools: ["@read"],
        allowed_tools: null,
        read_tools: [],
      }),
    ).toBe("No access")
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

describe("isLiteralToolName", () => {
  it("offers a name that means only itself", () => {
    expect(isLiteralToolName("list_issues")).toBe(true)
  })

  it.each([
    "*",
    "search*",
    "@read",
    " @read",
    "list_issues ",
    ",",
    "get_issue,delete_issue",
    "",
    "\u0085@read",
    "list issues",
  ])(
    "keeps %j out of the choices, since a rule would read it as a pattern",
    (name) => {
      expect(isLiteralToolName(name)).toBe(false)
    },
  )
})
