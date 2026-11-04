import { describe, expect, it } from "vitest"

import { traceSpan } from "@/tests/traceFixtures"

import { explainSpan, groupWords } from "./explain"

describe("groupWords", () => {
  it("adds up the verdicts that read the same", () => {
    expect(groupWords({ refused: 1, unchecked: 2, passed: 4 }, 7)).toBe(
      "3 could not decide, 4 passed",
    )
  })

  it("counts the checks when none says a verdict", () => {
    expect(groupWords({}, 2)).toBe("2 checks")
  })
})

describe("explainSpan for a direct MCP call", () => {
  it("names the server and the rule that refused it, linking to both", () => {
    const span = traceSpan({
      kind: "tool",
      tool_name: "create_issue",
      outcome: "blocked",
      attributes: {
        "otari.mcp.server": "github",
        "otari.mcp.server_id": "srv-1",
        "otari.tool.decision": "reject",
        "otari.tool.decision_source": "mcp_key",
      },
    })

    const explained = explainSpan(span, undefined, [], undefined, "key-bo")

    expect(explained?.sentence).toBe(
      "The agent called create_issue. It never ran.",
    )
    expect(explained?.facts).toEqual([
      ["Tool", "create_issue"],
      ["MCP server", "github", { serverId: "srv-1" }],
      [
        "Decision",
        "Refused by the key's rule on this MCP server",
        { serverId: "srv-1", key: "key-bo", tool: "create_issue" },
      ],
    ])
  })
})
