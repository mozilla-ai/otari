import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { withRouter } from "@/tests/router"

import { ExplainCard } from "./ExplainCard"

describe("ExplainCard", () => {
  it("links an MCP server in the workspace its call ran in", async () => {
    render(
      <ExplainCard
        explanation={{
          sentence: "The agent called create_issue. It never ran.",
          facts: [
            [
              "MCP server",
              "github",
              {
                serverId: "srv-1",
                workspace: "ws-1",
                key: "key-bo",
                tool: "create_issue",
              },
            ],
          ],
        }}
      />,
      { wrapper: withRouter() },
    )

    const link = await screen.findByRole("link", { name: "github" })
    const href = new URL(link.getAttribute("href") ?? "", "http://x")
    expect(href.pathname).toBe("/tools/mcp-servers/srv-1")
    expect(Object.fromEntries(href.searchParams)).toEqual({
      workspace: "ws-1",
      key: "key-bo",
      tool: "create_issue",
    })
  })
})
