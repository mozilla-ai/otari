import { describe, expect, it } from "vitest"

import { urlProblem } from "@/features/tools/mcpServerForm"

describe("urlProblem", () => {
  it("asks for https once a shared token rides on the URL", () => {
    expect(urlProblem("http://mcp.example.com", true)).toMatch(
      /authorization token/,
    )
    expect(urlProblem("http://mcp.example.com", false)).toBeUndefined()
  })

  it("asks for https on a per-user server in its own words, since it has no token field", () => {
    const reason = urlProblem("http://mcp.example.com", false, true)

    expect(reason).toMatch(/each user signs in to/)
    expect(reason).not.toMatch(/authorization token/)
    expect(urlProblem("https://mcp.example.com", false, true)).toBeUndefined()
  })
})
