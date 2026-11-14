import { describe, expect, it } from "vitest"

import {
  CREDENTIAL_MODE_COPY,
  urlProblem,
} from "@/features/tools/mcpServerForm"

describe("urlProblem", () => {
  it("asks for https once a shared token rides on the URL", () => {
    expect(urlProblem("http://mcp.example.com", true)).toMatch(
      /authorization token/,
    )
    expect(urlProblem("http://mcp.example.com", false)).toBeUndefined()
  })

  it("asks for https on a per-user server in its own words, since it has no token field", () => {
    const reason = urlProblem("http://mcp.example.com", false, true)

    expect(reason).toMatch(/personal server/)
    expect(reason).not.toMatch(/authorization token/)
    expect(urlProblem("https://mcp.example.com", false, true)).toBeUndefined()
  })
})

describe("CREDENTIAL_MODE_COPY", () => {
  it("names the two modes Shared and Personal without touching the API values", () => {
    expect(CREDENTIAL_MODE_COPY.shared.label).toBe("Shared")
    expect(CREDENTIAL_MODE_COPY.per_user.label).toBe("Personal")
  })
})
