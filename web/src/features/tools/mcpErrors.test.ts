import { describe, expect, it } from "vitest"

import { mcpErrorMessage } from "@/features/tools/mcpErrors"
import { ApiError } from "@/shared/api/client"

describe("mcpErrorMessage", () => {
  it("never shows a message that names an id", () => {
    const raw = new ApiError(
      403,
      "MCP server 'notion' (2b1c7a4e-1111-4c2d-9a3b-5d6e7f8a9b0c) uses each user's own account",
    )
    expect(mcpErrorMessage(raw, "fallback")).not.toContain("2b1c7a4e")
    const conflict = new ApiError(
      409,
      "Server 2b1c7a4e-1111-4c2d-9a3b-5d6e7f8a9b0c changed",
    )
    expect(mcpErrorMessage(conflict, "The check could not run.")).toBe(
      "The check could not run.",
    )
  })

  it("keeps the gateway's plain sentences and names a busy gateway", () => {
    expect(
      mcpErrorMessage(new ApiError(409, "Sign in first."), "fallback"),
    ).toBe("Sign in first.")
    expect(mcpErrorMessage(new ApiError(429, "busy"), "fallback")).toMatch(
      /Try again in a moment/,
    )
    expect(mcpErrorMessage(new Error("boom"), "fallback")).toBe("fallback")
  })
})
