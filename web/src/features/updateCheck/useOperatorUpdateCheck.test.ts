import { describe, expect, it } from "vitest"
import {
  endpointHost,
  newestVersion,
} from "@/features/updateCheck/useOperatorUpdateCheck"

describe("newestVersion", () => {
  it("compares each part as a number", () => {
    expect(newestVersion(["0.30.9", "0.30.10", "0.4.99"])).toBe("0.30.10")
  })

  it("is empty for none", () => {
    expect(newestVersion([])).toBe("")
  })
})

describe("endpointHost", () => {
  it("names the host a check goes to", () => {
    expect(endpointHost("https://api.otari.ai/api/v1/update-checks")).toBe(
      "api.otari.ai",
    )
  })

  it("falls back to the value when it is not a URL", () => {
    expect(endpointHost("not a url")).toBe("not a url")
  })
})
