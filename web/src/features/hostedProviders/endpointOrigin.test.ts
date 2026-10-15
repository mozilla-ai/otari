import { describe, expect, it } from "vitest"

import { movesToAnotherOrigin, originOf } from "./endpointOrigin"

describe("originOf", () => {
  it("makes the default port explicit and folds the host's case", () => {
    expect(originOf("https://API.example.com/v1")).toBe(
      "https://api.example.com:443",
    )
    expect(originOf("http://example.com:8080")).toBe("http://example.com:8080")
  })

  it("reads a blank or hostless value as the provider's default", () => {
    expect(originOf("")).toBe("")
    expect(originOf("  ")).toBe("")
    expect(originOf("example.com/v1")).toBe("")
  })
})

describe("movesToAnotherOrigin", () => {
  it("keeps the key for a new path on the same origin", () => {
    expect(
      movesToAnotherOrigin(
        "https://example.com/v1",
        "https://example.com:443/v2",
      ),
    ).toBe(false)
  })

  it("asks for the key on another host, scheme or port", () => {
    expect(
      movesToAnotherOrigin(
        "https://example.com/v1",
        "https://proxy.example.com",
      ),
    ).toBe(true)
    expect(
      movesToAnotherOrigin("https://example.com/v1", "http://example.com/v1"),
    ).toBe(true)
    expect(
      movesToAnotherOrigin(
        "https://example.com/v1",
        "https://example.com:8443",
      ),
    ).toBe(true)
    expect(movesToAnotherOrigin("", "https://proxy.example.com")).toBe(true)
  })

  it("keeps the key when the endpoint goes back to the default", () => {
    expect(movesToAnotherOrigin("https://example.com/v1", "")).toBe(false)
  })
})
