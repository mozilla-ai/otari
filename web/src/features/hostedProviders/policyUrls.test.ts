import { describe, expect, it } from "vitest"

import {
  policyUrlError,
  policyUrlValue,
} from "@/features/hostedProviders/policyUrls"

describe("policyUrlError", () => {
  it("allows blank and absolute http(s) URLs", () => {
    expect(policyUrlError("")).toBeUndefined()
    expect(policyUrlError("  ")).toBeUndefined()
    expect(policyUrlError("https://example.com/terms")).toBeUndefined()
    expect(policyUrlError("http://example.com/privacy")).toBeUndefined()
  })

  it("refuses a relative path and another scheme", () => {
    expect(policyUrlError("example.com/terms")).toBeDefined()
    expect(policyUrlError("javascript:alert(1)")).toBeDefined()
    expect(policyUrlError("ftp://example.com/terms")).toBeDefined()
    // `new URL` would repair this one; the server would not.
    expect(policyUrlError("https:example.com/terms")).toBeDefined()
  })

  it("refuses a username or password", () => {
    const refusal = "Leave out a username or password."
    expect(policyUrlError("https://user:pw@example.com/terms")).toBe(refusal)
    expect(policyUrlError("https://user@example.com/terms")).toBe(refusal)
    expect(policyUrlError("https://:pw@example.com/terms")).toBe(refusal)
  })
})

describe("policyUrlValue", () => {
  it("sends null for blank and the trimmed URL otherwise", () => {
    expect(policyUrlValue(" ")).toBeNull()
    expect(policyUrlValue(" https://example.com/terms ")).toBe(
      "https://example.com/terms",
    )
  })
})
