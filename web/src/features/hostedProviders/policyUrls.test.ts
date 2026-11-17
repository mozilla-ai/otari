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
