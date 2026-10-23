import { describe, expect, it } from "vitest"

import { formatCountries, parseCountries } from "./supportedCountries"

describe("parseCountries", () => {
  it("reads codes separated by commas or spaces, upper-cased and once each", () => {
    expect(parseCountries("us, de  gb,US")).toEqual({
      ok: true,
      value: ["US", "DE", "GB"],
    })
  })

  it("reads a blank field as every country", () => {
    expect(parseCountries("  ")).toEqual({ ok: true, value: null })
  })

  it("names every value that is not a two-letter code", () => {
    expect(parseCountries("US, USA, 1A")).toEqual({
      ok: false,
      invalid: ["USA", "1A"],
    })
  })
})

describe("formatCountries", () => {
  it("writes codes as the field shows them, and every country as blank", () => {
    expect(formatCountries(["US", "DE"])).toBe("US, DE")
    expect(formatCountries(null)).toBe("")
    expect(formatCountries(undefined)).toBe("")
  })
})
