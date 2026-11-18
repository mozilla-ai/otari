import { describe, expect, it } from "vitest"

import {
  countriesFromJson,
  formatCountries,
  parseCountries,
} from "./supportedCountries"

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

describe("countriesFromJson", () => {
  it("reads a JSON list of codes", () => {
    expect(countriesFromJson('["US", "de"]')).toEqual({
      ok: true,
      value: ["US", "de"],
    })
  })

  it("reads the list under supported_countries, as the API and config.yml spell it", () => {
    expect(countriesFromJson('{"supported_countries": ["GB"]}')).toEqual({
      ok: true,
      value: ["GB"],
    })
  })

  it.each([
    ["not JSON", "US, DE"],
    ["a list of numbers", "[1, 2]"],
    ["an object without the key", '{"countries": ["US"]}'],
    ["a bare string", '"US"'],
  ])("refuses %s", (_, text) => {
    expect(countriesFromJson(text)).toEqual({
      ok: false,
      error:
        'The file is not a JSON list of country codes, such as ["US", "DE"].',
    })
  })
})
