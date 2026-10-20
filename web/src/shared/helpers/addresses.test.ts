import { describe, expect, it } from "vitest"

import { looksLikeAddress, parseAddresses } from "@/shared/helpers/addresses"

describe("parseAddresses", () => {
  it("splits pasted addresses on commas, semicolons and whitespace, once each", () => {
    expect(
      parseAddresses(
        " ada@example.com, bob@example.com;\ncy@example.com  ADA@example.com,,",
      ),
    ).toEqual(["ada@example.com", "bob@example.com", "cy@example.com"])
  })
})

describe("looksLikeAddress", () => {
  it.each([
    ["ops@example.com", true],
    ["ops@mail.example.co", true],
    ["ops", false],
    ["ops@example", false],
    ["ops@@example.com", false],
    ["@example.com", false],
    ["ops@example..com", false],
    ["ops@example.com.", false],
  ])("%s → %s", (address, expected) => {
    expect(looksLikeAddress(address)).toBe(expected)
  })
})
