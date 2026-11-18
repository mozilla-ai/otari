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
    ["o'brien+ops@example.com", true],
    ["jös@example.com", true],
    ["ops@example.com(x.y)", false],
    ["x<ops@example.com>", false],
    ['"ops"@example.com', false],
    ["ops@example.com,ada@example.com", false],
    ["team:ops@example.com;", false],
    ["ops@[127.0.0.1]", false],
    ["ops\\x@example.com", false],
    ["ops x@example.com", false],
    ["ops\u200b@example.com", false],
    ["ops\u0000@example.com", false],
    ["=?utf-8?q?ops?=@example.com", false],
  ])("%s → %s", (address, expected) => {
    expect(looksLikeAddress(address)).toBe(expected)
  })
})
