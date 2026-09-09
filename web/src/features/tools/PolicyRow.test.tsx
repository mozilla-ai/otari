import { describe, expect, it } from "vitest"

import { ceilingParser, parsePhrase } from "@/features/tools/PolicyRow"

// The row itself is exercised through both policy cards, which drive it against
// a real mutation. What is worth asserting here is the parsing, because it is
// what decides whether a value reaches the server at all.

describe("ceilingParser", () => {
  const parse = ceilingParser(25, "rounds")

  it("reads a whole number inside the bound", () => {
    expect(parse("10")).toEqual({ value: 10, error: "" })
  })

  it("reads blank as no ceiling rather than as an error", () => {
    expect(parse("  ")).toEqual({ value: null, error: "" })
  })

  it.each(["0x10", "1e1", "1.5", "-3", "ten", "10px"])(
    "refuses %s rather than letting Number read it",
    (raw) => {
      // `Number` would take "0x10" as 16 and "1e1" as 10, so a value the
      // operator never typed would reach the server.
      expect(parse(raw).value).toBeNull()
      expect(parse(raw).error).toContain("whole number of rounds")
    },
  )

  it.each(["0", "26"])(
    "refuses %s, which is outside the server's bound",
    (raw) => {
      expect(parse(raw).value).toBeNull()
    },
  )

  it("names its own unit and bound, so two rows do not share a message", () => {
    expect(ceilingParser(60, "seconds")("600").error).toBe(
      "A whole number of seconds from 1 to 60.",
    )
  })
})

describe("parsePhrase", () => {
  it("trims what it stores", () => {
    expect(parsePhrase("  show your working  ")).toEqual({
      value: "show your working",
      error: "",
    })
  })

  it("clears the stored phrase when the field is emptied", () => {
    expect(parsePhrase("   ")).toEqual({ value: null, error: "" })
  })
})
