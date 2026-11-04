import { describe, expect, it } from "vitest"

import { groupWords } from "./explain"

describe("groupWords", () => {
  it("adds up the verdicts that read the same", () => {
    expect(groupWords({ refused: 1, unchecked: 2, passed: 4 }, 7)).toBe(
      "3 could not decide, 4 passed",
    )
  })

  it("counts the checks when none says a verdict", () => {
    expect(groupWords({}, 2)).toBe("2 checks")
  })
})
