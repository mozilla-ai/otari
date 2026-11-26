import { describe, expect, it } from "vitest"
import { busiestWorkspace } from "@/features/home/busiestWorkspace"

describe("busiestWorkspace", () => {
  it("picks the workspace that spent the most, the first on a tie", () => {
    expect(busiestWorkspace(["a", "b", "c"], [1, 3, 3])).toBe("b")
  })

  it("leaves a failed read out rather than reading it as nothing spent", () => {
    expect(busiestWorkspace(["a", "b"], [undefined, 0])).toBe("b")
  })

  it("has no answer when every read failed, so the selection stays put", () => {
    expect(busiestWorkspace(["a", "b"], [undefined, undefined])).toBeUndefined()
  })
})
