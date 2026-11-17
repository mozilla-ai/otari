import { describe, expect, it } from "vitest"
import { breakdownCut } from "@/features/home/breakdownCut"

describe("breakdownCut", () => {
  it("picks no cut while the roster is loading, so none is swapped later", () => {
    expect(
      breakdownCut({ chosen: undefined, roster: "loading", isTeam: false }),
    ).toBeUndefined()
  })

  it("opens a team on People and someone alone on API keys", () => {
    expect(
      breakdownCut({ chosen: undefined, roster: "ready", isTeam: true }),
    ).toBe("people")
    expect(
      breakdownCut({ chosen: undefined, roster: "ready", isTeam: false }),
    ).toBe("keys")
  })

  it("settles on API keys when the roster cannot be read", () => {
    expect(
      breakdownCut({ chosen: undefined, roster: "failed", isTeam: false }),
    ).toBe("keys")
  })

  it("keeps the cut the URL names, whatever the roster says", () => {
    expect(
      breakdownCut({ chosen: "models", roster: "loading", isTeam: false }),
    ).toBe("models")
  })
})
