import { describe, expect, it } from "vitest"

import { isLocalDashboard, readMixpanelToken } from "./mixpanelToken"

describe("readMixpanelToken", () => {
  it("maps a missing or non-string value to undefined", () => {
    // Do not pass `undefined`: that triggers the default parameter and reads
    // `import.meta.env.VITE_MIXPANEL_TOKEN`, so a Vitest process with a token
    // would fail this assertion. `null` and `1` cover the non-string arm.
    expect(readMixpanelToken(null)).toBeUndefined()
    expect(readMixpanelToken(1)).toBeUndefined()
  })

  it("maps an empty string to undefined", () => {
    expect(readMixpanelToken("")).toBeUndefined()
  })

  it("maps whitespace-only to undefined", () => {
    expect(readMixpanelToken("   ")).toBeUndefined()
    expect(readMixpanelToken("\n\t")).toBeUndefined()
  })

  it("trims a valid token", () => {
    expect(readMixpanelToken("  tok  ")).toBe("tok")
  })
})

describe("isLocalDashboard", () => {
  it("is true for Vite DEV even on a public hostname", () => {
    expect(isLocalDashboard("example.com", true)).toBe(true)
  })

  it("is true for make-dev loopback hosts on a production bundle", () => {
    expect(isLocalDashboard("localhost", false)).toBe(true)
    expect(isLocalDashboard("127.0.0.1", false)).toBe(true)
    expect(isLocalDashboard("[::1]", false)).toBe(true)
  })

  it("is false for a deployed host on a production bundle", () => {
    expect(isLocalDashboard("example.com", false)).toBe(false)
    expect(isLocalDashboard("otari.example", false)).toBe(false)
  })
})
