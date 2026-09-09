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
  it("is true for Vite DEV", () => {
    expect(isLocalDashboard(true)).toBe(true)
  })

  it("is false for a production bundle, loopback included", () => {
    // A self-hosted gateway is a production build on localhost:8000, so a
    // loopback hostname must not earn the "Mixpanel not initialized" line.
    expect(isLocalDashboard(false)).toBe(false)
  })
})
