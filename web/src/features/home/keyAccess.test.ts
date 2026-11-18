import { describe, expect, it } from "vitest"
import { canOpenInKeys } from "@/features/home/keyAccess"

// Three keys spent this month: the caller's own, the organization's service
// key, and a teammate's. What each caller's keys endpoint lists differs.
const OWN = { id: "k-own" }
const SERVICE = { id: "k-service" }
const TEAMMATE = { id: "k-teammate" }

describe("canOpenInKeys", () => {
  it("offers every key to the deployment's operator, who lists them all", () => {
    const listed = [OWN, SERVICE, TEAMMATE]
    expect(canOpenInKeys(OWN.id, listed)).toBe(true)
    expect(canOpenInKeys(SERVICE.id, listed)).toBe(true)
    expect(canOpenInKeys(TEAMMATE.id, listed)).toBe(true)
  })

  it("offers an owner or admin their own and service keys, never a teammate's", () => {
    const listed = [OWN, SERVICE]
    expect(canOpenInKeys(OWN.id, listed)).toBe(true)
    expect(canOpenInKeys(SERVICE.id, listed)).toBe(true)
    expect(canOpenInKeys(TEAMMATE.id, listed)).toBe(false)
  })

  it("offers a member their own keys only", () => {
    const listed = [OWN]
    expect(canOpenInKeys(OWN.id, listed)).toBe(true)
    expect(canOpenInKeys(SERVICE.id, listed)).toBe(false)
  })

  it("offers nothing for spend with no key behind it", () => {
    expect(canOpenInKeys(null, [OWN])).toBe(false)
  })
})
