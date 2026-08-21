import { describe, expect, it } from "vitest"

import {
  isPublicAuthPageAvailable,
  publicAuthPath,
} from "@/features/auth/publicAuthPaths"

describe("publicAuthPath", () => {
  it("matches on the path alone, so a query string does not hide a known page", () => {
    expect(publicAuthPath("#/signup")).toBe("/signup")
    expect(publicAuthPath("#/verify-email?token=abc")).toBe("/verify-email")
    expect(publicAuthPath("#/check-email?type=resend")).toBe("/check-email")
  })

  it("recognizes the two paths the gateway itself mails", () => {
    // `services/tenancy/user_service.py` builds `/#/verify-email?token=…` and
    // `/#/reset-password?token=…`; a rename here breaks a link already sent.
    expect(publicAuthPath("#/verify-email?token=t")).toBe("/verify-email")
    expect(publicAuthPath("#/reset-password?token=t")).toBe("/reset-password")
  })

  it("claims nothing it does not own", () => {
    expect(publicAuthPath("#/")).toBeNull()
    expect(publicAuthPath("#/keys")).toBeNull()
    expect(publicAuthPath("#/accept-invitation?token=t")).toBeNull()
    expect(publicAuthPath("")).toBeNull()
    // A prefix match would swallow this one; the table is exact.
    expect(publicAuthPath("#/signup-something-else")).toBeNull()
    // `in` would answer these, because it walks the prototype chain, and the
    // exhaustive switch that renders a page would fall off the end into a
    // blank screen. `Object.hasOwn` is what keeps them out.
    expect(publicAuthPath("#toString")).toBeNull()
    expect(publicAuthPath("#constructor")).toBeNull()
    expect(publicAuthPath("#__proto__")).toBeNull()
  })
})

describe("isPublicAuthPageAvailable", () => {
  it("hides the four flows that begin by sending a message when mail is off", () => {
    for (const path of [
      "/signup",
      "/check-email",
      "/resend-verification",
      "/recover-password",
    ] as const) {
      expect(isPublicAuthPageAvailable(path, false)).toBe(false)
      expect(isPublicAuthPageAvailable(path, true)).toBe(true)
    }
  })

  it("keeps the two token-landing pages open, since their message was already sent", () => {
    expect(isPublicAuthPageAvailable("/verify-email", false)).toBe(true)
    expect(isPublicAuthPageAvailable("/reset-password", false)).toBe(true)
  })
})
