import { describe, expect, it } from "vitest"

import { verificationPendingSentence, waysInSentence } from "./waysIn"

describe("waysInSentence", () => {
  it("names the providers and the reset the gateway offers", () => {
    expect(
      waysInSentence(["provider", "password_reset"], ["google", "github"]),
    ).toBe(
      'sign in with Google or GitHub, or choose a password with "Forgot your password?".',
    )
  })

  it("offers only what exists", () => {
    expect(waysInSentence(["provider"], ["github"])).toBe(
      "sign in with GitHub.",
    )
    expect(waysInSentence(["password_reset"], [])).toBe(
      'choose a password with "Forgot your password?".',
    )
  })

  it("sends the person to the administrator when nothing is offered", () => {
    expect(waysInSentence([], [])).toBe(
      "ask whoever administers this gateway how to sign in.",
    )
  })
})

describe("verificationPendingSentence", () => {
  it("names the verification link where links can be mailed", () => {
    expect(
      verificationPendingSentence(["provider", "password_reset"], ["google"]),
    ).toBe(
      'open the verification link emailed at signup. If you did not sign up, sign in with Google, or choose a password with "Forgot your password?".',
    )
  })

  it("names the provider alone where links cannot be mailed", () => {
    expect(verificationPendingSentence(["provider"], ["google"])).toBe(
      "sign in with Google.",
    )
  })

  it("names no link where the deployment delivers nothing", () => {
    expect(verificationPendingSentence([], [])).toBe(
      "ask whoever administers this gateway how to sign in.",
    )
  })
})
