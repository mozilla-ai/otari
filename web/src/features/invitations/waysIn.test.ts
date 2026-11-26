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
  it("names the verification link and whatever else is offered", () => {
    expect(
      verificationPendingSentence(["provider", "password_reset"], ["google"]),
    ).toBe(
      'open the verification link emailed at signup. If you did not sign up, sign in with Google, or choose a password with "Forgot your password?".',
    )
  })

  // `verify_email` is answered only where a link reaches the mailbox, so a
  // reset is always among the ways in by the time this is rendered.
  it("names the reset alone where the deployment has no provider", () => {
    expect(verificationPendingSentence(["password_reset"], [])).toBe(
      'open the verification link emailed at signup. If you did not sign up, choose a password with "Forgot your password?".',
    )
  })
})
