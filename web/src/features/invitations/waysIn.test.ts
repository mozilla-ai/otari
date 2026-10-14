import { describe, expect, it } from "vitest"

import { waysInSentence } from "./waysIn"

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
