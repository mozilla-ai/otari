import { afterEach, describe, expect, it, vi } from "vitest"

import { rememberOAuthState } from "@/features/auth/oauthRoundTrip"

const STATE_KEY = "otari.oauth.state"
const RETURN_KEY = "otari.oauth.return-hash"

describe("rememberOAuthState", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.sessionStorage.clear()
  })

  it("still stores the state when the page to return to cannot be stored", () => {
    // The state is what the sign-in needs; the page is a convenience, and a
    // failure to store it must neither throw nor leave an older one behind.
    window.sessionStorage.setItem(RETURN_KEY, "#/from-an-earlier-attempt")
    const setItem = Storage.prototype.setItem
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (
      this: Storage,
      key: string,
      value: string,
    ) {
      if (key === RETURN_KEY) {
        throw new DOMException("Storage is full.", "QuotaExceededError")
      }
      setItem.call(this, key, value)
    })

    expect(() =>
      rememberOAuthState("the-state", { returnHash: "#/organization/billing" }),
    ).not.toThrow()
    expect(window.sessionStorage.getItem(STATE_KEY)).toBe("the-state")
    expect(window.sessionStorage.getItem(RETURN_KEY)).toBeNull()
  })
})
