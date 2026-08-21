import { describe, expect, it } from "vitest"

import { tokenFromHash } from "@/shared/helpers/hashParams"

describe("tokenFromHash", () => {
  it("reads the token out of a hash route's own query string", () => {
    // Not `window.location.search`, which is empty under hash history: the
    // parameters live after the `?` inside the fragment.
    expect(tokenFromHash("#/verify-email?token=abc123")).toBe("abc123")
    expect(tokenFromHash("#/accept-invitation?token=abc123")).toBe("abc123")
  })

  it("answers null for a link that carries none", () => {
    expect(tokenFromHash("#/verify-email")).toBeNull()
    expect(tokenFromHash("#/verify-email?other=1")).toBeNull()
    expect(tokenFromHash("")).toBeNull()
  })
})
