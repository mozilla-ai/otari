import { describe, expect, it } from "vitest"

import {
  emailFromRoute,
  inviteFromRoute,
  tokenFromRoute,
} from "@/shared/helpers/routeParams"

describe("tokenFromRoute", () => {
  it("reads the token out of the link's fragment, where the gateway puts it", () => {
    expect(tokenFromRoute("/verify-email#token=abc123")).toBe("abc123")
    expect(tokenFromRoute("/accept-invitation#token=abc123")).toBe("abc123")
  })

  it("reads one from the query too, behind the fragment", () => {
    expect(tokenFromRoute("/verify-email?token=abc123")).toBe("abc123")
    expect(tokenFromRoute("/verify-email?token=old#token=new")).toBe("new")
  })

  it("answers null for a link that carries none", () => {
    expect(tokenFromRoute("/verify-email")).toBeNull()
    expect(tokenFromRoute("/verify-email?other=1")).toBeNull()
    expect(tokenFromRoute("")).toBeNull()
    // A truncated link, which `URLSearchParams` reports as "" rather than
    // null. Answered as malformed, because every caller reads null as "this
    // link carries nothing" and gates its request on a non-empty string: an
    // empty string satisfies neither and strands the page on its loading line.
    expect(tokenFromRoute("/verify-email#token=")).toBeNull()
    expect(tokenFromRoute("/accept-invitation?token=")).toBeNull()
  })
})

describe("emailFromRoute", () => {
  it("reads the address the accept page prefills the signup form with", () => {
    // Percent-encoded by URLSearchParams on the way in, decoded on the way out,
    // which is what makes the `@` and a `+` in an address survive the round
    // trip rather than arriving as a space.
    expect(emailFromRoute("/signup?email=ada%40example.com")).toBe(
      "ada@example.com",
    )
    expect(emailFromRoute("/signup?email=ada%2Botari%40example.com")).toBe(
      "ada+otari@example.com",
    )
  })

  it("answers null for a link that carries none", () => {
    expect(emailFromRoute("/signup")).toBeNull()
    expect(emailFromRoute("/signup?token=abc123")).toBeNull()
    expect(emailFromRoute("/signup?email=")).toBeNull()
    expect(emailFromRoute("")).toBeNull()
  })
})

describe("inviteFromRoute", () => {
  it("reads the invite from either half of the link", () => {
    expect(inviteFromRoute("/signup?email=a%40b.c&invite=k7pm")).toBe("k7pm")
    expect(inviteFromRoute("/signup?email=a%40b.c#invite=k7pm")).toBe("k7pm")
  })
})
