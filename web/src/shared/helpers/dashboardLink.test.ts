import { describe, expect, it } from "vitest"

import { absoluteDashboardLink } from "./dashboardLink"

describe("absoluteDashboardLink", () => {
  it("leaves a link the server already made absolute alone", () => {
    expect(
      absoluteDashboardLink(
        "https://otari.example.com/accept-invitation#token=a",
        "http://localhost:8000",
      ),
    ).toBe("https://otari.example.com/accept-invitation#token=a")
  })

  it("resolves a relative link against this page's origin and base", () => {
    expect(
      absoluteDashboardLink(
        "/accept-invitation#token=abc",
        "https://app.example.com",
      ),
    ).toBe("https://app.example.com/accept-invitation#token=abc")
  })
})
