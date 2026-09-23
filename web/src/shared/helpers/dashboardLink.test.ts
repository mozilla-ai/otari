import { describe, expect, it } from "vitest"

import { absoluteDashboardLink } from "./dashboardLink"

describe("absoluteDashboardLink", () => {
  it("leaves a link the server already made absolute alone", () => {
    expect(
      absoluteDashboardLink("https://otari.example.com/#/accept?token=a", {
        origin: "http://localhost:8000",
        pathname: "/",
      }),
    ).toBe("https://otari.example.com/#/accept?token=a")
  })

  it("resolves a relative link against the page, keeping a hosted path prefix", () => {
    expect(
      absoluteDashboardLink("/#/accept-invitation?token=abc", {
        origin: "https://app.example.com",
        pathname: "/dashboard/",
      }),
    ).toBe("https://app.example.com/dashboard/#/accept-invitation?token=abc")
  })

  it("resolves a link with no hash against the origin", () => {
    expect(
      absoluteDashboardLink("/welcome", {
        origin: "http://localhost:8000",
        pathname: "/",
      }),
    ).toBe("http://localhost:8000/welcome")
  })
})
