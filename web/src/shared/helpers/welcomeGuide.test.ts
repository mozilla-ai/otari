import { describe, expect, it } from "vitest"
import { welcomeGuideHref } from "./welcomeGuide"

describe("welcomeGuideHref", () => {
  // The gateway process serves the bundle, and so serves `/welcome` on the
  // same origin the dashboard was loaded from.
  it("links the page a standalone deployment serves beside the dashboard", () => {
    expect(welcomeGuideHref({ deployment_type: "standalone" })).toBe("/welcome")
  })

  // Not a 404 to route around: otari.ai publishes the bundle as static files
  // from its own edge, and the page it would reach documents a bootstrap key
  // read out of a server log that a tenant has no access to.
  it("links nowhere on a hosted deployment, which serves no such page", () => {
    expect(welcomeGuideHref({ deployment_type: "hosted" })).toBeUndefined()
  })
})
