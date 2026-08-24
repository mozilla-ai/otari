import { FiCreditCard, FiServer } from "react-icons/fi"
import { describe, expect, it, vi } from "vitest"

import { NAV_SECTIONS, ORG_NAV_SECTIONS } from "./registry"
import type { NavSection } from "./types"

/**
 * The seam as a build that replaces the module sees it, which is the one thing
 * `registry.test.ts` cannot show: the list is empty there, so its assertions
 * hold whether or not the registry applies the contributions at all. Unwiring
 * `composeNavItems` from either rail leaves that suite green and turns this file
 * red, which is the point of it, and the organization rail is the one that
 * matters most: Billing and Gateways are why the seam exists and both land
 * there.
 *
 * `to` borrows a base path on purpose. The real destinations are paths only the
 * composed build has a route for, and `NavPath` resolves against *this* build's
 * route tree; what is under test is where a contributed row lands, not where it
 * points.
 */
vi.mock("./overlayNavItems", () => ({
  OVERLAY_NAV_ITEMS: [
    {
      sectionId: "org-money",
      items: [{ to: "/organization", label: "Billing", icon: FiCreditCard }],
    },
    {
      sectionId: "org-gateway",
      items: [{ to: "/organization", label: "Gateways", icon: FiServer }],
    },
    // The workspace rail, contributed from the same list the organization rail
    // reads: one module for both is what a unique section id buys.
    {
      sectionId: "observe",
      items: [{ to: "/usage", label: "Reports", icon: FiServer }],
    },
  ],
}))

describe("a build that replaces the nav-item module", () => {
  const section = (sections: readonly NavSection[], id: string) =>
    sections.find((one) => one.id === id)

  it("appends a contributed row after the organization section's own", () => {
    expect(
      section(ORG_NAV_SECTIONS, "org-money")?.items.map((item) => item.label),
    ).toEqual(["Spend & budgets", "Model pricing", "Billing"])
  })

  it("appends into a second organization section from the same list", () => {
    expect(
      section(ORG_NAV_SECTIONS, "org-gateway")?.items.map((item) => item.label),
    ).toEqual(["Guardrails", "Gateways"])
  })

  it("appends into the workspace rail from that same list", () => {
    expect(
      section(NAV_SECTIONS, "observe")?.items.map((item) => item.label),
    ).toEqual(["Activity", "Usage", "Reports"])
  })

  it("leaves every section no contribution names alone", () => {
    expect(
      section(ORG_NAV_SECTIONS, "org-people")?.items.map((item) => item.label),
    ).toEqual(["Workspaces", "Members & roles", "Providers"])
    expect(
      section(NAV_SECTIONS, "gateway")?.items.map((item) => item.label),
    ).toEqual(["Models", "Routing", "Tools"])
  })

  it("leaves a contributed row ungated unless it says otherwise", () => {
    // A contribution carries its own gating, as a base row does. Neither rail
    // adds one, so an overlay's row is shown wherever the overlay mounts it.
    const billing = section(ORG_NAV_SECTIONS, "org-money")?.items.find(
      (item) => item.label === "Billing",
    )
    expect(billing?.surface).toBeUndefined()
    expect(billing?.capability).toBeUndefined()
  })
})
