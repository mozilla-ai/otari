import { describe, expect, it } from "vitest"

import { BASE_CAPABILITIES } from "@/shared/hooks/useEntitlements"

import {
  composeNavSections,
  NAV_ITEMS,
  NAV_SECTIONS,
  navItemForPath,
} from "./registry"
import type { NavSection } from "./types"

describe("nav registry", () => {
  it("exposes the base sections in display order", () => {
    expect(NAV_SECTIONS.map((section) => section.id)).toEqual([
      "home",
      "observability",
      "catalog",
      "access",
      "system",
    ])
  })

  it("declares every destination the standalone dashboard serves", () => {
    // The whole list, not a sample: the registry is now the only place a
    // destination exists, so an entry dropped in a refactor is a page that
    // silently stops being reachable.
    expect(NAV_ITEMS.map((item) => item.label)).toEqual([
      "Overview",
      "Activity",
      "Usage",
      "Providers",
      "Models",
      "Routing",
      "Users",
      "API keys",
      "Budgets",
      "Tools & Guardrails",
      "Settings",
    ])
  })

  it("points every entry at an absolute path", () => {
    for (const item of NAV_ITEMS) {
      expect(item.to.startsWith("/")).toBe(true)
    }
  })

  it("leaves the index ungated on all three axes", () => {
    const overview = NAV_ITEMS.find((item) => item.label === "Overview")
    // The deployment's own front page: it reads whatever it is allowed to, so
    // gating it would leave a deployment with no landing page at all.
    expect(overview?.surface).toBeUndefined()
    expect(overview?.capability).toBeUndefined()
    expect(overview?.flag).toBeUndefined()
  })

  it("names the usage surface on both observability pages", () => {
    // Activity and Usage are two views over /v1/usage, so they gate together on
    // the surface rather than each on a name of its own.
    const observability = NAV_SECTIONS.find(
      (section) => section.id === "observability",
    )
    expect(observability?.items.map((item) => item.surface)).toEqual([
      "usage",
      "usage",
    ])
  })

  it("entitles every capability the base registry gates on", () => {
    // A base entry gated on a capability the base build does not grant would
    // disappear from the sidebar of every standalone gateway. The two lists are
    // maintained in different files, so this is what keeps them honest.
    const gated = NAV_ITEMS.map((item) => item.capability).filter(
      (capability) => capability !== undefined,
    )
    expect(gated).not.toHaveLength(0)
    for (const capability of gated) {
      expect(BASE_CAPABILITIES).toContain(capability)
    }
  })

  it("gates no base entry on a feature flag", () => {
    // The base build ships no flags, so a flagged entry would be permanently
    // hidden. Flags belong to whoever is rolling something out.
    expect(NAV_ITEMS.every((item) => item.flag === undefined)).toBe(true)
  })

  it("appends overlay sections after the base sections", () => {
    const base: NavSection[] = [{ id: "base", items: [] }]
    const overlay: NavSection[] = [
      {
        id: "overlay-billing",
        items: [
          {
            to: "/settings",
            label: "Billing",
            icon: null,
            capability: "billing",
          },
        ],
      },
    ]
    expect(
      composeNavSections(base, overlay).map((section) => section.id),
    ).toEqual(["base", "overlay-billing"])
  })

  it("appends nothing in this build", () => {
    // The overlay tree lives in another repo; the seam here stays empty.
    expect(composeNavSections(NAV_SECTIONS, [])).toEqual(NAV_SECTIONS)
  })
})

describe("navItemForPath", () => {
  it("matches a registered destination exactly", () => {
    expect(navItemForPath("/routing")?.label).toBe("Routing")
  })

  it("matches a child path to its parent entry", () => {
    // A future /routing/new inherits the gating of the destination it sits under
    // rather than escaping it by being unregistered.
    expect(navItemForPath("/routing/new")?.label).toBe("Routing")
  })

  it("matches the index only exactly", () => {
    expect(navItemForPath("/")?.label).toBe("Overview")
    // As a prefix, "/" would claim every path in the dashboard and gate the
    // whole shell on the index's (absent) gating.
    expect(navItemForPath("/keys")?.label).toBe("API keys")
  })

  it("returns nothing for a path the registry does not declare", () => {
    // The bundled guide and the 404 splat are not destinations, so they are
    // never gated.
    expect(navItemForPath("/docs")).toBeUndefined()
    expect(navItemForPath("/nope")).toBeUndefined()
  })
})
