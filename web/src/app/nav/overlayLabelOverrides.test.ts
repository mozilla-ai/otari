import { describe, expect, it, vi } from "vitest"

import { NAV_SECTIONS, navLabelForPath, ORG_NAV_SECTIONS } from "./registry"

/**
 * The seam as a build that replaces the module sees it, which is the one thing
 * `registry.test.ts` cannot show: the list is empty there, so every assertion in
 * it holds whether or not the registry applies the overrides at all. Unwiring
 * `applyNavLabelOverrides` from both rails leaves that suite green and turns
 * this file red, which is the point of it.
 */
vi.mock("./overlayLabelOverrides", () => ({
  OVERLAY_NAV_LABEL_OVERRIDES: [
    {
      sectionId: "gateway",
      label: "Inference",
      disclosureLabels: { "/tools": "Built-in tools" },
    },
    // The organization rail, renamed from the same list the workspace rail
    // reads: one module for both is what a unique section id buys.
    { sectionId: "org-general", label: "Deployment" },
  ],
}))

describe("a build that replaces the label-override module", () => {
  const gateway = () => NAV_SECTIONS.find((section) => section.id === "gateway")

  it("renames a base section heading on the workspace rail", () => {
    expect(gateway()?.label).toBe("Inference")
  })

  it("renames a base disclosure label", () => {
    expect(gateway()?.items.map((item) => item.label)).toEqual([
      "Models",
      "Routing",
      "Built-in tools",
    ])
  })

  it("renames a section on the organization rail from the same list", () => {
    expect(
      ORG_NAV_SECTIONS.find((section) => section.id === "org-general")?.label,
    ).toBe("Deployment")
  })

  it("leaves every label it does not name alone", () => {
    expect(NAV_SECTIONS.map((section) => section.label)).toEqual([
      undefined,
      "Observe",
      "Inference",
      "Access",
    ])
    expect(ORG_NAV_SECTIONS.map((section) => section.label)).toEqual([
      "People & access",
      "Money",
      "Deployment",
    ])
  })

  it("leaves gating and nested destinations untouched", () => {
    // The rename must not cost a surface: an entry that lost one would be shown
    // by a deployment that does not serve it.
    const tools = gateway()?.items.find((item) => item.to === "/tools")
    expect(tools?.surface).toBe("tools")
    expect(tools?.children?.map((child) => child.label)).toEqual([
      "Web search",
      "Code execution",
    ])
    const routing = gateway()?.items.find((item) => item.to === "/routing")
    expect(routing?.children?.map((child) => child.surface)).toEqual([
      undefined,
      "tools",
    ])
  })

  it("carries a renamed group into the breadcrumbs", () => {
    // A group is a destination as well as a heading, so the rename reaches what
    // the breadcrumbs and the shell's gated-off panel call that page.
    expect(navLabelForPath("/tools")).toBe("Built-in tools")
    // A child keeps its own name; it is not part of this seam.
    expect(navLabelForPath("/tools/web-search")).toBe("Web search")
  })
})
