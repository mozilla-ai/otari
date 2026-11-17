import { describe, expect, it, vi } from "vitest"

import {
  NAV_SECTIONS,
  navItemForPath,
  navLabelForPath,
  ORG_NAV_SECTIONS,
} from "./registry"

/**
 * The seam as a build that replaces the module sees it, which is the one thing
 * `registry.test.ts` cannot show: the list is empty there, so every assertion in
 * it holds whether or not the registry applies the overrides at all. Unwiring
 * `applyNavLabelOverrides` from both rails leaves that suite green and turns
 * this file red, which is the point of it.
 */
vi.mock("./overlayLabelOverrides", () => ({
  OVERLAY_NAV_LABEL_OVERRIDES: [
    // The workspace rail's one section, which has no heading of its own.
    { sectionId: "index", label: "Workspace" },
    // The organization rail, renamed from the same list the workspace rail
    // reads: one module for both is what a unique section id buys.
    {
      sectionId: "org-general",
      label: "Deployment",
      disclosureLabels: { "/organization/provider-keys": "Provider keys" },
    },
  ],
}))

describe("a build that replaces the label-override module", () => {
  const general = () =>
    ORG_NAV_SECTIONS.find((section) => section.id === "org-general")

  it("renames a base section heading on the workspace rail", () => {
    expect(NAV_SECTIONS.find((section) => section.id === "index")?.label).toBe(
      "Workspace",
    )
  })

  it("renames a base disclosure label", () => {
    expect(general()?.items.map((item) => item.label)).toEqual([
      "Provider keys",
      "Guardrails",
      "Org settings",
    ])
  })

  it("renames a section on the organization rail from the same list", () => {
    expect(general()?.label).toBe("Deployment")
  })

  it("leaves every label it does not name alone", () => {
    expect(NAV_SECTIONS.map((section) => section.label)).toEqual(["Workspace"])
    expect(
      NAV_SECTIONS.flatMap((section) => section.items).map(
        (item) => item.label,
      ),
    ).toEqual([
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
    ])
    expect(ORG_NAV_SECTIONS.map((section) => section.label)).toEqual([
      "People & access",
      "Cost & billing",
      // The one the mocked list renames; the two before it are untouched.
      "Deployment",
    ])
  })

  it("leaves gating and nested destinations untouched", () => {
    // The rename must not cost a surface: an entry that lost one would be shown
    // by a deployment that does not serve it.
    const providers = general()?.items.find(
      (item) => item.to === "/organization/provider-keys",
    )
    expect(providers?.surface).toBe("organization_providers")
    expect(providers?.children?.map((child) => child.label)).toEqual([
      "Inference",
      "Web search",
    ])
  })

  it("renames the group, and leaves the pages under it their own names", () => {
    // The group is the entry the gate and the rail's disclosure read.
    expect(navItemForPath("/organization/provider-keys")?.label).toBe(
      "Provider keys",
    )
    // Its first child shares its address, so the breadcrumb and the shell's
    // gated-off panel name that page after the child, which is not part of
    // this seam.
    expect(navLabelForPath("/organization/provider-keys")).toBe("Inference")
    expect(navLabelForPath("/organization/web-search-keys")).toBe("Web search")
  })
})
