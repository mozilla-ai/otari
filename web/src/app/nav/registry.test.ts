import { FiBox } from "react-icons/fi"
import { describe, expect, it } from "vitest"
import { BASE_CAPABILITIES } from "@/shared/hooks/useEntitlements"
import { OVERLAY_NAV_LABEL_OVERRIDES } from "./overlayLabelOverrides"
import {
  applyNavLabelOverrides,
  composeNavSections,
  isPathVisible,
  NAV_ITEMS,
  NAV_SECTIONS,
  navContextForPath,
  navItemForPath,
  ORG_NAV_SECTIONS,
  visibleNavSections,
} from "./registry"
import type { NavItem, NavLabelOverride, NavSection } from "./types"

describe("nav registry", () => {
  it("exposes the base sections in display order", () => {
    expect(NAV_SECTIONS.map((section) => section.id)).toEqual([
      "index",
      "observe",
      "gateway",
      "access",
    ])
    // Exactly one unlabeled group, and it is the one the index sits alone in.
    // Asserted as a count rather than a property of that section, because the
    // thing worth catching is a *second* headingless group appearing: two of
    // them and the rail stops reading as one list with a landing row on top.
    expect(NAV_SECTIONS.filter((section) => !section.label)).toHaveLength(1)
    expect(NAV_SECTIONS.find((section) => !section.label)?.id).toBe("index")
  })

  it("exposes the organization sections in display order", () => {
    // The second rail, reached from the sidebar footer. Its own registry, so
    // its order is asserted separately from the workspace one's.
    expect(ORG_NAV_SECTIONS.map((section) => section.id)).toEqual([
      "org-people",
      "org-money",
      "org-gateway",
      "org-general",
    ])
  })

  it("declares every destination the standalone dashboard serves", () => {
    // The whole list, not a sample: the registry is now the only place a
    // destination exists, so an entry dropped in a refactor is a page that
    // silently stops being reachable.
    // Both rails, because NAV_ITEMS is what answers "which entry is this
    // pathname" and a route is gated the same way whichever sidebar links it.
    expect(NAV_ITEMS.map((item) => item.label)).toEqual([
      "Overview",
      "Activity",
      "Usage",
      "Models",
      "Routing",
      "Tools",
      "API keys",
      "Providers",
      "Members",
      "Workspaces",
      "Members & roles",
      "Providers",
      "Spend & budgets",
      "Billing",
      "Model pricing",
      "Guardrails",
      "Gateways",
      "Org settings",
      "Settings",
    ])
  })

  it("puts each destination on exactly one rail", () => {
    // The two registries are concatenated into NAV_ITEMS, so a path declared on
    // both would resolve to whichever came first and be gated by that one.
    const paths = NAV_ITEMS.map((item) => item.to)
    expect(paths).toEqual([...new Set(paths)])
  })

  it("sorts a pathname onto the rail that declares it", () => {
    // Not a URL-prefix rule: /workspaces and /settings are organization
    // destinations whose paths look like anything else, and /members is a
    // workspace one directly under the root.
    expect(navContextForPath("/members")).toBe("workspace")
    expect(navContextForPath("/workspaces")).toBe("organization")
    expect(navContextForPath("/settings")).toBe("organization")
    expect(navContextForPath("/organization/members")).toBe("organization")
    // Unregistered paths open in the context the shell starts in.
    expect(navContextForPath("/docs")).toBe("workspace")
  })

  it("splits the tenancy pages across their two surfaces", () => {
    // The organization pages and the workspace pages are separate management
    // prefixes, so they gate separately: a deployment that served one without
    // the other would otherwise show links it refuses every request behind.
    const tenancy = ORG_NAV_SECTIONS.find(
      (section) => section.id === "org-people",
    )
    expect(tenancy?.items.map((item) => [item.label, item.surface])).toEqual([
      ["Workspaces", "workspaces"],
      ["Members & roles", "organizations"],
      ["Providers", "organization_providers"],
    ])
    const money = ORG_NAV_SECTIONS.find((section) => section.id === "org-money")
    expect(money?.items.map((item) => [item.label, item.surface])).toEqual([
      ["Spend & budgets", "budgets"],
      ["Billing", "billing"],
      ["Model pricing", "pricing"],
    ])
    // No row gates on `users` any more. The gateway still serves that surface
    // (budgets, keys and the roster all read `/v1/users`), but a person is a
    // member now: what they may spend and what their keys may call are columns
    // on Members & roles rather than a second people-shaped destination.
    expect(NAV_ITEMS.map((item) => item.surface)).not.toContain("users")
  })

  it("resolves a child route to its own entry, not to its parent's", () => {
    // The first nested pair in the registry, and the reason navItemForPath
    // makes two passes: /organization is registered ahead of
    // /organization/members and matches it as a prefix. The shell titles its
    // gated-off panel from whatever comes back, and the sidebar highlights it,
    // so the parent winning here is a page announcing itself as "General".
    expect(navItemForPath("/organization/members")?.label).toBe(
      "Members & roles",
    )
    expect(navItemForPath("/organization")?.label).toBe("Org settings")
  })

  it("resolves a deeper path to the deepest entry above it", () => {
    // Both /organization and /organization/members are prefixes of this, and
    // the deeper one is what describes it.
    expect(navItemForPath("/organization/members/abc")?.label).toBe(
      "Members & roles",
    )
  })

  it("still resolves an unregistered child route to its parent", () => {
    // The prefix pass is what a future child route (/routing/new) relies on to
    // inherit its parent's gating; only an exact match outranks it.
    expect(navItemForPath("/routing/new")?.label).toBe("Routing")
    expect(navItemForPath("/docs")).toBeUndefined()
  })

  it("points every entry at an absolute path", () => {
    for (const item of NAV_ITEMS) {
      expect(item.to.startsWith("/")).toBe(true)
    }
  })

  it("keeps Routing gated on its surface only", () => {
    // Pinned because the tag is a decision, not an omission: otari.ai gates its
    // own Routing item on `capability: "routing"`, so adding it here would look
    // like a merge fix rather than a call on a provisional split.
    const routing = NAV_ITEMS.find((item) => item.label === "Routing")
    expect(routing?.surface).toBe("routing")
    expect(routing?.capability).toBeUndefined()
  })

  it("leaves the index ungated on both axes", () => {
    const overview = NAV_ITEMS.find((item) => item.label === "Overview")
    // The deployment's own front page: it reads whatever it is allowed to, so
    // gating it would leave a deployment with no landing page at all.
    expect(overview?.surface).toBeUndefined()
    expect(overview?.capability).toBeUndefined()
  })

  it("names the usage surface on both observability pages", () => {
    // Activity and Usage are two views over /v1/usage, so they gate together on
    // the surface rather than each on a name of its own.
    const observability = NAV_SECTIONS.find(
      (section) => section.id === "observe",
    )
    // The whole section now: the index moved out to its own group above, so
    // there is nothing ungated left in here to skip past.
    expect(observability?.items.map((item) => item.surface)).toEqual([
      "usage",
      "usage",
    ])
  })

  it("gates no base entry on a capability", () => {
    // The axis has no base user yet. Routing is the one candidate, and
    // ARCHITECTURE.md marks that split provisional, so the tag waits for the
    // decision rather than anticipating it.
    expect(NAV_ITEMS.every((item) => item.capability === undefined)).toBe(true)
  })

  it("entitles every capability the base registry gates on", () => {
    // Vacuous while the assertion above holds, and deliberately kept: it arms
    // itself the moment someone adds the first tag. A base entry gated on a
    // capability the base build does not grant disappears from the sidebar of
    // every standalone gateway, and the two lists are maintained in different
    // files, which is exactly the pair a single commit forgets to update.
    for (const item of NAV_ITEMS) {
      if (item.capability === undefined) continue
      expect(BASE_CAPABILITIES).toContain(item.capability)
    }
  })

  it("gates every declared-but-unserved destination on a surface", () => {
    // The design's organization rail draws four rows this gateway has no endpoint
    // for. Each is declared so the rail matches on a deployment that does serve
    // them, and gated on a surface `STANDALONE_SURFACES` does not report so the
    // row is absent here. Pinned as the whole set, because the failure mode is
    // silent in both directions: a missing gate ships a link to a page that
    // cannot work, and a gate on a surface the bootstrap *does* report hides a
    // page that can.
    const unserved = new Map([
      ["/organization/provider-keys", "organization_providers"],
      ["/organization/billing", "billing"],
      ["/organization/guardrails", "organization_guardrails"],
      ["/organization/gateways", "gateways"],
    ])
    for (const [to, surface] of unserved) {
      expect(navItemForPath(to)?.surface).toBe(surface)
    }
    // And none of those surface names is one a standalone gateway reports, or the
    // gate would be decoration.
    const standalone = [
      "budgets",
      "keys",
      "models",
      "organizations",
      "pricing",
      "providers",
      "routing",
      "settings",
      "tools",
      "usage",
      "users",
      "workspaces",
    ]
    for (const surface of unserved.values()) {
      expect(standalone).not.toContain(surface)
    }
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
            icon: FiBox,
            capability: "billing",
          },
        ],
      },
    ]
    expect(
      composeNavSections(base, overlay).map((section) => section.id),
    ).toEqual(["base", "overlay-billing"])
  })

  it("renames nothing in this build", () => {
    // The label seam's twin of the assertion below: the overlay tree lives in
    // another repo, so every base heading and disclosure label renders as
    // declared.
    expect(OVERLAY_NAV_LABEL_OVERRIDES).toEqual([])
    const gateway = NAV_SECTIONS.find((section) => section.id === "gateway")
    expect(gateway?.label).toBe("Gateway")
    expect(gateway?.items.map((item) => item.label)).toContain("Routing")
    const general = ORG_NAV_SECTIONS.find(
      (section) => section.id === "org-general",
    )
    expect(general?.label).toBe("General")
    expect(general?.items.map((item) => item.label)).toContain("Org settings")
  })

  it("keeps section ids unique across the two rails", () => {
    // What lets one override list address both rails: an id that appeared on
    // each would rename two sections from one entry.
    const ids = [...NAV_SECTIONS, ...ORG_NAV_SECTIONS].map(
      (section) => section.id,
    )
    expect(ids).toEqual([...new Set(ids)])
  })

  it("appends nothing in this build", () => {
    // The overlay tree lives in another repo; the seam here stays empty.
    expect(composeNavSections(NAV_SECTIONS, [])).toEqual(NAV_SECTIONS)
  })
})

describe("visibleNavSections", () => {
  const entry = (label: string, surface?: string): NavItem =>
    ({ to: "/", label, icon: FiBox, surface }) as NavItem

  const sections: NavSection[] = [
    { id: "first", items: [entry("A", "a")] },
    { id: "second", items: [entry("B", "b"), entry("C", "c")] },
    { id: "third", items: [entry("D")] },
  ]

  const hosting =
    (...surfaces: string[]) =>
    (item: NavItem) =>
      item.surface === undefined || surfaces.includes(item.surface)

  it("keeps only the entries that pass, and drops a section left with none", () => {
    const visible = visibleNavSections(sections, hosting("b"))
    expect(visible.map(({ section }) => section.id)).toEqual([
      "second",
      "third",
    ])
    expect(visible[0].items.map((item) => item.label)).toEqual(["B"])
  })

  it("indexes by rendered position, not registry position", () => {
    // The sidebar draws a divider and a top margin above every section after
    // the first, so the first surviving section has to land at index 0. Keyed
    // off the registry index instead, a hidden section ahead of it would leave
    // a stray top border above the first visible group. Not reachable through
    // today's registry, where the ungated index section always renders first,
    // and reachable the moment an overlay contributes a section.
    const visible = visibleNavSections(sections, hosting("c"))
    expect(visible[0].section.id).toBe("second")
    expect(visible.findIndex(({ section }) => section.id === "second")).toBe(0)
  })

  it("returns nothing when every entry is gated away", () => {
    expect(visibleNavSections([sections[0]], hosting())).toEqual([])
  })

  it("keeps an entry that declares no gate at all", () => {
    const visible = visibleNavSections(sections, hosting())
    expect(visible.map(({ section }) => section.id)).toEqual(["third"])
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

describe("isPathVisible", () => {
  // The predicate the shell composes, narrowed to the deployment axis: a
  // surface this test withholds stands for one the bootstrap does not report.
  const without = (surface: string) => (item: NavItem) =>
    item.surface !== surface

  it("serves a destination whose surface the deployment reports", () => {
    expect(isPathVisible("/routing", without("keys"))).toBe(true)
  })

  it("refuses a destination whose surface it does not", () => {
    expect(isPathVisible("/routing", without("routing"))).toBe(false)
  })

  it("gates a nested destination on its own surface, not its group's", () => {
    // Guardrails is grouped under Routing and served by the tools surface, so
    // the tools surface is the one that decides. Withholding `routing` leaves
    // the page served (and the rail without a row for it, which is the grouping
    // showing through); withholding `tools` is what refuses it.
    expect(isPathVisible("/tools/guardrails", without("tools"))).toBe(false)
    expect(isPathVisible("/tools/guardrails", without("routing"))).toBe(true)
  })

  it("inherits the gate of the destination a deeper path sits under", () => {
    expect(isPathVisible("/routing/new", without("routing"))).toBe(false)
  })

  it("leaves a path the registry does not declare ungated", () => {
    // The guide and the 404 splat: the registry governs what it declares and
    // nothing else, so withholding every surface must not gate them.
    expect(isPathVisible("/docs", () => false)).toBe(true)
    expect(isPathVisible("/nope", () => false)).toBe(true)
  })
})

describe("applyNavLabelOverrides", () => {
  const base: NavSection[] = [
    {
      id: "gateway",
      label: "Gateway",
      items: [
        // A real mark rather than a placeholder: `icon` is a `react-icons`
        // `IconType` on an item and on a child alike, so the fixture has to be a
        // shape the registry could actually hold.
        { to: "/models", label: "Models", surface: "models", icon: FiBox },
        {
          to: "/routing",
          label: "Routing",
          surface: "routing",
          icon: FiBox,
          children: [{ to: "/routing", label: "Policies", icon: FiBox }],
        },
      ],
    },
  ]

  const only = (overrides: readonly NavLabelOverride[]) =>
    applyNavLabelOverrides(base, overrides)[0]

  it("is a no-op with no overrides", () => {
    // Identity, not a copy: pins the short-circuit, so the seam this build
    // ships costs the base sections nothing at all.
    expect(applyNavLabelOverrides(base, [])).toBe(base)
  })

  it("renames a section heading and a disclosure label together", () => {
    const section = only([
      {
        sectionId: "gateway",
        label: "Inference",
        disclosureLabels: { "/routing": "Policies & guardrails" },
      },
    ])
    expect(section.label).toBe("Inference")
    expect(section.items.map((item) => item.label)).toEqual([
      "Models",
      "Policies & guardrails",
    ])
  })

  it("applies only the field that is set", () => {
    expect(only([{ sectionId: "gateway", label: "Inference" }]).label).toBe(
      "Inference",
    )
    const disclosureOnly = only([
      { sectionId: "gateway", disclosureLabels: { "/routing": "Policies" } },
    ])
    expect(disclosureOnly.label).toBe("Gateway")
    expect(disclosureOnly.items[1].label).toBe("Policies")
  })

  it("leaves ids, icons, gating, and nested destinations untouched", () => {
    const section = only([
      {
        sectionId: "gateway",
        label: "Inference",
        disclosureLabels: { "/routing": "Policies" },
      },
    ])
    expect(section.id).toBe("gateway")
    const routing = section.items[1]
    expect(routing.to).toBe("/routing")
    expect(routing.surface).toBe("routing")
    expect(routing.children).toBe(base[0].items[1].children)
    // An untargeted item is the same object, not a rebuilt one.
    expect(section.items[0]).toBe(base[0].items[0])
  })

  it("ignores an override whose sectionId matches no section", () => {
    expect(
      applyNavLabelOverrides(base, [
        { sectionId: "nope", label: "X", disclosureLabels: { "/": "Y" } },
      ]),
    ).toEqual(base)
  })

  it("ignores a path that is not a disclosure in this section", () => {
    // /models is a plain link and /keys belongs to another section: relabeling
    // either would be renaming a destination rather than a group.
    const section = only([
      {
        sectionId: "gateway",
        disclosureLabels: { "/models": "Catalog", "/keys": "Credentials" },
      },
    ])
    expect(section.items.map((item) => item.label)).toEqual([
      "Models",
      "Routing",
    ])
  })
})
