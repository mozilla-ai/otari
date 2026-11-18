import { FiBox } from "react-icons/fi"
import { describe, expect, it } from "vitest"
import { BASE_CAPABILITIES } from "@/shared/hooks/useEntitlements"
import { organizationContext } from "@/tests/fixtures"
import { OVERLAY_NAV_LABEL_OVERRIDES } from "./overlayLabelOverrides"
import { OVERLAY_NAV_ITEMS } from "./overlayNavItems"
import { OVERLAY_DEPLOYMENT_NAV_SECTIONS } from "./overlaySections"
import {
  applyNavLabelOverrides,
  composeNavItems,
  composeNavSections,
  DEPLOYMENT_NAV_SECTIONS,
  DRAWER_NAV_SECTIONS,
  isPathVisible,
  NAV_ITEMS,
  NAV_SECTIONS,
  navContextForPath,
  navItemForPath,
  navPageForPath,
  ORG_NAV_SECTIONS,
  PLAYGROUND_NAV_ITEM,
  pageLabelForPath,
  railLabelForPath,
  railPathForPath,
  visibleNavSections,
} from "./registry"
import type {
  NavItem,
  NavItemContribution,
  NavLabelOverride,
  NavSection,
} from "./types"
import { type Caller, routeAccess } from "./useNavVisibility"

describe("nav registry", () => {
  it("gates the Playground route without placing it in a sidebar", () => {
    expect(
      NAV_SECTIONS.flatMap((section) => section.items).some(
        (item) => item.to === "/playground",
      ),
    ).toBe(false)
    expect(navItemForPath("/playground")?.surface).toBe("playground")
    expect(isPathVisible("/playground", () => false)).toBe(false)
  })

  it("draws the workspace rail as one headingless section with no groups", () => {
    // Every row is a place in the selected workspace and none is a category,
    // so a heading or a disclosure here would be a second level the rail does
    // not have.
    expect(NAV_SECTIONS.map((section) => section.id)).toEqual(["index"])
    expect(NAV_SECTIONS[0].label).toBeUndefined()
    const items = NAV_SECTIONS.flatMap((section) => section.items)
    expect(items.map((item) => item.to)).toEqual([
      "/",
      "/sessions",
      "/people",
      "/rules",
      "/models",
      "/tools",
      "/keys",
    ])
    expect(items.flatMap((item) => item.children ?? [])).toEqual([])
  })

  it("draws the mobile drawer with the desk's rows", () => {
    // The Playground is offered by the account menu on a phone, so the drawer
    // carries no row of its own for it.
    expect(DRAWER_NAV_SECTIONS).toBe(NAV_SECTIONS)
  })

  it("exposes the organization sections in display order", () => {
    // The second rail, reached from the sidebar footer. Its own registry, so
    // its order is asserted separately from the workspace one's.
    expect(ORG_NAV_SECTIONS.map((section) => section.id)).toEqual([
      "org-people",
      "org-money",
      "org-general",
    ])
  })

  it("declares every destination the dashboard serves", () => {
    // The whole list, not a sample: the registry is now the only place a
    // destination exists, so an entry dropped in a refactor is a page that
    // silently stops being reachable.
    // Both rails, because NAV_ITEMS is what answers "which entry is this
    // pathname" and a route is gated the same way whichever sidebar links it.
    // The destinations with no row come right after the Playground, ahead of
    // every rail.
    expect(NAV_ITEMS.map((item) => item.label)).toEqual([
      "Playground",
      "Members",
      "Routing",
      "Guardrails",
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
      "Workspaces",
      "Members & roles",
      "Email domains",
      "Single sign-on",
      "Spend & budgets",
      "Providers",
      "Guardrails",
      "Org settings",
      "Settings",
      "Accounts",
      "Hosted providers",
      "Subagent router",
      "Web search",
      "Code execution pricing",
      "Health",
    ])
  })

  it("declares the operator-only destinations and no others", () => {
    // The third gating axis. It named one row until otari-ai#1880 gated the
    // deployment-wide routers on `require_deployment_operator`: a page built
    // entirely on those answers 403 to a member, so the rail stops offering it.
    // The list is pinned rather than derived because adding to it is a design
    // decision, not a refactor. A page that is only *partly* deployment-wide
    // stays off it on purpose (organization members, workspaces, the workspace
    // tool pages): hiding a destination a member does use costs more than a
    // panel on it reporting its own refusal.
    //
    // Activity and Usage left the list in otari#837, which is the other way a
    // row leaves it: not by loosening a gate but by the page behind it gaining a
    // tenant-scoped read, so there is no longer a caller it refuses. Models and
    // Routing left it the same way (otari-ai#1942): the catalog reads already
    // served any session, and Routing gained
    // /api/v1/organizations/me/routing-policies. API keys left it for the same
    // reason (otari-ai#1941): members create and manage their own keys through
    // /api/v1/organizations/me/keys. Removing a row from here is as
    // much a design decision as adding one.
    //
    // Model pricing left it a third way, and then left the rail entirely. It was
    // never one answer: its rate overrides were the organization's own and its
    // catalog read served any session, so the page withheld its operator-only
    // sections rather than the rail withholding the destination. Rates live on
    // Providers instead, beside the models they price.
    //
    // Spend & budgets left it a fourth way, in the same issue: the route now
    // resolves to *two* pages, the deployment's for an operator and the
    // organization's own for an owner or admin, so there is no caller the
    // destination refuses. That is the shape to copy for the rows still on this
    // list, not a loosened gate.
    expect(
      NAV_ITEMS.filter((item) => item.operatorOnly).map((item) => item.to),
    ).toEqual([
      "/settings",
      "/admin/accounts",
      "/admin/hosted-providers",
      "/admin/recommender",
      "/admin/web-search",
      "/admin/code-execution",
      "/health-checks",
    ])
  })

  it("puts each destination on exactly one rail", () => {
    // The two registries are concatenated into NAV_ITEMS, so a path declared on
    // both would resolve to whichever came first and be gated by that one.
    const paths = NAV_ITEMS.map((item) => item.to)
    expect(paths).toEqual([...new Set(paths)])
  })

  it("puts the deployment destinations on their own rail", () => {
    expect(
      DEPLOYMENT_NAV_SECTIONS.flatMap((section) =>
        section.items.map((item) => item.to),
      ),
    ).toEqual([
      "/settings",
      "/admin/accounts",
      "/admin/hosted-providers",
      "/admin/recommender",
      "/admin/web-search",
      "/admin/code-execution",
      "/health-checks",
    ])
    // And on neither of the others. A destination drawn on two rails would
    // resolve to whichever came first in NAV_ITEMS and be gated by that one.
    const others = [...NAV_SECTIONS, ...ORG_NAV_SECTIONS].flatMap((section) =>
      section.items.map((item) => item.to),
    )
    expect(others).not.toContain("/settings")
    expect(others).not.toContain("/admin/accounts")
    expect(others).not.toContain("/admin/hosted-providers")
    expect(others).not.toContain("/admin/recommender")
    expect(others).not.toContain("/admin/web-search")
    expect(others).not.toContain("/admin/code-execution")
    expect(others).not.toContain("/health-checks")
  })

  it("keeps a deployment destination registered, and therefore gated", () => {
    // The failure this exists to catch: expressing "not on the rail" by
    // deleting the entry. `NAV_ITEMS` is what `navItemForPath` answers from, so
    // an unregistered path is an *ungated* one, and the shell would render
    // these pages on a deployment that serves neither surface instead of
    // refusing them. Nothing about the rail being empty says so, which is why
    // it is asserted here rather than left to the rail tests.
    for (const [to, surface, operatorOnly] of [
      ["/settings", "settings", "refused"],
      ["/admin/accounts", "admin", "unlisted"],
      ["/admin/hosted-providers", "hosted_providers", "unlisted"],
      ["/admin/recommender", "agent_recommenders", "unlisted"],
      ["/admin/web-search", "tools", "refused"],
      ["/admin/code-execution", "pricing", "refused"],
      ["/health-checks", "health_checks", "refused"],
    ] as const) {
      const item = navItemForPath(to)
      expect(item?.to).toBe(to)
      expect(item?.surface).toBe(surface)
      // The two values encode which refusal the server gives, 403 against 404.
      // Flattening them would make the menu lie about one of the two.
      expect(item?.operatorOnly).toBe(operatorOnly)
      // And owned by the deployment rail, not the organization one it used to
      // sit inside. Asserted as the rule rather than as the value: the first
      // version of this line pinned "organization" on the strength of it being
      // what the code returned, and that is how a page ended up opening a rail
      // whose scope did not own it.
      expect(navContextForPath(to)).toBe("deployment")
    }
  })

  it("sorts a pathname onto the rail that declares it", () => {
    // Not a URL-prefix rule: /workspaces and /settings are organization
    // destinations whose paths look like anything else, and /members is a
    // workspace one directly under the root.
    expect(navContextForPath("/members")).toBe("workspace")
    expect(navContextForPath("/workspaces")).toBe("organization")
    // Its own context, not the organization's: the deployment's pages describe
    // the process every tenant shares. This line said "organization" while that
    // was merely what the code did, and an asserted value is indistinguishable
    // from a considered one.
    expect(navContextForPath("/settings")).toBe("deployment")
    expect(navContextForPath("/organization/members")).toBe("organization")
    // Unregistered paths open in the context the shell starts in.
    expect(navContextForPath("/account")).toBe("workspace")
  })

  it("places providers above Org settings in General", () => {
    const path = "/organization/provider-keys"
    const visible = (item: NavItem) =>
      item.surface === "organization_providers" ||
      item.surface === "organizations"
    const general = visibleNavSections(ORG_NAV_SECTIONS, visible).find(
      ({ section }) => section.id === "org-general",
    )
    expect(general?.items.map((item) => item.to)).toEqual([
      path,
      "/organization",
    ])
    expect(navContextForPath(path)).toBe("organization")
    expect(navContextForPath(`${path}/detail`)).toBe("organization")
    expect(
      visibleNavSections(NAV_SECTIONS, visible).flatMap(({ items }) => items),
    ).toEqual([])
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
      ["Email domains", "organizations"],
      ["Single sign-on", "organizations"],
    ])
    // One row. An organization's rates are set on Providers, beside the models
    // they price, and `/organization/pricing` is a redirect, which a rail row
    // may not point at.
    const money = ORG_NAV_SECTIONS.find((section) => section.id === "org-money")
    expect(money?.items.map((item) => [item.label, item.surface])).toEqual([
      ["Spend & budgets", "budgets"],
    ])
    // No row gates on `users` any more. The gateway still serves that surface
    // (budgets, keys and the roster all read /api/v1/users), but a person is a
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
    expect(navItemForPath("/account")).toBeUndefined()
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
    const overview = NAV_ITEMS.find((item) => item.label === "Home")
    // The deployment's own front page: it reads whatever it is allowed to, so
    // gating it would leave a deployment with no landing page at all.
    expect(overview?.surface).toBeUndefined()
    expect(overview?.capability).toBeUndefined()
  })

  it("files Requests under Sessions rather than a row of its own", () => {
    // Sessions and Requests are two views under one title, each gated on its
    // own surface inside the page, so Requests adds no row.
    const sessions = NAV_ITEMS.find((item) => item.label === "Sessions")
    expect(sessions?.views).toEqual(["/requests"])
    expect(NAV_ITEMS.map((item) => item.to)).not.toContain("/requests")
    expect(railPathForPath("/requests")).toBe("/sessions")
  })

  it("declares who in the organization may open a destination, and no others", () => {
    // Pinned as the whole list, like the operator-only one: a page someone may
    // not open is in none of their navigation and its address goes Home, so a
    // `requires` added or dropped changes who can reach a page at all. The
    // organization rail is a manager's without saying so; Budgets says more
    // because it serves the operator too.
    expect(
      NAV_ITEMS.filter((item) => item.requires).map((item) => [
        item.to,
        item.requires,
      ]),
    ).toEqual([
      ["/playground", "contributor"],
      ["/tools/guardrails", "manager"],
      ["/budgets", "manager-or-operator"],
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
    // every gateway, and the two lists are maintained in different
    // files, which is exactly the pair a single commit forgets to update.
    for (const item of NAV_ITEMS) {
      if (item.capability === undefined) continue
      expect(BASE_CAPABILITIES).toContain(item.capability)
    }
  })

  it("gates every organization destination on a surface the bootstrap reports", () => {
    // Pinned as the whole rail, because the failure mode is silent in both
    // directions: a missing gate ships a link to a page that cannot work, and a
    // gate on a surface the bootstrap does not report hides a page that can. A
    // typo in a surface name is silent the same way, since `NavItemBase.surface`
    // is a bare string.
    const surfaces = [
      "admin",
      "budgets",
      "keys",
      "mcp_catalog",
      "models",
      "organization_guardrails",
      "organization_providers",
      "organization_usage",
      "organizations",
      "pricing",
      "routing",
      "settings",
      "tools",
      "usage",
      "users",
      "workspaces",
    ]
    for (const item of ORG_NAV_SECTIONS.flatMap(({ items }) => items)) {
      expect(surfaces).toContain(item.surface)
    }
    expect(navItemForPath("/organization/provider-keys")?.surface).toBe(
      "organization_providers",
    )
    expect(navItemForPath("/organization/web-search-keys")?.surface).toBe(
      "organization_providers",
    )
    expect(navItemForPath("/organization/guardrails")?.surface).toBe(
      "organization_guardrails",
    )
  })

  it("declares no destination an overlay owns", () => {
    // Billing is the row the design draws that this registry deliberately does
    // not declare: it is ARCHITECTURE.md's overlay-only capability. Declaring
    // it back is not the harmless
    // redundancy it looks like. The route file a nav entry needs would sit at a
    // path the overlay's own route file also claims, and the generator refuses
    // that pair ("Conflicting configuration paths were found") rather than
    // letting the overlay shadow the base, so the composed build gets no route
    // tree for the overlay's page. An overlay contributes it through
    // `OVERLAY_NAV_ITEMS`.
    expect(NAV_ITEMS.map((item) => item.to)).not.toContain(
      "/organization/billing",
    )
    expect(NAV_ITEMS.map((item) => item.surface)).not.toContain("billing")
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
    // The index is the id overlays address on the workspace rail, and it has no
    // heading of its own to rename.
    const index = NAV_SECTIONS.find((section) => section.id === "index")
    expect(index?.label).toBeUndefined()
    expect(index?.items.map((item) => item.label)).toEqual([
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
    ])
    const general = ORG_NAV_SECTIONS.find(
      (section) => section.id === "org-general",
    )
    expect(general?.label).toBe("General")
    expect(general?.items.map((item) => item.to)).toEqual([
      "/organization/provider-keys",
      "/organization/guardrails",
      "/organization",
    ])
    expect(general?.items.map((item) => item.label)).toEqual([
      "Providers",
      "Guardrails",
      "Org settings",
    ])
    // Providers groups the organization's two kinds of provider key.
    expect(
      general?.items[0]?.children?.map((child) => [child.label, child.to]),
    ).toEqual([
      ["Inference", "/organization/provider-keys"],
      ["Web search", "/organization/web-search-keys"],
    ])
  })

  it("keeps section ids unique across all three rails", () => {
    // What lets one override list and one contribution list address every
    // rail: an id that appeared on two would rename both sections, or land one
    // contribution's rows twice, from a single entry.
    const ids = [
      ...NAV_SECTIONS,
      ...ORG_NAV_SECTIONS,
      ...DEPLOYMENT_NAV_SECTIONS,
    ].map((section) => section.id)
    expect(ids).toEqual([...new Set(ids)])
  })

  it("appends nothing in this build", () => {
    // The overlay tree lives in another repo; the seams here stay empty.
    expect(composeNavSections(NAV_SECTIONS, [])).toEqual(NAV_SECTIONS)
    expect(OVERLAY_NAV_ITEMS).toEqual([])
    expect(OVERLAY_DEPLOYMENT_NAV_SECTIONS).toEqual([])
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
    expect(navItemForPath("/")?.label).toBe("Home")
    // As a prefix, "/" would claim every path in the dashboard and gate the
    // whole shell on the index's (absent) gating.
    expect(navItemForPath("/keys")?.label).toBe("API keys")
  })

  it("returns nothing for a path the registry does not declare", () => {
    // The account page and the 404 splat are not destinations, so they are
    // never gated.
    expect(navItemForPath("/account")).toBeUndefined()
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

  it("gates a destination with no row on its own surface, not its row's", () => {
    // The workspace's guardrails open from Rules and are served by the tools
    // surface, so the tools surface is the one that decides. Withholding
    // `routing` leaves the page served; withholding `tools` is what refuses it.
    expect(isPathVisible("/tools/guardrails", without("tools"))).toBe(false)
    expect(isPathVisible("/tools/guardrails", without("routing"))).toBe(true)
  })

  it("gates a nested destination on the entry it sits under", () => {
    // A child declares no gate of its own here, so the group's surface decides.
    expect(
      isPathVisible(
        "/organization/web-search-keys",
        without("organization_providers"),
      ),
    ).toBe(false)
    expect(
      isPathVisible("/organization/web-search-keys", without("organizations")),
    ).toBe(true)
  })

  it("inherits the gate of the destination a deeper path sits under", () => {
    expect(isPathVisible("/routing/new", without("routing"))).toBe(false)
  })

  it("answers a view of Sessions with Sessions", () => {
    expect(navItemForPath("/requests")?.label).toBe("Sessions")
    expect(navItemForPath("/sessions")?.label).toBe("Sessions")
    // Ungated, like the page it is a view of: each tab gates itself.
    expect(isPathVisible("/sessions", without("traces"))).toBe(true)
  })

  it("leaves a path the registry does not declare ungated", () => {
    // The account page and the 404 splat: the registry governs what it declares and
    // nothing else, so withholding every surface must not gate them.
    expect(isPathVisible("/account", () => false)).toBe(true)
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

describe("composeNavItems", () => {
  const base: NavSection[] = [
    {
      id: "org-money",
      label: "Cost & billing",
      items: [
        { to: "/budgets", label: "Spend & budgets", icon: FiBox },
        { to: "/organization/usage", label: "Usage", icon: FiBox },
      ],
    },
    { id: "org-general", label: "General", items: [] },
  ]

  // The contribution the seam exists for: a row inside a section the base owns.
  // Its real `to` is a path only the composed build has a route for, and
  // `NavPath` resolves against *this* build's route tree, so the fixture borrows
  // a base path. What is under test is where the row lands, not where it points.
  const billing: NavItemContribution = {
    sectionId: "org-money",
    items: [{ to: "/organization", label: "Billing", icon: FiBox }],
  }

  it("is a no-op with no contributions", () => {
    // Identity, not a copy: this build contributes nothing, so the seam must
    // cost the base sections nothing at all.
    expect(composeNavItems(base, [])).toBe(base)
  })

  it("appends contributed rows after the section's own", () => {
    const composed = composeNavItems(base, [billing])
    expect(composed[0].items.map((item) => item.label)).toEqual([
      "Spend & budgets",
      "Usage",
      "Billing",
    ])
    // The section keeps everything else it declared, heading included.
    expect(composed[0].label).toBe("Cost & billing")
  })

  it("leaves a section no contribution names untouched", () => {
    // The same object, not a rebuilt one, so an overlay contributing to one
    // section does not churn the rest of the rail.
    expect(composeNavItems(base, [billing])[1]).toBe(base[1])
  })

  it("drops a contribution whose section the registry does not declare", () => {
    // A stale contribution costs an overlay the rows it named, as a stale
    // NavLabelOverride costs it a rename. It does not throw and it does not
    // invent a section, which would put an unheaded group on the rail.
    const composed = composeNavItems(base, [
      { sectionId: "org-nope", items: [{ to: "/", label: "X", icon: FiBox }] },
    ])
    expect(composed.map((section) => section.id)).toEqual([
      "org-money",
      "org-general",
    ])
    expect(composed.flatMap((section) => section.items)).toEqual(
      base.flatMap((section) => section.items),
    )
  })

  it("lands two contributions naming one section in list order", () => {
    const composed = composeNavItems(base, [
      billing,
      {
        sectionId: "org-money",
        items: [{ to: "/settings", label: "Invoices", icon: FiBox }],
      },
    ])
    expect(composed[0].items.map((item) => item.label)).toEqual([
      "Spend & budgets",
      "Usage",
      "Billing",
      "Invoices",
    ])
  })
})

describe("pageLabelForPath", () => {
  it("names a rail page and a chrome page from one place", () => {
    expect(pageLabelForPath("/keys")).toBe("API keys")
    expect(pageLabelForPath("/sessions")).toBe("Sessions")
    expect(pageLabelForPath("/invitations")).toBe("Invitations")
    expect(pageLabelForPath("/account")).toBe("Account")
    expect(pageLabelForPath("/nowhere")).toBeUndefined()
  })
})

describe("navPageForPath", () => {
  it("files a record's page under the child destination it belongs to", () => {
    // A child wins over its group for a path below it.
    expect(navPageForPath("/organization/web-search-keys/abc")).toEqual({
      to: "/organization/web-search-keys",
      label: "Web search",
    })
    expect(pageLabelForPath("/organization/web-search-keys/abc")).toBe(
      "Web search",
    )
  })

  it("files a tool's pages under Tools, the row they open from", () => {
    expect(navPageForPath("/tools/mcp-servers/abc")).toEqual({
      to: "/tools",
      label: "Tools",
    })
    expect(pageLabelForPath("/tools/web-search")).toBe("Tools")
  })

  it("names a destination with no row after itself, not after its row", () => {
    expect(navPageForPath("/routing/coding")).toEqual({
      to: "/routing",
      label: "Routing",
    })
    expect(pageLabelForPath("/tools/guardrails")).toBe("Guardrails")
    expect(pageLabelForPath("/members")).toBe("Members")
  })
})

describe("destinations with no row", () => {
  it("selects Rules on the Routing pages and the workspace's guardrails", () => {
    for (const path of ["/routing", "/routing/coding", "/tools/guardrails"]) {
      expect(railPathForPath(path)).toBe("/rules")
      expect(railLabelForPath(path)).toBe("Rules")
      expect(navContextForPath(path)).toBe("workspace")
    }
    // Gated as their own entries, not as Rules.
    expect(navItemForPath("/routing")?.surface).toBe("routing")
    expect(navItemForPath("/routing/coding")?.surface).toBe("routing")
    expect(navItemForPath("/tools/guardrails")?.surface).toBe("tools")
  })

  it("selects People on the workspace roster", () => {
    expect(navContextForPath("/members")).toBe("workspace")
    expect(navItemForPath("/members")?.surface).toBe("workspaces")
    expect(railPathForPath("/members")).toBe("/people")
    expect(railLabelForPath("/members")).toBe("People")
  })

  it("draws none of them as a row", () => {
    const rows = [
      ...NAV_SECTIONS,
      ...ORG_NAV_SECTIONS,
      ...DEPLOYMENT_NAV_SECTIONS,
    ].flatMap((section) => [
      ...section.items.map((item) => item.to),
      ...section.items.flatMap((item) =>
        (item.children ?? []).map((child) => child.to),
      ),
    ])
    for (const path of ["/routing", "/tools/guardrails", "/members"]) {
      expect(rows).not.toContain(path)
    }
  })

  it("leaves a page with a row of its own to select that row, naming no other", () => {
    expect(railPathForPath("/keys")).toBe("/keys")
    expect(railPathForPath("/tools/web-search")).toBe("/tools")
    expect(railPathForPath("/people/some-person")).toBe("/people")
    expect(railPathForPath("/rules/some-rule")).toBe("/rules")
    expect(railLabelForPath("/rules")).toBeUndefined()
    expect(railLabelForPath("/tools/web-search")).toBeUndefined()
    expect(railPathForPath("/account")).toBeUndefined()
  })
})

const PLATFORM = "ws-platform"

/** A caller of `role` in the Platform workspace, as `routeAccess` reads one. */
function callerWith(
  role: "owner" | "admin" | "member" | "viewer",
  isOperator = false,
): Caller {
  return {
    context: organizationContext({
      role,
      deployment_operator: isOperator,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role },
      ],
    }),
    answer: isOperator ? "operator" : "not-operator",
    selectedWorkspaceRole: role,
  }
}

const ROLES = {
  owner: callerWith("owner"),
  admin: callerWith("admin"),
  member: callerWith("member"),
  viewer: callerWith("viewer"),
  operator: callerWith("owner", true),
} as const

/**
 * Every destination the rails offer `caller` by a click: rows, views, a
 * group's rows, and a destination with no row once the row that stands for it
 * is offered. On a deployment that hosts every surface, so only the caller
 * axis varies, read as the rail reads it.
 */
function reach(caller: Caller): string[] {
  const opens = (item: NavItem) =>
    routeAccess({ route: item.to, params: {} }, caller) === "open"
  const offered = [
    ...NAV_SECTIONS,
    ...ORG_NAV_SECTIONS,
    ...DEPLOYMENT_NAV_SECTIONS,
  ].flatMap((section) =>
    visibleNavSections([section], opens).flatMap(({ items }) =>
      items.flatMap((item) => [
        item.to,
        ...(item.views ?? []),
        ...(item.children ?? [])
          .filter((child) => opens(navItemForPath(child.to) ?? item))
          .map((child) => child.to),
      ]),
    ),
  )
  const rows = new Set<string>(offered)
  const withoutRow = NAV_ITEMS.filter(
    (item) =>
      item !== PLAYGROUND_NAV_ITEM &&
      railPathForPath(item.to) !== item.to &&
      rows.has(railPathForPath(item.to) ?? "") &&
      opens(item),
  ).map((item) => item.to)
  return [...new Set([...offered, ...withoutRow])]
}

const WORKSPACE = [
  "/",
  "/sessions",
  "/requests",
  "/people",
  "/rules",
  "/models",
  "/tools",
  "/keys",
  "/members",
  "/routing",
]
const ORGANIZATION = [
  "/workspaces",
  "/organization/members",
  "/organization/domains",
  "/organization/sso",
  "/budgets",
  "/organization/provider-keys",
  "/organization/web-search-keys",
  "/organization/guardrails",
  "/organization",
]
const DEPLOYMENT = [
  "/settings",
  "/admin/accounts",
  "/admin/hosted-providers",
  "/admin/recommender",
  "/admin/web-search",
  "/admin/code-execution",
  "/health-checks",
]

describe("the rails, per role", () => {
  for (const [role, expected] of [
    ["owner", [...WORKSPACE, "/tools/guardrails", ...ORGANIZATION]],
    ["admin", [...WORKSPACE, "/tools/guardrails", ...ORGANIZATION]],
    ["member", WORKSPACE],
    ["viewer", WORKSPACE],
    [
      "operator",
      [...WORKSPACE, "/tools/guardrails", ...ORGANIZATION, ...DEPLOYMENT],
    ],
  ] as const) {
    it(`offers the ${role} the pages that open for them, and no others`, () => {
      expect([...reach(ROLES[role])].sort()).toEqual([...expected].sort())
    })
  }

  it("keeps each page with no row within reach of a row the same role is offered", () => {
    // A tool's page opens from Tools, a routing page and the workspace's
    // guardrails from Rules, and the roster from People. A role that may open
    // the page has to be offered the row it opens from, or the page is only
    // reachable by typing its address.
    const OPENED_FROM: Readonly<Record<string, string>> = {
      "/tools/web-search": "/tools",
      "/tools/code-execution": "/tools",
      "/tools/mcp-servers": "/tools",
      "/routing": "/rules",
      "/tools/guardrails": "/rules",
      "/members": "/people",
    }
    for (const caller of Object.values(ROLES)) {
      const offered = reach(caller)
      for (const [page, row] of Object.entries(OPENED_FROM)) {
        const target = { route: page, params: {} }
        if (routeAccess(target, caller) === "open") {
          expect(offered, page).toContain(row)
        }
      }
    }
  })

  it("leaves no registered destination without a way to it", () => {
    // Every entry is a row, a view or a group's row on some rail, a page that
    // selects a row, or the Playground the chrome links.
    const linked = new Set(reach(ROLES.operator))
    for (const item of NAV_ITEMS) {
      if (item === PLAYGROUND_NAV_ITEM) continue
      expect(linked, item.to).toContain(item.to)
    }
  })
})

describe("reachableWithoutOrganization", () => {
  it("is declared by Home alone in the base registry", () => {
    const optedIn = [
      ...NAV_SECTIONS,
      ...ORG_NAV_SECTIONS,
      ...DEPLOYMENT_NAV_SECTIONS,
    ]
      .flatMap((section) => section.items)
      .filter((item) => item.reachableWithoutOrganization)
      .map((item) => item.to)
    expect(optedIn).toEqual(["/"])
  })
})
