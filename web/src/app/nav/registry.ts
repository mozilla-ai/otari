import {
  FiActivity,
  FiAtSign,
  FiBox,
  FiCode,
  FiCpu,
  FiDollarSign,
  FiGlobe,
  FiGrid,
  FiHome,
  FiKey,
  FiLayers,
  FiList,
  FiLogIn,
  FiMessageSquare,
  FiRepeat,
  FiShield,
  FiShuffle,
  FiSliders,
  FiTool,
  FiUserCheck,
  FiUsers,
} from "react-icons/fi"
import { OVERLAY_NAV_LABEL_OVERRIDES } from "@/app/nav/overlayLabelOverrides"
import { OVERLAY_NAV_ITEMS } from "@/app/nav/overlayNavItems"
import {
  OVERLAY_DEPLOYMENT_NAV_SECTIONS,
  OVERLAY_NAV_SECTIONS,
  OVERLAY_ORG_NAV_SECTIONS,
} from "@/app/nav/overlaySections"
import type {
  NavItem,
  NavItemContribution,
  NavLabelOverride,
  NavPath,
  NavSection,
} from "./types"

/**
 * The workspace rail, and the only place its destinations are declared.
 *
 * One section, headingless: every row is a place in the selected workspace,
 * and none of them is a category. Three destinations are registered without a
 * row (`OFF_RAIL_ITEMS`): Routing and the workspace's Guardrails, which open
 * from Rules, and `/members`, the workspace roster, which opens from People.
 *
 * Each thing exists once: Billing, Providers and Email domains are
 * organization pages, reached from the footer's Organization row.
 *
 * Each entry declares its own gating, and the two axes are independent:
 * `surface` (does this deployment host it) and `capability` (is it entitled).
 * The sidebar composes them as AND. Hiding a link is a convenience and never an
 * authorization; the server still authorizes every request the page behind it
 * makes.
 */
const BASE_NAV_SECTIONS = [
  {
    id: "index",
    items: [
      // Ungated on every axis: the index is the deployment's own front page and
      // reads whatever it is allowed to. The one row a caller with no
      // organization keeps, since it is where they are told how to get in.
      {
        to: "/",
        label: "Home",
        icon: FiHome,
        reachableWithoutOrganization: true,
      },
      {
        to: "/sessions",
        label: "Sessions",
        icon: FiList,
        views: ["/requests"],
      },
      { to: "/people", label: "People", icon: FiUsers },
      { to: "/rules", label: "Rules", icon: FiShield },
      // No `operatorOnly` on Models, per the rule the axis carries: a row
      // leaves the list by its page ceasing to refuse anyone (otari-ai#1942).
      // Models reads the catalog any session may read.
      {
        to: "/models",
        label: "Models",
        surface: "models",
        icon: FiLayers,
      },
      {
        to: "/tools",
        label: "Tools",
        surface: "tools",
        icon: FiTool,
        // One row, one page. `/tools` lists every tool with what it does and
        // whether it is on; each one's settings open from it, and the row
        // stays selected there (the prefix match below).
      },
      // Create API key above the rail starts one; this lists and edits them.
      { to: "/keys", label: "API keys", surface: "keys", icon: FiKey },
    ],
  },
] as const satisfies readonly NavSection[]

/**
 * The organization context: what belongs to the tenant rather than to one
 * workspace inside it.
 *
 * Reached from the sidebar footer and left by the "Back to" link at its top,
 * so the two contexts never render together. Gated on the caller managing the
 * organization, which today is always true: there is one session, the local
 * operator, and it owns the organization the gateway provisioned for itself.
 * The gate is written anyway because it is the thing that becomes load-bearing the moment per-user sign-in lands (otari-ai#1716).
 *
 * Every row is gated on a surface the bootstrap publishes (`SURFACES` in
 * `src/gateway/api/routes/bootstrap.py`), and a group whose every row a
 * deployment gates off drops entirely, heading included.
 *
 * The organization's guardrails are in General, under Providers.
 *
 * The design draws one more, Billing, and it is not declared here at all,
 * because it is not this build's to declare: Billing is ARCHITECTURE.md's
 * canonical overlay-only capability, so an overlay owns the row together with
 * the page under it and contributes it through `overlayNavItems.ts`. Declaring
 * it here would be worse than redundant: a base route file at a path the overlay also serves is not shadowed by the
 * overlay's, it collides with it. The generator refuses the pair ("Conflicting
 * configuration paths were found") instead of choosing between them, so the
 * composed tree is never written: the Vite plugin logs that error and keeps
 * building against the tree already on disk, where the overlay's page has no
 * route at all, and a step that generates the tree on its own fails outright.
 *
 * The surface axis rather than the capability one, and that is a constraint
 * rather than a preference: `registry.test.ts` requires every capability a base
 * entry names to be in `BASE_CAPABILITIES`, that is, to be granted, so a
 * capability gate cannot express "declared but not served" without relaxing that
 * invariant. A surface gate says exactly this and needs no test change.
 */
const ORGANIZATION_NAV_SECTIONS = [
  {
    id: "org-people",
    label: "People & access",
    items: [
      // Absent from the design, which switches workspace from the scope menu and
      // has no list page. Kept because this is the only place a workspace is
      // renamed, deleted, or has its roster read, and the scope menu offers none
      // of that.
      {
        to: "/workspaces",
        label: "Workspaces",
        surface: "workspaces",
        icon: FiGrid,
      },
      {
        to: "/organization/members",
        label: "Members & roles",
        surface: "organizations",
        icon: FiUsers,
      },
      // Beside the roster because it is the other way somebody becomes a
      // member: added by hand there, or by proving a domain here.
      {
        to: "/organization/domains",
        label: "Email domains",
        surface: "organizations",
        icon: FiAtSign,
      },
      // Below the domains it admits through, since BYO OIDC reuses their
      // DNS proofs rather than carrying its own.
      {
        to: "/organization/sso",
        label: "Single sign-on",
        surface: "organizations",
        icon: FiLogIn,
      },
    ],
  },
  {
    id: "org-money",
    label: "Cost & billing",
    items: [
      // No `operatorOnly`, because the destination is two pages now: an
      // operator gets the deployment's budgets and an organization owner or
      // admin gets their own organization's (otari-ai#1943). The roles matrix
      // has this row at Edit for an admin, and `/organizations/me/budgets`
      // plus `/organizations/me/spend-ceilings` are what it edits. A plain
      // member is not offered it, because the organization rail opens only to a
      // caller who manages the organization.
      {
        to: "/budgets",
        label: "Spend & budgets",
        surface: "budgets",
        icon: FiDollarSign,
        // The operator opens it for the deployment's budgets whatever their
        // role in the organization (`BudgetsPage`).
        requires: "manager-or-operator",
      },
    ],
  },
  {
    id: "org-general",
    label: "General",
    items: [
      // The organization's own provider keys: Inference is where its models
      // are offered, priced and switched, and Web search holds its search and
      // fetch keys. The deployment's own keys are Hosted providers and Web
      // search, on the deployment rail.
      {
        to: "/organization/provider-keys",
        label: "Providers",
        surface: "organization_providers",
        icon: FiBox,
        children: [
          {
            to: "/organization/provider-keys",
            label: "Inference",
            icon: FiBox,
          },
          {
            to: "/organization/web-search-keys",
            label: "Web search",
            icon: FiGlobe,
          },
        ],
      },
      // The organization's own guardrails and where each one runs. Not the
      // workspace rail's `/tools/guardrails`, which sets the deployment's own
      // guardrails service.
      {
        to: "/organization/guardrails",
        label: "Guardrails",
        surface: "organization_guardrails",
        icon: FiShield,
      },
      {
        to: "/organization",
        label: "Org settings",
        surface: "organizations",
        icon: FiSliders,
      },
    ],
  },
] as const satisfies readonly NavSection[]

/**
 * The deployment's own rail: what the gateway process is, rather than what any
 * tenant on it has.
 *
 * A third context rather than a section in either rail, because that is the
 * distinction these two pages were always making and neither other rail could
 * hold. `/settings` is the running process (its master key, its safety toggles,
 * its defaults) and `/admin/accounts` reaches an account in any organization on
 * it, or in none that still admits it, which is a wider scope than the
 * organization rail's own roster.
 *
 * Headingless on purpose. One section with no siblings needs nothing to
 * separate it from, and a heading inside a context already named Deployment
 * would read as "Deployment, Deployment, Settings".
 *
 * Every row keeps the refusal it declares, which is the axis the server
 * enforces and not a property of which rail draws them.
 */
const DEPLOYMENT_SECTIONS = [
  {
    id: "deployment-general",
    items: [
      {
        to: "/settings",
        label: "Settings",
        surface: "settings",
        icon: FiSliders,
        operatorOnly: "refused",
      },
      {
        to: "/admin/accounts",
        label: "Accounts",
        surface: "admin",
        operatorOnly: "unlisted",
        icon: FiUserCheck,
      },
      // The keys the deployment serves inference on for a request that brings
      // none of its own. Unlisted like Accounts, because its API refuses a
      // non-operator with the same 404.
      {
        to: "/admin/hosted-providers",
        label: "Hosted providers",
        surface: "hosted_providers",
        operatorOnly: "unlisted",
        icon: FiCpu,
      },
      // The decision models that pick a subagent's model, and what a
      // recommendation costs. Unlisted like Hosted providers: its API refuses
      // a non-operator with the same 404.
      {
        to: "/admin/recommender",
        label: "Subagent router",
        surface: "agent_recommenders",
        operatorOnly: "unlisted",
        icon: FiShuffle,
      },
      // The search and fetch tools the deployment runs for every workspace, and
      // their prices.
      {
        to: "/admin/web-search",
        label: "Web search",
        surface: "tools",
        operatorOnly: "refused",
        icon: FiGlobe,
      },
      {
        to: "/admin/code-execution",
        label: "Code execution pricing",
        surface: "pricing",
        operatorOnly: "refused",
        icon: FiCode,
      },
      {
        to: "/health-checks",
        label: "Health",
        surface: "health_checks",
        icon: FiActivity,
        operatorOnly: "refused",
      },
    ],
  },
] as const satisfies readonly NavSection[]

/**
 * Rename base section headings and disclosure labels, matched by `sectionId`.
 *
 * Those two fields only: everything else about a section, its items, and their
 * icons and gating passes through untouched, and a section no override targets
 * is returned as it was. An override naming a section, or a path inside one,
 * that the registry does not declare is a no-op rather than a throw, so a stale
 * override in an overlay costs a rename and not the sidebar. Two overrides
 * naming one section is the overlay's own bug, and the last of them wins.
 *
 * Applied to the base sections before `composeNavSections` appends an overlay's
 * own, which declare their labels directly and have nothing to rename.
 */
export function applyNavLabelOverrides(
  sections: readonly NavSection[],
  overrides: readonly NavLabelOverride[],
): readonly NavSection[] {
  if (overrides.length === 0) return sections
  const overrideBySectionId = new Map(
    overrides.map((override) => [override.sectionId, override]),
  )
  return sections.map((section) => {
    const override = overrideBySectionId.get(section.id)
    if (!override) return section
    const { disclosureLabels } = override
    return {
      ...section,
      ...(override.label !== undefined && { label: override.label }),
      ...(disclosureLabels && {
        items: section.items.map((item) => {
          const label = disclosureLabels[item.to]
          // Groups only, which is what the platform's single `NavDisclosure`
          // label maps to. A group is a destination here as well as a heading,
          // so renaming one also renames what the breadcrumbs and the shell's
          // gated-off panel call that page; the group and the page sharing a
          // name is why that reads correctly rather than as a bug.
          // A plain link's label and a `NavChild`'s are outside the seam, as
          // they are over there; widen it when a contribution needs one.
          return label !== undefined && item.children
            ? { ...item, label }
            : item
        }),
      }),
    }
  })
}

/**
 * Append an overlay's contributed rows to the base sections they name.
 *
 * The finer of the two composition seams: `composeNavSections` adds a section,
 * this adds rows inside one the base declares, which is what a destination like
 * Billing needs (it belongs under "Cost & billing", among base rows, and a
 * section of its own would put a second heading of that name on the rail).
 *
 * Appended after the section's base rows, in the order the contribution lists
 * them: an overlay orders its own rows and does not interleave them with the
 * base's, the same rule `composeNavSections` follows one grain coarser. A
 * contribution naming a section this registry does not declare is dropped, as a
 * stale `NavLabelOverride` is, so a section renamed or removed here costs an
 * overlay the rows it put there rather than the whole sidebar. Two contributions
 * naming one section both land, in list order.
 */
export function composeNavItems(
  sections: readonly NavSection[],
  contributions: readonly NavItemContribution[],
): readonly NavSection[] {
  if (contributions.length === 0) return sections
  return sections.map((section) => {
    const items = contributions
      .filter((contribution) => contribution.sectionId === section.id)
      .flatMap((contribution) => contribution.items)
    return items.length === 0
      ? section
      : { ...section, items: [...section.items, ...items] }
  })
}

/**
 * Compose the base sections with an overlay build's contributions.
 *
 * Base first, then overlay, so an overlay appends its own sections without
 * reordering the base sidebar.
 */
export function composeNavSections(
  base: readonly NavSection[],
  overlay: readonly NavSection[],
): readonly NavSection[] {
  return [...base, ...overlay]
}

/**
 * The composed workspace sidebar.
 *
 * Three seams, applied in that order: rename what the base declares, add rows
 * inside a base section, then append the overlay's own sections. Renaming runs
 * first so a `disclosureLabels` entry addresses base groups only, which is what
 * it is for; a contributed row carries the label the overlay gave it.
 *
 * This build renames nothing and appends nothing, so it is the base sections
 * alone.
 */
export const NAV_SECTIONS: readonly NavSection[] = composeNavSections(
  composeNavItems(
    applyNavLabelOverrides(BASE_NAV_SECTIONS, OVERLAY_NAV_LABEL_OVERRIDES),
    OVERLAY_NAV_ITEMS,
  ),
  OVERLAY_NAV_SECTIONS,
)

/**
 * The composed organization sidebar.
 *
 * Composed the same way the workspace rail is, and for the same reason: Billing
 * is the canonical overlay-only capability (ARCHITECTURE.md's capability table)
 * and it belongs on this rail, so an overlay that could only contribute to the
 * workspace one would have to edit this file to register it, which is what
 * cardinal rule 6 rules out. Both rails need the row seam as well as the section
 * one, and this rail is why: Billing lands inside "Cost & billing", among rows
 * the base declares, so appending a section could only put a second heading of
 * the same name below the first. This build appends
 * nothing.
 *
 * Label overrides and row contributions come from the same lists the workspace
 * rail reads: a section id is unique across the two rails (`registry.test.ts`
 * pins that), so one list addresses both and an overlay has one module to
 * replace rather than two.
 */
export const ORG_NAV_SECTIONS: readonly NavSection[] = composeNavSections(
  composeNavItems(
    applyNavLabelOverrides(
      ORGANIZATION_NAV_SECTIONS,
      OVERLAY_NAV_LABEL_OVERRIDES,
    ),
    OVERLAY_NAV_ITEMS,
  ),
  OVERLAY_ORG_NAV_SECTIONS,
)

/**
 * The composed deployment sidebar.
 *
 * Composed through the same three seams as the other two rails, because the
 * distinction this rail draws is one an overlay can land on: otari.ai
 * administers the process every tenant is served by, and those destinations
 * belong here rather than beside a tenant's own settings. Reached through the
 * row seam, since the rail declares one headingless section and a contributed
 * section would put a heading above rows that have none.
 *
 * Label overrides and row contributions come from the lists both other rails
 * read. A section id is unique across all three (`registry.test.ts` pins that),
 * so one list still addresses every rail and an overlay has one module to
 * replace rather than three. This build appends nothing.
 */
export const DEPLOYMENT_NAV_SECTIONS: readonly NavSection[] =
  composeNavSections(
    composeNavItems(
      applyNavLabelOverrides(DEPLOYMENT_SECTIONS, OVERLAY_NAV_LABEL_OVERRIDES),
      OVERLAY_NAV_ITEMS,
    ),
    OVERLAY_DEPLOYMENT_NAV_SECTIONS,
  )

// Kept in route lookup for surface gating, outside the sidebar sections.
// Availability follows the gateway's inference surface, not a paid capability,
// and a viewer, who may not run anything, has no use for it.
export const PLAYGROUND_NAV_ITEM = {
  to: "/playground",
  label: "Playground",
  surface: "playground",
  icon: FiMessageSquare,
  requires: "contributor",
} as const satisfies NavItem

/**
 * The workspace rail as the mobile drawer draws it: the same rows as the desk.
 * The account menu offers the Playground on a phone as everywhere else.
 */
export const DRAWER_NAV_SECTIONS: readonly NavSection[] = NAV_SECTIONS

/**
 * The Routing pages, registered without a row. Routing is a tab of Rules
 * (`features/rules/RulesFrame.tsx`), and the workspace's guardrails open from
 * its Data tab, so both select Rules in the rail (`railPathForPath`). Not
 * tagged `capability: "routing"`: ARCHITECTURE.md marks the routing split
 * provisional. `/aliases` redirects onto `/routing`.
 */
const ROUTING_NAV_ITEM = {
  to: "/routing",
  label: "Routing",
  surface: "routing",
  icon: FiRepeat,
} as const satisfies NavItem

// An owner or admin sets the workspace's guardrails; the page has nothing for
// anyone else to do.
const WORKSPACE_GUARDRAILS_NAV_ITEM = {
  to: "/tools/guardrails",
  label: "Guardrails",
  surface: "tools",
  icon: FiShield,
  requires: "manager",
} as const satisfies NavItem

/**
 * The workspace roster, registered without a row. People lists the workspace
 * and links here for what only the roster does (add someone to the workspace,
 * change or drop their workspace role), so People stays selected on it.
 */
const WORKSPACE_MEMBERS_NAV_ITEM = {
  to: "/members",
  label: "Members",
  surface: "workspaces",
  icon: FiUsers,
} as const satisfies NavItem

/** Destinations with no rail row, by the rail row that stands for them. */
const OFF_RAIL_ITEMS: readonly { item: NavItem; railPath?: NavPath }[] = [
  { item: WORKSPACE_MEMBERS_NAV_ITEM, railPath: "/people" },
  { item: ROUTING_NAV_ITEM, railPath: "/rules" },
  { item: WORKSPACE_GUARDRAILS_NAV_ITEM, railPath: "/rules" },
]

const RAIL_ITEMS: readonly NavItem[] = [
  ...NAV_SECTIONS,
  ...ORG_NAV_SECTIONS,
  ...DEPLOYMENT_NAV_SECTIONS,
].flatMap((section) => section.items)

/**
 * Route lookup includes chrome destinations as well as all three sidebar scopes.
 * And the destinations with no row (`OFF_RAIL_ITEMS`).
 */
export const NAV_ITEMS: readonly NavItem[] = [
  PLAYGROUND_NAV_ITEM,
  ...OFF_RAIL_ITEMS.map(({ item }) => item),
  ...RAIL_ITEMS,
]

/**
 * The rail row to mark selected at this pathname: for a destination with no
 * row, the row that stands for it (Rules for Routing), and otherwise the
 * destination's own entry.
 */
export function railPathForPath(pathname: string): NavPath | undefined {
  const item = navItemForPath(pathname)
  if (!item) return undefined
  const offRail = OFF_RAIL_ITEMS.find((one) => one.item.to === item.to)
  return offRail ? offRail.railPath : item.to
}

/**
 * The rail row a destination with no row of its own sits under, by name, for
 * the breadcrumb ("Rules" for Routing), or nothing.
 */
export function railLabelForPath(pathname: string): string | undefined {
  const item = navItemForPath(pathname)
  const railPath = OFF_RAIL_ITEMS.find(
    (one) => one.item.to === item?.to,
  )?.railPath
  return railPath ? navItemForPath(railPath)?.label : undefined
}

/**
 * Every nested destination, paired with the entry it is gated by.
 *
 * A child has no gating of its own, so `navItemForPath` has to answer with the
 * parent: that is what the shell reads to decide whether the route is served
 * and what to call it when it is not.
 */
const NAV_CHILD_PARENTS: ReadonlyMap<string, NavItem> = new Map(
  NAV_ITEMS.flatMap((item) =>
    (item.children ?? []).map(
      (child) =>
        [
          child.to,
          // The parent, so the sidebar still highlights the group this belongs
          // to, but carrying the child's own surface when it declares one.
          child.surface ? { ...item, surface: child.surface } : item,
        ] as const,
    ),
  ),
)

/**
 * Which sidebar a destination lives under.
 *
 * Three rather than two since the deployment's own pages stopped being a
 * section inside the organization's: they describe the process every tenant
 * shares, so neither tenant-shaped rail could hold them truthfully.
 */
export type NavContext = "workspace" | "organization" | "deployment"

const ORG_PATHS: readonly string[] = ORG_NAV_SECTIONS.flatMap((section) =>
  section.items.map((item) => item.to),
)

const DEPLOYMENT_PATHS: readonly string[] = DEPLOYMENT_NAV_SECTIONS.flatMap(
  (section) => section.items.map((item) => item.to),
)

/**
 * What to call the destination at this pathname, as a breadcrumb would.
 *
 * Distinct from `navItemForPath`, which answers with the entry that *gates* a
 * path: for a nested destination that is the parent, so it would name
 * `/tools/web-search` "Tools". A breadcrumb wants the leaf.
 */
export function navLabelForPath(pathname: string): string | undefined {
  return navPageForPath(pathname)?.label
}

/**
 * The rail destination a pathname is, or sits under, as `{ to, label }`.
 *
 * A child wins over its parent for its own subpages too, so a server's page at
 * `/tools/mcp-servers/<id>` is under "MCP servers" rather than "Tools".
 */
export function navPageForPath(
  pathname: string,
): { to: string; label: string } | undefined {
  const child = NAV_ITEMS.flatMap((item) => item.children ?? []).find(
    (one) => one.to === pathname || pathname.startsWith(`${one.to}/`),
  )
  const entry = child ?? navItemForPath(pathname)
  return entry ? { to: entry.to, label: entry.label } : undefined
}

/** Names for the chrome destinations, which no registry entry declares. */
const CHROME_PAGE_LABELS: Readonly<Record<string, string | undefined>> = {
  "/account": "Account",
  "/invitations": "Invitations",
}

/**
 * What to call the page at this pathname, rail or chrome, for the tab title
 * and the breadcrumb alike, so the two cannot name one page differently.
 */
export function pageLabelForPath(pathname: string): string | undefined {
  return navLabelForPath(pathname) ?? CHROME_PAGE_LABELS[pathname]
}

/**
 * Which sidebar a pathname belongs under.
 *
 * Derived from the registry rather than from a path prefix, because the three
 * contexts do not split cleanly by URL: `/workspaces` is an organization
 * destination and `/settings` a deployment one, both of which look like
 * anything else, and `/members` is a workspace one that sits directly under the
 * root. Anything
 * unregistered (the guide, the 404 splat) belongs to the workspace context,
 * which is the one the shell opens in.
 */
export function navContextForPath(pathname: string): NavContext {
  const item = navItemForPath(pathname)
  if (!item) return "workspace"
  if (DEPLOYMENT_PATHS.includes(item.to)) return "deployment"
  return ORG_PATHS.includes(item.to) ? "organization" : "workspace"
}

/**
 * The registry entry a pathname belongs to, if any.
 *
 * What lets the shell answer a gated-off destination with "not available here"
 * rather than rendering a page whose every request the server will refuse: the
 * link is gone from the sidebar, but a bookmark, a shared URL, or a gateway
 * restarted into another mode can still land on the route.
 *
 * A path matches its entry exactly or as a prefix, so a future child route
 * (`/routing/new`) inherits its parent's gating. The index is matched exactly
 * only; as a prefix it would claim every path in the dashboard.
 *
 * **An exact match wins over a prefix match**, and the two passes are why: with
 * a single scan, `/organization/members` would resolve to `/organization`,
 * which is registered ahead of it. The gating would still be right (both name
 * the same surface), but everything else that asks "which entry is this page"
 * would be wrong: the shell titles its gated-off panel from the entry it gets
 * back, and the sidebar highlights it.
 *
 * An unregistered path (`/docs`, the 404 splat) has no entry and is therefore
 * never gated, which is right: the registry governs the destinations it
 * declares and nothing else.
 */
export function navItemForPath(pathname: string): NavItem | undefined {
  const exact = NAV_ITEMS.find((item) => pathname === item.to)
  if (exact) return exact
  // A nested destination answers with the entry that gates it, so a child route
  // is never treated as unregistered (and therefore ungated).
  const child = NAV_CHILD_PARENTS.get(pathname)
  if (child) return child
  // Another view of one destination (the Overview's tabs) is that destination.
  const view = NAV_ITEMS.find((item) =>
    item.views?.some((path) => path === pathname),
  )
  if (view) return view
  // Longest prefix, not first: `/organization/members/x` is under both
  // `/organization` and `/organization/members`, and the deeper entry is the
  // one that describes it. Ordering the scan rather than the registry, because
  // the registry's order is the sidebar's.
  return NAV_ITEMS.filter(
    (item) => item.to !== "/" && pathname.startsWith(`${item.to}/`),
  ).sort((a, b) => b.to.length - a.to.length)[0]
}

/**
 * Whether this deployment serves the destination at this pathname.
 *
 * One predicate for every caller that asks that question, which is the point:
 * the shell asks it to decide between a page and the "not available here" panel,
 * and the rail memory asks it to decide whether a stored destination is still
 * somewhere to send you. Two implementations of the same rule drift, and they
 * drift in the direction where the memory resumes onto the panel.
 *
 * Nesting is where the rule has an edge: `navItemForPath` answers a child with
 * its parent carrying the child's `surface`, so a child is gated on *its own*
 * surface rather than on the group's, plus the parent's capability. A
 * path the registry does not declare (the guide, the 404 splat) is not gated at
 * all, and this says so with `true`.
 */
export function isPathVisible(
  pathname: string,
  isVisible: (item: NavItem) => boolean,
): boolean {
  const item = navItemForPath(pathname)
  return item === undefined || isVisible(item)
}

/** A section that has at least one visible entry, paired with those entries. */
export interface VisibleNavSection {
  section: NavSection
  items: readonly NavItem[]
}

/**
 * The sections worth rendering, with the entries worth rendering in them.
 *
 * Filters before the caller indexes, which is the point: the sidebar draws its
 * divider and top margin above every section *after* the first, so keying that
 * off the registry index would leave a stray top border above the first visible
 * group once a section ahead of it empties out. Not reachable in this build,
 * where the index section is ungated and so always renders first, and reachable
 * as soon as an overlay contributes a section or a gated one empties, which is
 * the whole point of the seam. `otari-ai/frontend`'s sidebar keys off rendered
 * position for the same reason.
 *
 * A section with no visible entry is dropped whole, heading included: an empty
 * heading over nothing reads worse than no heading.
 */
export function visibleNavSections(
  sections: readonly NavSection[],
  isVisible: (item: NavItem) => boolean,
): VisibleNavSection[] {
  return sections
    .map((section) => ({ section, items: section.items.filter(isVisible) }))
    .filter(({ items }) => items.length > 0)
}
