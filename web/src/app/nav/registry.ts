import {
  FiActivity,
  FiAtSign,
  FiBarChart2,
  FiBox,
  FiCode,
  FiDollarSign,
  FiGlobe,
  FiGrid,
  FiHome,
  FiKey,
  FiLayers,
  FiMessageSquare,
  FiRepeat,
  FiServer,
  FiShield,
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
  NavSection,
} from "./types"

/**
 * The sidebar the base build ships, and the only place a destination is
 * declared.
 *
 * Sections render in this order. The index leads, alone and unlabeled, because
 * it is the rail's destination rather than a member of a category: it is where
 * the sidebar puts you back, not one of the places you go from it. The three
 * below it are the design's: "Observe" is where you look (the request log and
 * the usage rollups over it), "Build" is what the gateway serves (models, the
 * policies that route over them, and the tools it can call; the roles matrix's
 * name for the section, otari-ai#1942), and "Access" is who may call it (keys
 * and the workspace's roster).
 *
 * Each entry declares its own gating, and the two axes are independent:
 * `surface` (does this deployment host it) and `capability` (is it entitled).
 * The sidebar composes them as AND. Hiding a link is a convenience and never an
 * authorization; the server still authorizes every request the page behind it
 * makes.
 */
const BASE_NAV_SECTIONS = [
  {
    // Headingless on purpose, and the one section in this rail that is. A label
    // over a single row would read as a category with one member, and the index
    // is not a category: it is the row every other row is a departure from. The
    // gap the shell already puts between sections is what separates it.
    id: "index",
    items: [
      // Ungated on every axis: the index is the deployment's own front page and
      // reads whatever it is allowed to.
      { to: "/", label: "Overview", icon: FiHome },
    ],
  },
  {
    id: "observe",
    label: "Observe",
    items: [
      // Both read usage, which is why they name the surface rather than
      // themselves: a deployment that does not host usage loses both.
      //
      // Neither declares `operatorOnly` any more, and that is the point of
      // otari#837 rather than an oversight. They used to, because everything
      // behind them was `/usage` and a member was refused all of it; now the
      // pages read `/organizations/me/usage` for a caller who does not
      // operate the deployment, so both destinations serve every signed-in
      // identity something true. Tagging them would hide a page that works.
      {
        to: "/activity",
        label: "Activity",
        surface: "usage",
        icon: FiActivity,
      },
      {
        to: "/usage",
        label: "Usage",
        surface: "usage",
        icon: FiBarChart2,
      },
    ],
  },
  {
    // The id stays "gateway" while the label reads "Build": overlay
    // contributions and label overrides key on the id, so renaming it would
    // silently drop whatever a superset build files under this section.
    id: "gateway",
    label: "Build",
    items: [
      // No `operatorOnly` on Models or Routing any more, per the rule the axis
      // carries: a row leaves the list by its page ceasing to refuse anyone
      // (otari-ai#1942). Models reads the catalog any session may read, and
      // Routing reads the tenant-scoped policy list for a non-operator.
      {
        to: "/models",
        label: "Models",
        surface: "models",
        icon: FiLayers,
      },
      // Deliberately not tagged `capability: "routing"`, though otari.ai's
      // registry tags its own Routing item that way. ARCHITECTURE.md's
      // capability lines mark the routing split (how much is core base, how
      // much an overlay adapter) **provisional**, and say it is a decision for
      // the maintainers rather than something a contributor assumes. Tagging
      // this entry would assume it, and would withhold nothing today either
      // way, since the base grants what it ships. Add the tag when the split is
      // decided, together with its name in `BASE_CAPABILITIES`.
      {
        to: "/routing",
        label: "Routing",
        surface: "routing",
        icon: FiRepeat,
        // Policies and Guardrails as the navigation prototype groups them. The
        // prototype's third entry, Aliases, is deliberately absent: this
        // dashboard lists an alias as the one-target policy it is, in the same
        // table (see `RoutingPage`), so `/aliases` is a compatibility redirect
        // onto `/routing` rather than a destination. Linking it would give the
        // group two entries for one page, and the second could never highlight.
        // It comes back if and when Routing grows a separate alias view.
        children: [
          { to: "/routing", label: "Policies", icon: FiRepeat },
          {
            to: "/tools/guardrails",
            label: "Guardrails",
            icon: FiShield,
            // Grouped with Routing, served by the tools surface.
            surface: "tools",
          },
        ],
      },
      {
        to: "/tools",
        label: "Tools",
        surface: "tools",
        icon: FiTool,
        // Two of the three services the page configures, plus MCP servers;
        // Guardrails is grouped under Routing, where the prototype puts it.
        // MCP is the row that is not a tool-service view, so it has a page of
        // its own rather than a filter over the settings page; web/AGENTS.md
        // says why.
        //
        // `to` gates and names the group; it is not somewhere the rail
        // navigates. A group with more than one visible child is a disclosure
        // when expanded and a flyout when collapsed, and neither offers the
        // parent, so `/tools` (the three services on one page) is reachable by
        // URL only. Deliberate: each service has its own destination here, and
        // Guardrails belongs to Routing, so a row for the combined page would
        // duplicate all three and cross the grouping the design draws. Routing
        // leads with Policies because `/routing` *is* the policies page, not to
        // make a parent reachable.
        children: [
          { to: "/tools/web-search", label: "Web search", icon: FiGlobe },
          {
            to: "/tools/code-execution",
            label: "Code execution",
            icon: FiCode,
          },
          { to: "/tools/mcp-servers", label: "MCP servers", icon: FiServer },
        ],
      },
    ],
  },
  {
    id: "access",
    label: "Access",
    items: [
      // No `operatorOnly`, for the reason Activity and Usage dropped theirs
      // (otari-ai#1941 this time): the page reads and mints through
      // `/organizations/me/keys` for a caller who does not operate the
      // deployment, so the destination serves every signed-in identity
      // something true, their own keys.
      {
        to: "/keys",
        label: "API keys",
        surface: "keys",
        icon: FiKey,
      },
      // The selected workspace's roster, not the organization's. The
      // organization roster is "Members & roles" in the other context, and the
      // two pages cross-link, which is the distinction the prototype draws.
      {
        to: "/members",
        label: "Members",
        surface: "workspaces",
        icon: FiUsers,
      },
    ],
  },
] as const satisfies readonly NavSection[]

/**
 * The organization context: what belongs to the tenant rather than to one
 * workspace inside it.
 *
 * Reached from the sidebar footer and left by the "Back to" link at its top,
 * so the two contexts never render together. Gated on the caller managing the
 * organization, which in a standalone deployment is always true: there is one
 * session, the local operator, and it owns the organization the gateway
 * provisioned for itself. The gate is written anyway because it is the thing
 * that becomes load-bearing the moment per-user sign-in lands (otari-ai#1716).
 *
 * One of the design's rows is **declared and gated on a surface the standalone
 * bootstrap does not report** (`STANDALONE_SURFACES` in
 * `src/gateway/api/routes/bootstrap.py` is that list), so it is absent here and
 * present on a deployment that serves it, and a group whose every row is gated
 * drops entirely, heading included. The organization guardrail ceiling is that
 * row, and this gateway serves no such surface at all; it is in Gateway, beside
 * nothing, which is why that whole group drops here.
 *
 * The organization's own provider credentials are not such a row:
 * `organization_providers` is published by both topologies, because the page
 * behind it is where an organization's models are offered, priced and switched.
 * It sits under General beside the process-global `providers` row, under a
 * different label; see that pair for why.
 *
 * The design draws two more, Billing and Gateways, and neither is declared here
 * at all, because neither is this build's to declare: Billing is
 * ARCHITECTURE.md's canonical overlay-only capability, and the attached-gateway
 * surface behind Gateways is hosted depth (otari-ai#1779), so an overlay owns
 * both rows together with the pages under them and contributes them through
 * `overlayNavItems.ts`. Declaring either here would be worse than redundant: a
 * base route file at a path the overlay also serves is not shadowed by the
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
    // The workspace rail's "Observe" question asked one level up. One row for
    // now, with a heading all the same: unlike the index's headingless section,
    // this is a category (an organization Activity page would join it), and a
    // bare "Usage" row above "People & access" would read as a stray.
    id: "org-observe",
    label: "Observe",
    items: [
      // The organization as a whole, which the workspace rail's Usage page
      // cannot ask for: its scope is the sidebar's selected workspace, and the
      // switcher offers only the caller's own memberships (otari-ai#1963).
      // Gated on a surface only the hosted bootstrap reports, because on a
      // standalone deployment the organization is the deployment and `/usage`
      // already answers it whole. No `operatorOnly` and no role gate here: the
      // organization rail already opens only to a caller who manages the
      // organization, and the server scopes a member who lands here by URL to
      // their own workspaces, the same rows `/usage` shows them.
      {
        to: "/organization/usage",
        label: "Usage",
        surface: "organization_usage",
        icon: FiBarChart2,
      },
    ],
  },
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
      },
    ],
  },
  {
    id: "org-gateway",
    label: "Gateway",
    // Empty in this build, so the rail drops it, heading included. Declared
    // anyway because it is an overlay seam: an overlay contributes Gateways into
    // this section by its id (`overlayNavItems.ts`), and a section that is not
    // declared has nothing for that row to join.
    items: [],
  },
  {
    id: "org-general",
    label: "General",
    items: [
      // Two rows, and on a standalone deployment both render, which is why they
      // cannot share a label. The organization row is where an organization's
      // models are offered, priced and switched, a tenant's question on either
      // topology, so both surfaces are published there; hosted reports only the
      // organization one.
      //
      // The organization's own row keeps the bare noun, because this rail is
      // already scoped to the organization and the row whose scope is *not* the
      // rail's is the one that needs qualifying. "Deployment providers" is also
      // what its `operatorOnly` already says in the gating: those credentials
      // are keyed on an instance name and belong to the process, so every
      // organization on the deployment is served them.
      {
        to: "/organization/provider-keys",
        label: "Providers",
        surface: "organization_providers",
        icon: FiBox,
      },
      // `operatorOnly` here rather than on the organization rail's own entry
      // point is the one exception to the deployment-rail placement the other
      // two `operatorOnly` rows follow: a deployment operator who is not an
      // organization owner or admin has no rail path to this row at all. Silent
      // today, because standalone's one operator always owns the organization
      // its own gateway provisioned; it stops holding once the rail's own gate
      // (docblock above) becomes load-bearing at otari-ai#1716.
      {
        to: "/providers",
        label: "Deployment providers",
        surface: "providers",
        icon: FiBox,
        operatorOnly: "refused",
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
 * Both rows keep the refusal they declare, which is the axis the server
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
          // gated-off panel call that page (`/tools`); the group and the page
          // sharing a name is why that reads correctly rather than as a bug.
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
 * one, and this rail is why: Billing lands inside "Cost & billing" and Gateways
 * inside "Gateway", among rows the base declares, so appending a section could
 * only put a second heading of the same name below the first. This build appends
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
 * distinction this rail draws is one an overlay can land on: a hosted edition
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
// Availability follows the gateway's inference surface, not a paid capability.
export const PLAYGROUND_NAV_ITEM = {
  to: "/playground",
  label: "Playground",
  surface: "playground",
  icon: FiMessageSquare,
} as const satisfies NavItem

/** Route lookup includes chrome destinations as well as all three sidebar scopes. */
export const NAV_ITEMS: readonly NavItem[] = [
  PLAYGROUND_NAV_ITEM,
  ...[...NAV_SECTIONS, ...ORG_NAV_SECTIONS, ...DEPLOYMENT_NAV_SECTIONS].flatMap(
    (section) => section.items,
  ),
]

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
  const child = NAV_ITEMS.flatMap((item) => item.children ?? []).find(
    (one) => one.to === pathname,
  )
  return child?.label ?? navItemForPath(pathname)?.label
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
