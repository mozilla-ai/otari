import {
  ActivityIcon,
  BudgetsIcon,
  KeysIcon,
  MembersIcon,
  ModelsIcon,
  OrganizationIcon,
  OverviewIcon,
  ProvidersIcon,
  RoutingIcon,
  SettingsIcon,
  ToolsIcon,
  UsageIcon,
  UsersIcon,
  WorkspacesIcon,
} from "./icons"
import {
  OVERLAY_NAV_SECTIONS,
  OVERLAY_ORG_NAV_SECTIONS,
} from "./overlaySections"
import type { NavItem, NavSection } from "./types"

/**
 * The sidebar the base build ships, and the only place a destination is
 * declared.
 *
 * Sections render in this order. "Observability" is what the gateway did (the
 * request log and the usage rollups over it); "Catalog" is what the gateway
 * serves (providers, their models, and the policies that route over them);
 * "Organization" is the tenant the deployment belongs to (itself, its roster,
 * and the workspaces it is divided into); "Access" is who may call it (users,
 * keys, budgets); the unlabeled first and last groups hold the index and the
 * standalone config, set off by a divider rather than a heading. Grouping keeps
 * the list legible as the dashboard grows.
 *
 * Each entry declares its own gating, and the three axes are independent:
 * `surface` (does this deployment host it), `capability` (is it entitled), and
 * `flag` (is its rollout on). The sidebar composes them as AND. Hiding a link
 * is a convenience and never an authorization; the server still authorizes
 * every request the page behind it makes.
 */
const BASE_NAV_SECTIONS = [
  {
    id: "home",
    items: [
      // Ungated on every axis: the index is the deployment's own front page and
      // reads whatever it is allowed to.
      { to: "/", label: "Overview", icon: OverviewIcon },
    ],
  },
  {
    id: "observe",
    label: "Observe",
    items: [
      // Both read /v1/usage, which is why they name the surface rather than
      // themselves: a deployment that does not host usage loses both.
      {
        to: "/activity",
        label: "Activity",
        surface: "usage",
        icon: ActivityIcon,
      },
      { to: "/usage", label: "Usage", surface: "usage", icon: UsageIcon },
    ],
  },
  {
    id: "gateway",
    label: "Gateway",
    items: [
      { to: "/models", label: "Models", surface: "models", icon: ModelsIcon },
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
        icon: RoutingIcon,
        // Policies and Guardrails as the navigation prototype groups them. The
        // prototype's third entry, Aliases, is deliberately absent: this
        // dashboard lists an alias as the one-target policy it is, in the same
        // table (see `RoutingPage`), so `/aliases` is a compatibility redirect
        // onto `/routing` rather than a destination. Linking it would give the
        // group two entries for one page, and the second could never highlight.
        // It comes back if and when Routing grows a separate alias view.
        children: [
          { to: "/routing", label: "Policies" },
          {
            to: "/tools/guardrails",
            label: "Guardrails",
            // Grouped with Routing, served by the tools surface.
            surface: "tools",
          },
        ],
      },
      {
        to: "/tools",
        label: "Tools",
        surface: "tools",
        icon: ToolsIcon,
        // Two of the three services the page configures; Guardrails is grouped
        // under Routing, where the prototype puts it. The prototype lists MCP
        // servers here too, and the gateway has no MCP server registry to
        // manage (only per-request config a caller passes in, plus two safety
        // toggles on Settings), so it is left out rather than linked to an
        // empty page.
        children: [
          { to: "/tools/web-search", label: "Web search" },
          { to: "/tools/code-execution", label: "Code execution" },
        ],
      },
    ],
  },
  {
    id: "access",
    label: "Access",
    items: [
      { to: "/keys", label: "API keys", surface: "keys", icon: KeysIcon },
      {
        to: "/providers",
        label: "Provider credentials",
        surface: "providers",
        icon: ProvidersIcon,
      },
      // The selected workspace's roster, not the organization's. The
      // organization roster is "Members & roles" in the other context, and the
      // two pages cross-link, which is the distinction the prototype draws.
      {
        to: "/members",
        label: "Members",
        surface: "workspaces",
        icon: MembersIcon,
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
 * Four entries in the prototype have no page here and are deliberately absent
 * rather than stubbed: Billing, Gateways, Guardrail ceiling, and a separate
 * org-scoped provider-credentials view.
 */
const ORGANIZATION_NAV_SECTIONS = [
  {
    id: "org-people",
    label: "People & access",
    items: [
      {
        to: "/organization/members",
        label: "Members & roles",
        surface: "organizations",
        icon: MembersIcon,
      },
      {
        to: "/workspaces",
        label: "Workspaces",
        surface: "workspaces",
        icon: WorkspacesIcon,
      },
    ],
  },
  {
    id: "org-money",
    label: "Money",
    items: [
      {
        to: "/budgets",
        label: "Spend & budgets",
        surface: "budgets",
        icon: BudgetsIcon,
      },
      // Absent from the prototype, which folds per-user spend into "Spend &
      // budgets". Kept because it is still the only place a budget is attached
      // to anything: a budget names a `users` row, and that table has not merged
      // into the tenancy identity yet (M4). It moves under Spend & budgets, and
      // stops being a destination, when it does.
      { to: "/users", label: "Users", surface: "users", icon: UsersIcon },
    ],
  },
  {
    id: "org-general",
    label: "General",
    items: [
      {
        to: "/organization",
        label: "Organization",
        surface: "organizations",
        icon: OrganizationIcon,
      },
      {
        to: "/settings",
        label: "Settings",
        surface: "settings",
        icon: SettingsIcon,
      },
    ],
  },
] as const satisfies readonly NavSection[]

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
 * This build appends nothing, so it is the base sections alone.
 */
export const NAV_SECTIONS: readonly NavSection[] = composeNavSections(
  BASE_NAV_SECTIONS,
  OVERLAY_NAV_SECTIONS,
)

/**
 * The composed organization sidebar.
 *
 * Composed the same way the workspace rail is, and for the same reason: Billing
 * is the canonical overlay-only capability (ARCHITECTURE.md's capability table)
 * and it belongs on this rail, so an overlay that could only contribute to the
 * workspace one would have to edit this file to register it, which is what
 * cardinal rule 6 rules out. This build appends nothing.
 */
export const ORG_NAV_SECTIONS: readonly NavSection[] = composeNavSections(
  ORGANIZATION_NAV_SECTIONS,
  OVERLAY_ORG_NAV_SECTIONS,
)

/**
 * Every registered entry, across both contexts.
 *
 * Flattened over both because this is what answers "which entry is this
 * pathname", and a route is gated the same way whichever sidebar links to it.
 */
export const NAV_ITEMS: readonly NavItem[] = [
  ...NAV_SECTIONS,
  ...ORG_NAV_SECTIONS,
].flatMap((section) => section.items)

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

/** Where a destination lives: the workspace sidebar, or the organization one. */
export type NavContext = "workspace" | "organization"

const ORG_PATHS: readonly string[] = ORG_NAV_SECTIONS.flatMap((section) =>
  section.items.map((item) => item.to),
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
 * Derived from the registry rather than from a path prefix, because the two
 * contexts do not split cleanly by URL: `/workspaces` and `/settings` are
 * organization destinations whose paths look like anything else, and
 * `/members` is a workspace one that sits directly under the root. Anything
 * unregistered (the guide, the 404 splat) belongs to the workspace context,
 * which is the one the shell opens in.
 */
export function navContextForPath(pathname: string): NavContext {
  const item = navItemForPath(pathname)
  if (!item) return "workspace"
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
