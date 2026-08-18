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
import { OVERLAY_NAV_SECTIONS } from "./overlaySections"
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
    id: "observability",
    label: "Observability",
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
    id: "catalog",
    label: "Catalog",
    items: [
      {
        to: "/providers",
        label: "Providers",
        surface: "providers",
        icon: ProvidersIcon,
      },
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
      },
    ],
  },
  {
    // The label is the one mozilla-ai/otari-ai#1539 is assigned to change when
    // the control-plane UI rehomes: "Organization" reads as a multi-tenancy
    // word, and that issue's plan is for the OSS sidebar to say something
    // single-tenant while the enterprise overlay overrides it back. Left as is
    // rather than pre-empted, because the obvious substitute ("Settings")
    // already names an item two sections down, and because standalone Otari now
    // does hold more than one organization, which is the premise that issue was
    // written against. Rename it there, with the override, not here.
    id: "organization",
    label: "Organization",
    items: [
      // Two pages over /v1/organizations, so they name the surface rather than
      // themselves; the workspace pages are a separate surface because a
      // deployment could serve one without the other.
      {
        to: "/organization",
        label: "General",
        surface: "organizations",
        icon: OrganizationIcon,
      },
      {
        to: "/organization/members",
        label: "Members",
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
    id: "access",
    label: "Access",
    items: [
      { to: "/users", label: "Users", surface: "users", icon: UsersIcon },
      { to: "/keys", label: "API keys", surface: "keys", icon: KeysIcon },
      {
        to: "/budgets",
        label: "Budgets",
        surface: "budgets",
        icon: BudgetsIcon,
      },
    ],
  },
  {
    id: "system",
    items: [
      {
        to: "/tools",
        label: "Tools & Guardrails",
        surface: "tools",
        icon: ToolsIcon,
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
 * The composed sidebar.
 *
 * This build appends nothing, so it is the base sections alone.
 */
export const NAV_SECTIONS: readonly NavSection[] = composeNavSections(
  BASE_NAV_SECTIONS,
  OVERLAY_NAV_SECTIONS,
)

/** Every registered entry, flattened out of its section. */
export const NAV_ITEMS: readonly NavItem[] = NAV_SECTIONS.flatMap(
  (section) => section.items,
)

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
