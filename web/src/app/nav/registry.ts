import {
  ActivityIcon,
  BudgetsIcon,
  KeysIcon,
  ModelsIcon,
  OverviewIcon,
  ProvidersIcon,
  RoutingIcon,
  SettingsIcon,
  ToolsIcon,
  UsageIcon,
  UsersIcon,
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
 * "Access" is who may call it (users, keys, budgets); the unlabeled first and
 * last groups hold the index and the standalone config, set off by a divider
 * rather than a heading. Grouping keeps the list legible as the dashboard
 * grows.
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
      {
        to: "/routing",
        label: "Routing",
        surface: "routing",
        // The one base entry that also carries an entitlement, and the reason
        // the gate is not dead code here. Routing is a capability in its own
        // right (ARCHITECTURE.md's capability lines: core base plus an overlay
        // adapter for richer model selection), and otari.ai's registry already
        // gates its own Routing item on this same name. The base build entitles
        // it, so it renders; tagging it now is what keeps one nav entry from
        // meaning two different things once the registries converge.
        capability: "routing",
        icon: RoutingIcon,
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
 * An unregistered path (`/docs`, the 404 splat) has no entry and is therefore
 * never gated, which is right: the registry governs the destinations it
 * declares and nothing else.
 */
export function navItemForPath(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find(
    (item) =>
      pathname === item.to ||
      (item.to !== "/" && pathname.startsWith(`${item.to}/`)),
  )
}
