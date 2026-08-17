/**
 * The sidebar's glyphs, one element per destination.
 *
 * Split out of `registry.ts` so that module stays a pure data declaration, the
 * way `otari-ai/frontend/src/app/nav/registry.ts` is: over there an icon is a
 * `react-icons` `IconType` reference, and there is nothing to draw inline. This
 * dashboard draws its own marks and has no react-icons dependency, so they live
 * here instead of turning the registry into a wall of SVG.
 *
 * Every glyph is decorative: each nav link carries a visible text label (and an
 * `aria-label` when the rail is collapsed), so the icon must not be announced a
 * second time. Hence `aria-hidden` on all of them, which `AppShell.test.tsx`
 * asserts across the whole shell.
 */

/** Four panes: an at-a-glance dashboard of the gateway. */
export const OverviewIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <rect
      x="3.5"
      y="3.5"
      width="7"
      height="7"
      rx="1.5"
      strokeLinejoin="round"
    />
    <rect
      x="13.5"
      y="3.5"
      width="7"
      height="7"
      rx="1.5"
      strokeLinejoin="round"
    />
    <rect
      x="3.5"
      y="13.5"
      width="7"
      height="7"
      rx="1.5"
      strokeLinejoin="round"
    />
    <rect
      x="13.5"
      y="13.5"
      width="7"
      height="7"
      rx="1.5"
      strokeLinejoin="round"
    />
  </svg>
)

/** A pulse line: the per-request log of what the gateway served. */
export const ActivityIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M3 12h4l2.5-6 4 12 2.5-6H21"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

/** A bar chart: aggregate spend and volume over time, beside the activity log. */
export const UsageIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M4 20V10M10 20V4M16 20v-7M22 20H2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

/** A server stack: upstream provider services, distinct from the API-keys key. */
export const ProvidersIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <rect
      x="3.5"
      y="4.5"
      width="17"
      height="6"
      rx="1.5"
      strokeLinejoin="round"
    />
    <rect
      x="3.5"
      y="13.5"
      width="17"
      height="6"
      rx="1.5"
      strokeLinejoin="round"
    />
    <path
      d="M7 7.5h.01M7 16.5h.01"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

/** A cube net: the model catalog the gateway serves. */
export const ModelsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" strokeLinejoin="round" />
    <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" strokeLinejoin="round" />
  </svg>
)

/** Two paths forking from one call: the routing policies over the catalog. */
export const RoutingIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path d="M4 5h4l4 7 4-7h4" strokeLinejoin="round" />
    <path d="M4 19h4l4-7" strokeLinejoin="round" />
    <circle cx="19" cy="19" r="2" />
    <circle cx="19" cy="5" r="2" />
  </svg>
)

/** Two figures: the principals that keys and budgets attach to. */
export const UsersIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <circle cx="9" cy="8" r="3.2" strokeLinejoin="round" />
    <path
      d="M3.5 19a5.5 5.5 0 0 1 11 0"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M16 5.2a3.2 3.2 0 0 1 0 5.6M17.5 19a5.5 5.5 0 0 0-3-4.9"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

/** The key glyph belongs to API keys (Providers uses the server stack above). */
export const KeysIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <circle cx="7.5" cy="15.5" r="3.5" />
    <path
      d="M10 13l7-7M14 5l3 3M16.5 7.5l2-2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

/** A wallet: the spending limits callers are held to, alongside the keys. */
export const BudgetsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M3 7.5A1.5 1.5 0 0 1 4.5 6H18a1.5 1.5 0 0 1 1.5 1.5V9"
      strokeLinejoin="round"
    />
    <rect
      x="3"
      y="7.5"
      width="18"
      height="12"
      rx="1.5"
      strokeLinejoin="round"
    />
    <path d="M16 13.5h.01" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M21 12v3h-3.5a1.5 1.5 0 0 1 0-3H21z" strokeLinejoin="round" />
  </svg>
)

/** A wrench: the runtime tools and guardrails the gateway can call. */
export const ToolsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M14.7 6.3a4 4 0 0 1 5 5l-8.4 8.4a2 2 0 0 1-2.8 0l-2.2-2.2a2 2 0 0 1 0-2.8z"
      strokeLinejoin="round"
    />
    <path d="M12 9 5 16" strokeLinecap="round" />
  </svg>
)

/** A cog: the deployment's own runtime configuration. */
export const SettingsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <circle cx="12" cy="12" r="3" />
    <path
      d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
      strokeLinejoin="round"
    />
  </svg>
)
