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

// A house: the deployment's front page, and what the navigation prototype
// uses for Overview.
export const OverviewIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M3.5 10.5 12 3.5l8.5 7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path d="M5.5 9v10.5h13V9" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M9.75 19.5v-5h4.5v5" strokeLinejoin="round" />
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

// A shield: these are stored credentials, which is what the prototype's
// Provider credentials entry draws, rather than the hardware serving them.
export const ProvidersIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path
      d="M12 3.5 19 6v6c0 4-2.9 6.9-7 8.5-4.1-1.6-7-4.5-7-8.5V6z"
      strokeLinejoin="round"
    />
  </svg>
)

// Stacked layers: a catalog of models, as the prototype draws it.
export const ModelsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path d="M12 3.5 20 8l-8 4.5L4 8z" strokeLinejoin="round" />
    <path d="m4 12 8 4.5L20 12" strokeLinejoin="round" strokeLinecap="round" />
    <path d="m4 16 8 4.5L20 16" strokeLinejoin="round" strokeLinecap="round" />
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

// A gear: the organization rail is where a deployment is configured, so it
// reads as settings rather than as a building.
export const OrganizationIcon = (
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

// Two people: a roster, as the prototype draws Members.
export const MembersIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <circle cx="9.5" cy="8.5" r="3" />
    <path d="M3.5 19.5a6 6 0 0 1 12 0" strokeLinecap="round" />
    <path d="M16 6.2a3 3 0 0 1 0 5.6" strokeLinecap="round" />
    <path d="M17.5 14.5a5 5 0 0 1 3 4.6" strokeLinecap="round" />
  </svg>
)

/** Stacked frames: the workspaces an organization is divided into. */
export const WorkspacesIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <rect x="3" y="4" width="12" height="9" rx="1.5" strokeLinejoin="round" />
    <path
      d="M8 17.5h11a1.5 1.5 0 0 0 1.5-1.5V8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M11 21h8a3 3 0 0 0 3-3"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
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

/** The key glyph belongs to API keys (Provider credentials uses the shield). */
export const KeysIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <g transform="translate(12 12) scale(0.86) translate(-12 -12)">
      <circle cx="7.5" cy="15.5" r="4.5" />
      <path d="M10.7 12.3 19 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m15 8 3 3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m17.5 5.5 3 3" strokeLinecap="round" strokeLinejoin="round" />
    </g>
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

// A wrench, as the prototype draws Tools.
export const ToolsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <g transform="translate(12 12) scale(0.86) translate(-12 -12)">
      <path
        d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>
  </svg>
)

// Sliders, not a gear: Organization is the gear now, and two identical glyphs
// two rows apart on the same rail is worse than either choice alone.
export const SettingsIcon = (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    className="h-5 w-5 shrink-0"
  >
    <path d="M4 7h9M17 7h3M4 17h3M11 17h9" strokeLinecap="round" />
    <circle cx="15" cy="7" r="2" />
    <circle cx="9" cy="17" r="2" />
  </svg>
)
