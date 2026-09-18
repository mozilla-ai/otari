/**
 * Where you were on each rail, so crossing between them does not lose your place.
 *
 * The design's `Interaction · A ⇄ B` note is explicit about this: pressing
 * "Organization" should open the organization page you were last on, and "Back
 * to {Workspace}" should return you to the workspace page you left, rather than
 * both journeys resetting to a fixed landing. The shell used to send you to
 * `/organization/members` and `/` every time, which treats a rail as a place you
 * arrive at rather than one you were already working in.
 *
 * Three things keep a stored value honest. It is **written only for a registered
 * destination**, and read back through the same check, so a stale entry left by
 * an older build, a hand-edited localStorage, or a route that has since been
 * removed is dropped rather than navigated to. It is **written and read only for
 * a destination this deployment still shows**, which is a separate question: a
 * gated-off route is still registered and still reachable by URL, so without
 * that check the resume link lands on the shell's "not available here" panel,
 * and the visit that saw the panel is recorded as somewhere worth returning to.
 * That second check is `isPathVisible` rather than a rule of its own, so the
 * answer here is by construction the one the shell will give when the link is
 * followed. And because the first check resolves against the registry, what
 * comes back is a `NavPath`: the caller gets something a `Link` will accept, not
 * a string it has to cast.
 *
 * It lives here rather than in `shared/helpers` (where `otari-ai` keeps its
 * counterpart) because deciding which rail a path belongs to is
 * `navContextForPath`'s job, and `src/shared` may not import `src/app`. Over
 * there the same decision is a pair of path-prefix regexes, which this
 * dashboard's URLs do not support: `/workspaces` and `/settings` are
 * organization destinations that look like anything else.
 */

import {
  isPathVisible,
  NAV_ITEMS,
  type NavContext,
  navContextForPath,
} from "./registry"
import type { NavItem, NavPath } from "./types"

const STORAGE_KEYS: Record<NavContext, string> = {
  workspace: "otari.dashboard.lastWorkspaceLocation",
  organization: "otari.dashboard.lastOrganizationLocation",
  deployment: "otari.dashboard.lastDeploymentLocation",
}

/**
 * Which rail to return to when leaving the deployment one.
 *
 * A context rather than a location, which is why it is not in the record above:
 * the deployment rail is reached from a control that renders on both of the
 * other rails, so "back" has no static answer and the shell holds nothing else
 * that remembers which one you were on.
 */
const RAIL_CONTEXT_KEY = "otari.dashboard.lastRailContext"

/**
 * A place to return to.
 *
 * The destination only, not its filters. The design asks for the page you were
 * last on, and this dashboard keeps every filter in the URL: restoring those too
 * would drop you back into a twelve-month usage window or a model filter you set
 * days ago, which reads as the app having lost your place rather than kept it.
 */
export interface RememberedLocation {
  to: NavPath
}

/**
 * The registered destination at this pathname, if it is one.
 *
 * Exact matches only, leaves and nested children alike. A deeper path
 * (`/organization/members/abc`) resolves to no destination and is therefore not
 * remembered, which is the right trade: the design asks for the page you were
 * on, and a detail view's id may not exist by the time you come back.
 *
 * Note the asymmetry with the visibility question below, which is deliberate:
 * *what counts as a destination* is stricter here than in the registry's own
 * prefix matching, while *whether a destination is served* is the registry's
 * answer verbatim, because that is the answer the shell will act on when the
 * link is followed.
 */
function destinationAt(pathname: string): NavPath | undefined {
  return NAV_ITEMS.flatMap((item) => [
    item.to,
    ...(item.children ?? []).map((child) => child.to),
  ]).find((to) => to === pathname)
}

/**
 * Record the current location under the rail it belongs to.
 *
 * Called on every navigation, and a no-op for anything that is not a registered
 * destination, so the guide and the 404 splat never become somewhere to return
 * to. Also a no-op for a destination this deployment gates off, which a URL can
 * still reach and which answers with the "not available here" panel: the page
 * someone was just told they cannot have is not the page to send them back to.
 * Both rails are recorded the same way, which is what makes the memory right
 * however you left: a link on a page and a bookmark update it as surely as the
 * control that crosses the rail does.
 */
export function rememberLocation(
  pathname: string,
  isVisible: (item: NavItem) => boolean,
): void {
  const to = destinationAt(pathname)
  if (!to || !isPathVisible(to, isVisible)) return
  const context = navContextForPath(pathname)
  try {
    window.localStorage.setItem(STORAGE_KEYS[context], to)
    // Which rail to come back to, which is any rail but the deployment's own.
    // Written unconditionally this would be a state that destroys its own
    // input: landing on a deployment page would record "deployment" as the
    // place to return to, and the back link reads this to decide where that is.
    if (context !== "deployment") {
      window.localStorage.setItem(RAIL_CONTEXT_KEY, context)
    }
  } catch {
    // Storage can throw when it is disabled (blocked cookies, private mode).
    // Losing the memory costs a landing page, so it is not worth an error path.
  }
}

/** Where you last were on this rail, if it is still somewhere you can go. */
export function lastLocation(
  context: NavContext,
  isVisible: (item: NavItem) => boolean,
): RememberedLocation | undefined {
  let raw: string | null = null
  try {
    raw = window.localStorage.getItem(STORAGE_KEYS[context])
  } catch {
    return undefined
  }
  if (!raw) return undefined
  // Re-validated on the way out, not just on the way in: a destination that has
  // been removed, moved to the other rail, or gated off since it was stored
  // would otherwise send someone to the shell's "not available here" panel, or
  // to the wrong rail. The gate is the half a write-time check cannot cover on
  // its own, because a gateway restarted against a config that reports fewer
  // surfaces changes the answer under an entry already written.
  const known = destinationAt(raw)
  if (!known || navContextForPath(known) !== context) return undefined
  if (!isPathVisible(known, isVisible)) return undefined
  return { to: known }
}

/**
 * The rail to return to from the deployment one.
 *
 * Validated rather than trusted in both directions. The stored string is
 * checked against the union, because a key left by an older build or edited by
 * hand would otherwise arrive as a `NavContext` the compiler believes; and the
 * remembered context is only offered when it still has somewhere reachable in
 * it, which is asked of `lastLocation` rather than reimplemented here so the two
 * answers cannot disagree.
 *
 * Falls back to the workspace rail, which covers both a browser that has never
 * recorded one (a bookmark straight to a deployment page) and a context that has
 * since been gated off. A back link into a rail someone can no longer use is a
 * worse failure than one that goes somewhere merely unexpected.
 */
export function lastRailContext(
  isVisible: (item: NavItem) => boolean,
): Exclude<NavContext, "deployment"> {
  let raw: string | null = null
  try {
    raw = window.localStorage.getItem(RAIL_CONTEXT_KEY)
  } catch {
    return "workspace"
  }
  if (raw !== "workspace" && raw !== "organization") return "workspace"
  return lastLocation(raw, isVisible) ? raw : "workspace"
}
