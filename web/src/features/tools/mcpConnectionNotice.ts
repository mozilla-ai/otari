/**
 * What carries an MCP sign-in from the page that started it, through the
 * server's own sign-in page, back to a dashboard page.
 *
 * Two halves, kept for different spans:
 *
 * - **The page a sign-in started from** outlives the full-page trip to the
 *   provider, so it is kept in session storage. The gateway hands the page back
 *   (`return_to`) on success and on a cancel, but not on a cancel that arrives
 *   after the sign-in expired, and that is the case this exists for.
 * - **The notice** is said once by the page the callback lands on, so it is
 *   kept in memory, as `features/auth/signInNotice.ts` does for the sign-in
 *   screen. A sign-in that ran in a popup comes back to a page that is already
 *   showing, so a banner on screen also hears of a notice as it is set.
 */

const ORIGIN_KEY = "otari.mcp-sign-in.origin"

/** Only what did not happen is ever said: a page shows a result that worked. */
export interface McpConnectionNotice {
  tone: "info" | "warning"
  message: string
}

let notice: McpConnectionNotice | undefined
const listeners = new Set<(next: McpConnectionNotice) => void>()

export function rememberMcpConnectionNotice(next: McpConnectionNotice): void {
  notice = next
  for (const listener of listeners) listener(next)
}

/** Hears every notice set from now on. Returns the unsubscribe. */
export function subscribeToMcpConnectionNotice(
  listener: (next: McpConnectionNotice) => void,
): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Read without being removed, because React may run a state initializer twice. */
export function readMcpConnectionNotice(): McpConnectionNotice | undefined {
  return notice
}

export function forgetMcpConnectionNotice(): void {
  notice = undefined
}

const CALLBACK_PATH = "/mcp-connections/callback"
const DASHBOARD_ORIGIN = "http://dashboard.invalid"

/**
 * A dashboard path to land on after a sign-in, and nothing that could leave the
 * dashboard. The callback page itself is refused: landing there again finishes
 * nothing and shows it was opened without a sign-in.
 */
export function pickDashboardPath(
  path: string | null | undefined,
): string | undefined {
  // A backslash reads as a slash in a URL, so "/\\host" would leave the dashboard.
  if (!path?.startsWith("/") || path.startsWith("//") || path.includes("\\")) {
    return undefined
  }
  // Resolved as the router would read it: dot segments, case, a trailing slash.
  let resolved: URL
  try {
    resolved = new URL(path, DASHBOARD_ORIGIN)
  } catch {
    return undefined
  }
  if (resolved.origin !== DASHBOARD_ORIGIN) return undefined
  const { pathname } = resolved
  const isCallback =
    pathname.replace(/\/+$/, "").toLowerCase() === CALLBACK_PATH
  return isCallback ? undefined : path
}

/** Notes the page a sign-in is leaving from. Storage can be refused; then there is none. */
export function rememberSignInOrigin(path: string): void {
  try {
    window.sessionStorage.setItem(ORIGIN_KEY, path)
  } catch {
    // Private mode or blocked storage: the callback falls back to its own page.
  }
}

/** The page a sign-in started from in this tab, if it noted one. */
export function readSignInOrigin(): string | undefined {
  try {
    return pickDashboardPath(window.sessionStorage.getItem(ORIGIN_KEY))
  } catch {
    return undefined
  }
}

export function forgetSignInOrigin(): void {
  try {
    window.sessionStorage.removeItem(ORIGIN_KEY)
  } catch {
    // Nothing was stored if storage is refused.
  }
}
