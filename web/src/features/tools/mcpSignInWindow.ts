/**
 * The small window an MCP sign-in runs in, and how it reports back.
 *
 * The provider's sign-in page opens in a popup so the dashboard stays where it
 * is. The popup ends on the dashboard's own callback page (same origin), which
 * finishes the sign-in and says how it went over a `BroadcastChannel`, then
 * closes itself. Not `window.opener`: a provider that sends
 * Cross-Origin-Opener-Policy cuts the popup off from the page that opened it,
 * and a channel reaches every page of this origin regardless. Only pages of
 * this origin can post on it, so there is no origin to check.
 *
 * The callback page tells a popup sign-in from a full-page one by its `state`:
 * the opener notes the state in local storage before the popup leaves for the
 * provider. The state is the gateway's own random value, single-use, and the
 * popup still finishes the sign-in through the same session-bound POST a
 * full-page one does, so the note changes where the result is shown and
 * nothing about what is accepted.
 */

const CHANNEL = "otari.mcp-sign-in"
const PENDING_KEY = "otari.mcp-sign-in.windows"
/** The gateway forgets a sign-in after ten minutes; a note outliving that names nothing. */
const PENDING_LIFETIME_MS = 15 * 60_000
const WINDOW_NAME = "otari-mcp-sign-in"
const WINDOW_FEATURES = "popup,width=520,height=720"

export type McpSignInResult =
  | {
      state: string
      outcome: "connected" | "checked"
      serverId: string
      serverName: string
    }
  | { state: string; outcome: "cancelled" }
  | { state: string; outcome: "failed"; message: string }

/**
 * Opens the empty window the sign-in will run in. Called straight from the
 * press, before the gateway has answered, because a browser only lets a page
 * open a window while it is handling the person's own action. `undefined`
 * when the browser blocked it.
 */
export function openSignInWindow(): Window | undefined {
  try {
    const popup = window.open("", WINDOW_NAME, WINDOW_FEATURES)
    if (!popup) return undefined
    popup.document.title = "Signing in…"
    popup.document.body.textContent = "Opening the sign-in…"
    return popup
  } catch {
    return undefined
  }
}

/** Sends the opened window to the provider and cuts it off from this page. */
export function sendSignInWindow(popup: Window, url: string): void {
  popup.location.href = url
  try {
    popup.opener = null
  } catch {
    // Some browsers refuse the write once the window has left this origin.
  }
}

/** The `state` an authorization URL carries, or `""` when it carries none. */
export function stateOf(authorizationUrl: string): string {
  try {
    return new URL(authorizationUrl).searchParams.get("state") ?? ""
  } catch {
    return ""
  }
}

function readPending(): Record<string, number> {
  try {
    const raw = window.localStorage.getItem(PENDING_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : {}
    if (typeof parsed !== "object" || parsed === null) return {}
    const now = Date.now()
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, number] =>
          typeof entry[1] === "number" && now - entry[1] < PENDING_LIFETIME_MS,
      ),
    )
  } catch {
    return {}
  }
}

function writePending(pending: Record<string, number>): void {
  try {
    window.localStorage.setItem(PENDING_KEY, JSON.stringify(pending))
  } catch {
    // Blocked storage: the callback then treats the window as a full page.
  }
}

/** Notes that the sign-in with this state runs in a popup. */
export function markSignInWindow(state: string): void {
  if (state === "") return
  writePending({ ...readPending(), [state]: Date.now() })
}

/** Whether this sign-in was started in a popup by a dashboard page. */
export function isSignInWindow(state: string | undefined): boolean {
  return state !== undefined && state !== "" && state in readPending()
}

export function forgetSignInWindow(state: string): void {
  const pending = readPending()
  if (!(state in pending)) return
  delete pending[state]
  writePending(pending)
}

/** Tells the page that opened the sign-in how it went. */
export function publishSignInResult(result: McpSignInResult): void {
  if (typeof BroadcastChannel === "undefined") return
  const channel = new BroadcastChannel(CHANNEL)
  channel.postMessage(result)
  channel.close()
}

function isResult(value: unknown): value is McpSignInResult {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { state?: unknown }).state === "string" &&
    typeof (value as { outcome?: unknown }).outcome === "string"
  )
}

/** Calls `onResult` once with the result for `state`. Returns the unsubscribe. */
export function subscribeToSignInResult(
  state: string,
  onResult: (result: McpSignInResult) => void,
): () => void {
  if (typeof BroadcastChannel === "undefined") return () => {}
  const channel = new BroadcastChannel(CHANNEL)
  channel.onmessage = (event: MessageEvent<unknown>) => {
    if (!isResult(event.data) || event.data.state !== state) return
    channel.close()
    onResult(event.data)
  }
  return () => channel.close()
}
