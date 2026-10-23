// Non-secret marker that a session cookie was minted for this browser. The
// credential itself is an HttpOnly cookie the page cannot read, so this flag is
// what lets the app render signed-in synchronously on load instead of probing
// the server first. If it is ever stale (cookie expired or revoked), the first
// 401 drops it and bounces to sign-in, exactly like any mid-session revocation.
// `AuthProvider` owns it; the kept organization context reads it too.
export const SESSION_MARKER_KEY = "otari.dashboard.hasSession"

export function hasSessionMarker(): boolean {
  try {
    return window.localStorage.getItem(SESSION_MARKER_KEY) === "1"
  } catch {
    return false
  }
}
