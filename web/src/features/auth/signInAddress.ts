/**
 * The address a page in front of the session hands to the next one, so
 * somebody moving between sign-in, recovery, verification and an invitation
 * types it once: sign-in to "Forgot your password?", signup or a resend to the
 * check-email page and on to another resend, and a finished verification,
 * recovery request or invitation back to sign-in.
 *
 * Session storage rather than the URL: the hash is the address bar the router
 * reads after sign-in, and an address in it would ride along into the
 * dashboard. Only ever an address the visitor typed or a link proved, never a
 * password. Read without being removed, because React may run a state
 * initializer twice; the sign-in screen forgets it once it has mounted.
 */
const STORAGE_KEY = "otari.dashboard.signInAddress"

export function rememberSignInAddress(email: string): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, email)
  } catch {
    // Storage off: the next page opens with an empty address.
  }
}

export function readSignInAddress(): string {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY) ?? ""
  } catch {
    return ""
  }
}

export function forgetSignInAddress(): void {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // Nothing was stored.
  }
}
