/**
 * The address a page in front of the session hands to the sign-in screen, so
 * somebody who has just set a password from an invitation link signs in
 * without typing the address again.
 *
 * Session storage rather than the URL: the hash is the address bar the router
 * reads after sign-in, and an address in it would ride along into the
 * dashboard. Read without being removed, because React may run a state
 * initializer twice; the sign-in screen forgets it once it has mounted.
 */
const STORAGE_KEY = "otari.dashboard.signInAddress"

export function rememberSignInAddress(email: string): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, email)
  } catch {
    // Storage off: the sign-in screen opens with an empty address.
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
