/**
 * The organization slug remembered after a sign-in through its own connection,
 * so returning here skips straight back to it.
 *
 * `localStorage`, not `sessionStorage`: the point is to survive between
 * browser sessions on this device. Storage can throw in a private window, with
 * site data blocked, or in a preview, so every read and write is wrapped in
 * try/catch and the sign-in card renders the same either way: nothing
 * remembered, asking for an address again.
 */
const STORAGE_KEY = "otari.dashboard.rememberedCompany"

export function rememberCompany(slug: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, slug)
  } catch {
    // Storage off: nothing is remembered for next time.
  }
}

export function readRememberedCompany(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function forgetRememberedCompany(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Nothing was stored.
  }
}
