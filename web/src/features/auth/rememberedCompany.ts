/**
 * The organization remembered after a sign-in through its own connection, so
 * returning here skips straight back to it.
 *
 * Both halves are kept: the name to show a person, and the slug the sign-in
 * URL needs. A slug is an identifier, so showing one asks somebody to
 * recognize a string they never chose.
 *
 * `localStorage`, not `sessionStorage`: the point is to survive between
 * browser sessions on this device. Storage can throw in a private window, with
 * site data blocked, or in a preview, so every read and write is wrapped in
 * try/catch and the sign-in card renders the same either way: nothing
 * remembered, asking for an address again.
 */
const STORAGE_KEY = "otari.dashboard.rememberedCompany"

export interface RememberedCompany {
  name: string
  slug: string
}

export function rememberCompany(company: RememberedCompany): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(company))
  } catch {
    // Storage off: nothing is remembered for next time.
  }
}

export function readRememberedCompany(): RememberedCompany | null {
  let stored: string | null = null
  try {
    stored = window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
  if (stored === null || stored === "") {
    return null
  }
  try {
    const parsed: unknown = JSON.parse(stored)
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof (parsed as RememberedCompany).slug === "string" &&
      typeof (parsed as RememberedCompany).name === "string"
    ) {
      return parsed as RememberedCompany
    }
  } catch {
    // A slug stored before the name was kept. Showing it is what this fixed,
    // so it is dropped rather than rendered, and the next sign-in stores both.
    return null
  }
  return null
}

export function forgetRememberedCompany(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Nothing was stored.
  }
}
