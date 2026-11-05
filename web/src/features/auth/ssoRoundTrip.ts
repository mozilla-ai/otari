import { type SsoStartResult, startSsoSignIn } from "@/shared/api/client"

import { goToPublicAuthPage } from "./PublicAuthLayout"
import { resolveSignInReturnHash } from "./publicAuthPaths"

/**
 * Where the CSRF `state`, the organization's slug and the terms acceptance
 * wait while the browser is away at the organization's own identity provider.
 *
 * `sessionStorage`, scoped to the tab that started the sign-in and gone once
 * it is read. The slug names the organization's connection to continue with
 * or, on a refusal that asks for the terms, to retry.
 */
const STATE_KEY = "otari.sso.state"
const SLUG_KEY = "otari.sso.slug"
const TERMS_KEY = "otari.sso.terms-accepted"
const RETURN_KEY = "otari.sso.return-hash"

interface RoundTripOptions {
  termsAccepted?: boolean
  /** Home unless the page starting the round trip stands in for another. */
  returnHash?: string
}

/** Remember the slug, the state, the terms acceptance and the page to return to, for the round trip about to start. */
export function rememberSsoState(
  slug: string,
  state: string,
  { termsAccepted = false, returnHash = "#/" }: RoundTripOptions = {},
): void {
  window.sessionStorage.setItem(STATE_KEY, state)
  window.sessionStorage.setItem(SLUG_KEY, slug)
  if (termsAccepted) {
    window.sessionStorage.setItem(TERMS_KEY, "true")
  } else {
    window.sessionStorage.removeItem(TERMS_KEY)
  }
  window.sessionStorage.removeItem(RETURN_KEY)
  try {
    window.sessionStorage.setItem(RETURN_KEY, returnHash)
  } catch {
    // Storage full: the sign-in still starts, and lands on the home page.
  }
}

/** Take what was remembered, clearing it so one value answers one callback. */
export function takeSsoState(): {
  slug: string
  state: string
  termsAccepted: boolean
} {
  const slug = window.sessionStorage.getItem(SLUG_KEY) ?? ""
  const state = window.sessionStorage.getItem(STATE_KEY) ?? ""
  const termsAccepted = window.sessionStorage.getItem(TERMS_KEY) === "true"
  window.sessionStorage.removeItem(SLUG_KEY)
  window.sessionStorage.removeItem(STATE_KEY)
  window.sessionStorage.removeItem(TERMS_KEY)
  return { slug, state, termsAccepted }
}

/**
 * Take the page to return to, cleared with the state whatever the outcome, and
 * checked on the way out: storage is not a source to navigate on unchecked.
 */
export function takeSsoReturnHash(): string {
  const stored = window.sessionStorage.getItem(RETURN_KEY) ?? ""
  window.sessionStorage.removeItem(RETURN_KEY)
  return resolveSignInReturnHash(stored)
}

/**
 * Send the browser to one organization's own identity provider, with whatever
 * finishing the sign-in will need already held.
 *
 * Returns a refusal where one will not start; otherwise the tab is already
 * navigating to that provider.
 */
export async function leaveForSsoConnection(
  slug: string,
  options: RoundTripOptions = {},
): Promise<SsoStartResult> {
  const started = await startSsoSignIn(slug)
  if (started.isOk) {
    rememberSsoState(slug, started.state, options)
    goToPublicAuthPage(`#/sso/${encodeURIComponent(slug)}`)
    window.location.assign(started.authorizationUrl)
  }
  return started
}
