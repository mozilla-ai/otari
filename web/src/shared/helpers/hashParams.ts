/**
 * Reading a query parameter out of a hash route.
 *
 * The dashboard runs on hash history, so `window.location.search` is empty and
 * a link's parameters live after the `?` *inside* the fragment
 * (`#/verify-email?token=…`). `URLSearchParams` cannot find them on its own,
 * which is what this splits out.
 *
 * Shared rather than private to one page because the pages that read a
 * parameter are the pages in front of a session, and there are four of them
 * across two features (`features/auth`'s verify, reset and signup,
 * `features/invitations`' accept), each landed on from a link the gateway
 * mailed or from the accept page's own handoff.
 */

function paramFromHash(hash: string, name: string): string | null {
  const value = new URLSearchParams(hash.split("?")[1] ?? "").get(name)
  // A truncated link (`?token=` with nothing after it) is malformed, not a
  // value of length zero, and the difference is a page that strands. Every
  // caller branches on null to say "this link carries nothing to act on", and
  // gates its request on a non-empty string; an empty string satisfies neither,
  // so the page would sit on its loading line forever having asked nothing.
  return value === "" ? null : value
}

/** The `token` in `#/verify-email?token=…`, or null if the link is malformed. */
export function tokenFromHash(hash: string): string | null {
  return paramFromHash(hash, "token")
}

/**
 * The `email` in `#/signup?email=…`, or null when the link carries none.
 *
 * Not a credential and not trusted as one: it only prefills the signup form's
 * address field, and the claim it stands for is checked by
 * `POST /v1/auth/signup` against the roster, which answers the same sentence
 * for an address it has nothing to do with. The accept page puts it here
 * (otari#835) so an invitee who just accepted does not have to retype the
 * address their invitation was sent to.
 */
export function emailFromHash(hash: string): string | null {
  return paramFromHash(hash, "email")
}
