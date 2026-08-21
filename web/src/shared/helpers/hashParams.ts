/**
 * Reading a query parameter out of a hash route.
 *
 * The dashboard runs on hash history, so `window.location.search` is empty and
 * a link's parameters live after the `?` *inside* the fragment
 * (`#/verify-email?token=…`). `URLSearchParams` cannot find them on its own,
 * which is what this splits out.
 *
 * Shared rather than private to one page because the pages that read a token
 * are the pages in front of a session, and there are three of them across two
 * features (`features/auth`'s verify and reset, `features/invitations`'
 * accept), each landed on from a link the gateway mailed.
 */

/** The `token` in `#/verify-email?token=…`, or null if the link is malformed. */
export function tokenFromHash(hash: string): string | null {
  return new URLSearchParams(hash.split("?")[1] ?? "").get("token")
}
