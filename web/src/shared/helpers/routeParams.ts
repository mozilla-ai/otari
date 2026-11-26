/**
 * The parameters the links into the pages in front of a session carry.
 *
 * Shared rather than private to one page because those pages span two
 * features (`features/auth`'s verify, reset and signup, `features/invitations`'
 * accept), each landed on from a link the gateway mailed or from the accept
 * page's own handoff. A link's secret rides in its fragment
 * (`/verify-email#token=…`), which `routeParam` reads ahead of the query.
 */
import { routeParam } from "@/shared/helpers/appLocation"

/** The `token` in `/verify-email#token=…`, or null if the link is malformed. */
export function tokenFromRoute(route: string): string | null {
  return routeParam(route, "token")
}

/**
 * The `email` in `/signup?email=…`, or null when the link carries none.
 *
 * Not a credential and not trusted as one: it prefills a form field, and
 * `POST /v1/auth/signup` checks the address against the roster itself.
 * `SignupPage` is the reader and carries the rest of the reasoning.
 */
export function emailFromRoute(route: string): string | null {
  return routeParam(route, "email")
}

/** The `invite` a build's invitation link carries, which proves the address it was mailed to. */
export function inviteFromRoute(route: string): string | null {
  return routeParam(route, "invite")
}
