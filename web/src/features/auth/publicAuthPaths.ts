/**
 * The hash paths that answer in front of a session, and what each one needs
 * from the deployment before it can work.
 *
 * These are deliberately not files in `src/routes/`, for the reason
 * `AcceptInvitationPage` is not one either: every route there sits behind the
 * session gate `App.tsx`'s `DeploymentRoot` puts in front of the router, and
 * the whole point of these six is that the visitor holds neither a session nor
 * the master key. `DeploymentRoot` renders them directly, ahead of that gate.
 *
 * The gateway agrees on the spelling rather than being told it: the
 * verification and reset messages it sends carry `/#/verify-email?token=…` and
 * `/#/reset-password?token=…` (`services/tenancy/user_service.py`), so a
 * rename here breaks a link that is already in somebody's inbox.
 */

/**
 * Whether a page's flow can work at all on this deployment.
 *
 * `mail` means the flow starts by sending a message: `POST /v1/auth/signup`,
 * `/resend-verification` and `/password/reset` each call `require_ready()` and
 * refuse with a 503 when this gateway has no transport or no public URL of its
 * own to put in a link. The bootstrap publishes the same fact as `mail_ready`,
 * so the affordance is hidden rather than offered and then refused, the way
 * otari#648 already settled it for the invitation form.
 *
 * `none` is the pair a token already in hand reaches. Neither sends anything,
 * and the message that carried the token was sent while mail *was* configured,
 * so refusing them on today's setting would strand a link that still works.
 */
type PublicAuthRequirement = "mail" | "none"

export const PUBLIC_AUTH_PAGES = {
  "/signup": "mail",
  "/check-email": "mail",
  "/resend-verification": "mail",
  "/recover-password": "mail",
  "/verify-email": "none",
  "/reset-password": "none",
} as const satisfies Record<string, PublicAuthRequirement>

export type PublicAuthPath = keyof typeof PUBLIC_AUTH_PAGES

/**
 * Which public auth page a hash names, or null when it names none of them.
 *
 * Matched on the path alone, so a query string (`?token=…`, `?type=…`) does
 * not have to be parsed here and a trailing one cannot make a known path
 * unrecognizable.
 */
export function publicAuthPath(hash: string): PublicAuthPath | null {
  const path = hash.replace(/^#/, "").split("?")[0] ?? ""
  return path in PUBLIC_AUTH_PAGES ? (path as PublicAuthPath) : null
}

/** Whether a page's flow can start on a deployment with mail in this state. */
export function isPublicAuthPageAvailable(
  path: PublicAuthPath,
  mailReady: boolean,
): boolean {
  return PUBLIC_AUTH_PAGES[path] === "none" || mailReady
}

/** The `token` in `#/verify-email?token=…`, or null if the link is malformed. */
export function tokenFromHash(hash: string): string | null {
  return new URLSearchParams(hash.split("?")[1] ?? "").get("token")
}
