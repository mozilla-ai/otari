import type { OAuthProvider } from "./oauthProviders"

/**
 * The hash paths that answer in front of a session, and what each one needs
 * from the deployment before it can work.
 *
 * These are deliberately not files in `src/routes/`, for the reason
 * `AcceptInvitationPage` is not one either: every route there sits behind the
 * session gate `App.tsx`'s `DeploymentRoot` puts in front of the router, and
 * the whole point of these eight is that the visitor holds neither a session nor
 * the master key. `DeploymentRoot` renders them directly, ahead of that gate.
 *
 * The gateway agrees on the spelling rather than being told it: the
 * verification and reset messages it sends carry `/#/verify-email?token=…` and
 * `/#/reset-password?token=…` (`services/tenancy/user_service.py`), so a
 * rename here breaks a link that is already in somebody's inbox. The two OAuth
 * callbacks are the same kind of agreement for a different reason: the gateway
 * redirects `/auth/{provider}/callback` here (`gateway/main.py`), because a
 * redirect URI may not carry a fragment and so cannot be a hash path itself, so
 * renaming one breaks a URI already registered with a provider.
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
 * `oauth` means the flow needs the page's own provider configured on this
 * deployment: the bootstrap's `oauth_providers` lists one entry per provider
 * with a client ID, a secret, and a public base URL to build a redirect URI
 * from, and a provider missing from it can neither issue an authorization URL
 * nor spend a code. The sign-in screen drops the button, and the callback URL a
 * provider or a bookmark still reaches answers with a panel saying so.
 *
 * `none` is the pair a token already in hand reaches. Neither sends anything,
 * and the message that carried the token was sent while mail *was* configured,
 * so refusing them on today's setting would strand a link that still works.
 */
type PublicAuthRequirement = "mail" | "oauth" | "none"

/**
 * The callback path every OAuth provider this dashboard renders must have.
 *
 * Derived from the `OAuthProvider` union rather than written out, so the table
 * below cannot fall behind `oauthProviders.ts`. Forgetting an entry used to be
 * silent and expensive: the sign-in screen renders the button, the gateway
 * redirects to `#/auth/<provider>/callback`, `publicAuthPath` answers `null`,
 * `DeploymentRoot` falls through to the auth gate, and the person lands back on
 * the sign-in screen with their authorization code silently unspent and nothing
 * anywhere saying why.
 *
 * The gateway guards its own half of the same pair at import
 * (`assert set(_PROVIDERS) == set(OAUTH_PROVIDERS)` in `services/oauth_service.py`),
 * and the nav registry guards its capability pair the same way. This is that
 * check, moved from a test into the type: adding a provider without its callback
 * path is a compile error rather than a runtime dead end.
 */
type OAuthCallbackPath = `/auth/${OAuthProvider}/callback`

export const PUBLIC_AUTH_PAGES = {
  "/signup": "mail",
  "/check-email": "mail",
  "/resend-verification": "mail",
  "/recover-password": "mail",
  "/verify-email": "none",
  "/reset-password": "none",
  // Spelled out per provider rather than matched with a parameter, so this
  // table stays the single closed list of what renders in front of a session
  // and `publicAuthPath` keeps answering on an exact lookup. The `satisfies`
  // below is what keeps that spelling-out honest.
  "/auth/google/callback": "oauth",
  "/auth/github/callback": "oauth",
} as const satisfies Record<string, PublicAuthRequirement> &
  Record<OAuthCallbackPath, "oauth">

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
  // `Object.hasOwn`, not `in`: `in` walks the prototype chain, so `#toString`
  // would be answered as a page this table does not hold and the exhaustive
  // switch that renders one would fall off the end into a blank screen.
  return Object.hasOwn(PUBLIC_AUTH_PAGES, path)
    ? (path as PublicAuthPath)
    : null
}

/** Which provider an OAuth callback path finishes, or null for any other page. */
export function oauthCallbackProvider(path: PublicAuthPath): string | null {
  const match = /^\/auth\/([^/]+)\/callback$/.exec(path)
  return match?.[1] ?? null
}

/** What the deployment offers, as the availability check below reads it. */
export interface PublicAuthDeployment {
  mailReady: boolean
  /** The bootstrap's `oauth_providers`, verbatim. */
  oauthProviders: readonly string[]
}

/** Whether a page's flow can run at all on a deployment in this state. */
export function isPublicAuthPageAvailable(
  path: PublicAuthPath,
  deployment: PublicAuthDeployment,
): boolean {
  switch (PUBLIC_AUTH_PAGES[path]) {
    case "none":
      return true
    case "mail":
      return deployment.mailReady
    case "oauth": {
      const provider = oauthCallbackProvider(path)
      return provider !== null && deployment.oauthProviders.includes(provider)
    }
  }
}
