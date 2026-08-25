import type { IconType } from "react-icons"
import { FaGithub } from "react-icons/fa"
import { FcGoogle } from "react-icons/fc"

/**
 * The OAuth providers this dashboard can sign in with, and what to call them.
 *
 * The names are the gateway's (`core/config.py`'s `OAUTH_PROVIDERS`): they are
 * the path segment `/v1/auth/oauth/{provider}/…` takes and the values
 * `/v1/bootstrap`'s `oauth_providers` carries. This table adds only what a
 * server has no business deciding, which is how the provider's name is written
 * on a button.
 *
 * Which of them a deployment actually offers is never decided here. The
 * bootstrap answers that, one entry per provider an operator configured, so a
 * provider nobody set up is absent from the sign-in screen rather than rendered
 * disabled.
 */

/** A provider name the gateway and this dashboard both know. */
export type OAuthProvider = "github" | "google"

/** How each provider writes its own name. */
export const OAUTH_PROVIDER_LABELS: Record<OAuthProvider, string> = {
  github: "GitHub",
  google: "Google",
}

/**
 * Each provider's own mark, for the button that signs in with it.
 *
 * The same two marks `otari-ai/frontend`'s login route uses, from the
 * `react-icons` sets this dashboard already draws its nav from: `Fc` is the
 * full-color Google G, and `Fa` the GitHub logo. Brand marks rather than a
 * generic glyph, because a person scans a sign-in screen for the logo of the
 * account they hold before they read any of the labels.
 */
export const OAUTH_PROVIDER_ICONS: Record<OAuthProvider, IconType> = {
  github: FaGithub,
  google: FcGoogle,
}

/**
 * Whether a string names a provider this dashboard can render.
 *
 * The bootstrap's `oauth_providers` is typed as a plain string list, because
 * the gateway's own vocabulary is open: an overlay may bind an identity adapter
 * for a connection this build never named. So a value is narrowed here rather
 * than assumed, and one this dashboard has no label for is skipped instead of
 * rendered as a button reading `undefined`.
 */
export function isOAuthProvider(value: string): value is OAuthProvider {
  return Object.hasOwn(OAUTH_PROVIDER_LABELS, value)
}

/** The providers from a bootstrap that this dashboard can render, in its order. */
export function renderableOAuthProviders(
  configured: readonly string[],
): OAuthProvider[] {
  return configured.filter(isOAuthProvider)
}

/** How to name a provider in a sentence, falling back to the raw name. */
export function oauthProviderLabel(provider: string): string {
  return isOAuthProvider(provider) ? OAUTH_PROVIDER_LABELS[provider] : provider
}
