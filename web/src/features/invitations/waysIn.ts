import type { InvitationPreview } from "@/client"
import { oauthProviderLabel } from "@/features/auth/oauthProviders"

/**
 * How an invited address that cannot sign in yet gets in, as the gateway's
 * `ways_in` says, in one sentence that follows "then" or stands alone.
 *
 * `providers` is the bootstrap's `oauth_providers`, which names them; the
 * gateway's answer says only that one exists.
 */
export function waysInSentence(
  ways: InvitationPreview["ways_in"],
  providers: readonly string[],
): string {
  const names = providers.map(oauthProviderLabel).join(" or ")
  const provider = ways.includes("provider")
    ? `sign in with ${names || "a provider"}`
    : ""
  const reset = ways.includes("password_reset")
    ? 'choose a password with "Forgot your password?"'
    : ""
  if (provider && reset) return `${provider}, or ${reset}.`
  if (provider || reset) return `${provider || reset}.`
  return "ask whoever administers this gateway how to sign in."
}
