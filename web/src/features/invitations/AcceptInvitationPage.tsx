/**
 * The page an accept link lands on: `#/accept-invitation?token=...`.
 *
 * Deliberately not one of `src/routes/`'s files. Every route there lives
 * behind `App.tsx`'s auth gate (`DeploymentRoot` renders `<Login/>` instead
 * of the router when a session is required and absent), and the whole point
 * of this page is that the recipient holds neither a session nor the master
 * key. `App.tsx` renders this component directly, ahead of that gate, the
 * same way it renders `<Login/>` as a plain component rather than a route.
 *
 * The link reaches the invitee by email or from an admin who copied it, and
 * this page works the same either way. When the preview says the invited
 * address has never signed in (`needs_password`), the page asks for a first
 * password and sends it with the accept, so the invitee can sign in straight
 * away with no verification
 * email: the link already proves what that email would, and it is the only
 * way in on a deployment that sends no mail. When the address can already
 * sign in, accepting is one button and the next step is signing in. When it
 * cannot and the link may not set its password either (another organization
 * has it too), the page names the way in the gateway offers instead.
 *
 * Accepting takes no identity, so the membership always goes to the invited
 * address whoever is signed in here. A session for a different address is
 * therefore told who the invitation is for and offered a way to sign out and
 * continue, rather than an accept that reads as if it joined them.
 */

import { Button } from "@heroui/react"
import { useState } from "react"
import type { CallerIdentity } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { useAuth } from "@/features/auth/AuthContext"
import { LoginPageShell } from "@/features/auth/LoginPageShell"
import { OAuthContinueRows } from "@/features/auth/OAuthContinueRows"
import { renderableOAuthProviders } from "@/features/auth/oauthProviders"
import {
  goToPublicAuthPage,
  PublicAuthLink,
} from "@/features/auth/PublicAuthLayout"
import { rememberSignInAddress } from "@/features/auth/signInAddress"
import { useProfile } from "@/shared/api/auth"
import { ApiError } from "@/shared/api/client"
import {
  useAcceptInvitation,
  useValidateInvitation,
} from "@/shared/api/organizations"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { ClaimForm } from "./ClaimForm"
import { waysInSentence } from "./waysIn"

/**
 * Whose session this browser holds, against the invited address.
 *
 * `"invited"` only when the session's address is known and is the invited
 * one. An address that cannot be known (an identity with none, or a gateway
 * older than the profile read, which 404s) counts as someone else's, since the
 * accept would not go to it either. Any other failed read is `"unreadable"`:
 * neither answer can be claimed, so the page holds its actions.
 */
type SessionStanding =
  | "signed-out"
  | "checking"
  | "unreadable"
  | "invited"
  | "someone-else"

function sessionStanding(
  isAuthenticated: boolean,
  profile: { data?: CallerIdentity; error: unknown; isPending: boolean },
  invitedEmail: string | undefined,
): SessionStanding {
  if (!isAuthenticated) {
    return "signed-out"
  }
  if (profile.isPending) {
    return "checking"
  }
  if (
    profile.error &&
    !(profile.error instanceof ApiError && profile.error.status === 404)
  ) {
    return "unreadable"
  }
  const sessionEmail = profile.data?.email
  return sessionEmail &&
    invitedEmail &&
    sessionEmail.toLowerCase() === invitedEmail.toLowerCase()
    ? "invited"
    : "someone-else"
}

export function AcceptInvitationPage() {
  // Read once, safely: App.tsx renders this keyed on the hash
  // (`<AcceptInvitationPage key={hash} />`), so a *different* invitation link
  // opened in the same tab remounts a fresh instance of this component rather
  // than re-rendering this one with a stale token frozen in its initial
  // state. Without that key, a same-type re-render on hashchange would keep
  // this state (and its token) exactly as it was.
  const [token] = useState(() => tokenFromHash(window.location.hash))
  const preview = useValidateInvitation(token ?? "")
  const accept = useAcceptInvitation()
  const { isAuthenticated, logout } = useAuth()
  // The profile rather than the organization context, which 404s for a caller
  // with no organization and so cannot say whose session this is.
  const profile = useProfile(isAuthenticated)
  const sessionEmail = profile.data?.email ?? ""
  const invitedEmail = preview.data?.email
  const standing = sessionStanding(isAuthenticated, profile, invitedEmail)
  const deployment = useDeployment()
  // A provider-verified address resolves a rostered identity that has no
  // password (`adapters/identity_provider_adapter.py`), so where there is a
  // provider, setting a password here is one way in rather than the only one.
  const providers = renderableOAuthProviders(deployment.oauth_providers)
  const offersProviderSignIn = providers.length > 0

  return (
    <LoginPageShell>
      <h1 className="text-display">Organization invitation</h1>

      {token === null ? (
        <>
          <ErrorBanner
            error={
              new Error(
                "This link is missing its invitation token, so there is nothing to accept.",
              )
            }
          />
          {/* A refusal still owes a door: Back onto a spent token lands here. */}
          <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
        </>
      ) : accept.isSuccess ? (
        // Ahead of the preview's own branches, because the preview refuses a
        // token that has been spent: a refetch after this accept (a
        // reconnect is enough) would otherwise replace what happened with
        // "already used" and take the next step away with it.
        <>
          {/* No article before the role: two of the three ("a owner", "a
              admin") read wrong, and the roles are the server's words. */}
          <p className="text-sm text-foreground">
            <strong>{invitedEmail ?? "The invited address"}</strong> is now a
            member of <strong>{accept.data.organization_name}</strong>, with the{" "}
            <strong>{accept.data.role}</strong> role.
          </p>
          {standing === "invited" && !accept.data.password_set ? (
            <>
              <p className="text-center text-xs text-muted">
                You're already signed in, so there is nothing left to set up.
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => goToPublicAuthPage("#/")}
              >
                Go to the dashboard
              </Button>
            </>
          ) : !accept.data.password_set &&
            !accept.data.verification_pending &&
            preview.data?.needs_password &&
            offersProviderSignIn ? (
            // "Sign in with a provider instead": the providers are offered
            // here rather than on the sign-in screen, so the terms accepted
            // above travel with the sign-in that is this address's first.
            <>
              <p className="text-center text-xs text-muted">
                Sign in with your provider to get started.
              </p>
              <OAuthContinueRows
                providers={providers}
                termsAccepted={accept.variables?.terms_accepted === true}
              />
            </>
          ) : (
            <>
              <p className="text-center text-xs text-muted">
                {accept.data.password_set
                  ? `Your password is set. Sign in as ${preview.data?.email ?? "the invited address"} to get started.`
                  : accept.data.verification_pending
                    ? `Verify ${preview.data?.email ?? "the invited address"} first: open the verification link we emailed when you signed up, then sign in.`
                    : accept.data.can_sign_in === false
                      ? `${invitedEmail ?? "The invited address"} has no password yet. To get started, ${waysInSentence(accept.data.ways_in ?? [], deployment.oauth_providers)}`
                      : "Sign in to get started."}
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => {
                  if (invitedEmail) rememberSignInAddress(invitedEmail)
                  goToPublicAuthPage("#/")
                }}
              >
                Go to sign in
              </Button>
              {accept.data.can_sign_in === false &&
              accept.data.ways_in?.includes("password_reset") ? (
                <PublicAuthLink to="#/recover-password">
                  Choose a password
                </PublicAuthLink>
              ) : null}
              {/* Whoever signed up may not be this invitee, who then takes the
                  address back by choosing a password through the mailbox. */}
              {accept.data.verification_pending ? (
                <PublicAuthLink to="#/recover-password">
                  Didn't sign up? Set a password instead
                </PublicAuthLink>
              ) : null}
            </>
          )}
        </>
      ) : preview.isLoading || standing === "checking" ? (
        <p className="text-sm text-muted">Checking your invitation…</p>
      ) : preview.error ? (
        <>
          <ErrorBanner error={preview.error} />
          <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
        </>
      ) : preview.data ? (
        <>
          <p className="text-sm text-foreground">
            <strong>{preview.data.organization_name}</strong> has invited{" "}
            <strong>{preview.data.email}</strong> to join with the{" "}
            <strong>{preview.data.role}</strong> role.
          </p>
          {standing === "unreadable" ? (
            // Either action would be a guess about whose session this is.
            <ErrorBanner error={profile.error} />
          ) : standing === "someone-else" ? (
            <>
              <InfoBanner tone="warning">
                This invitation is for {preview.data.email}, and you're signed
                in as {sessionEmail || "someone else"}. Sign out to accept it as{" "}
                {preview.data.email}.
              </InfoBanner>
              <Button variant="primary" fullWidth onPress={() => void logout()}>
                Sign out and continue
              </Button>
            </>
          ) : preview.data.needs_password ? (
            <ClaimForm
              token={token}
              preview={preview.data}
              accept={accept}
              offersProviderSignIn={offersProviderSignIn}
            />
          ) : (
            <>
              {/* Another organization has this address too, so neither link
                  may choose its password; say which way in exists instead of
                  ending at a sign-in screen it cannot pass. Compared with
                  false because an older gateway sends no such field. */}
              {preview.data.can_sign_in !== false ? null : (
                <InfoBanner>
                  This link can't set a password for {preview.data.email}.
                  Accept the invitation, then{" "}
                  {waysInSentence(
                    preview.data.ways_in ?? [],
                    deployment.oauth_providers,
                  )}
                </InfoBanner>
              )}
              <ErrorBanner error={accept.error} />
              <Button
                variant="primary"
                fullWidth
                isPending={accept.isPending}
                onPress={() => accept.mutate({ token })}
              >
                Accept invitation
              </Button>
            </>
          )}
        </>
      ) : null}
    </LoginPageShell>
  )
}
