/**
 * The page an accept link lands on: `/accept-invitation#token=...`.
 *
 * Deliberately not one of `src/routes/`'s files. Every route there lives
 * behind `App.tsx`'s auth gate (`DeploymentRoot` renders `<Login/>` instead
 * of the router when a session is required and absent), and the whole point
 * of this page is that the recipient holds neither a session nor the master
 * key. `App.tsx` renders this component directly, ahead of that gate, the
 * same way it renders `<Login/>` as a plain component rather than a route.
 *
 * The link reaches the invitee by email or from an admin who copied it, and
 * this page works the same either way. Otari chooses the step and this page
 * renders it. `set_password` asks for a first password and sends it with
 * the accept, so the invitee signs in straight away with no verification
 * email: the link already proves what that email would, and it is the only way
 * in on a deployment that sends no mail. `sign_in` is one button and the
 * sign-in screen. `verify_email` and `use_another_way_in` name what the
 * deployment offers instead.
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
import { rememberSignInReason } from "@/features/auth/signInReason"
import { useProfile } from "@/shared/api/auth"
import { ApiError } from "@/shared/api/client"
import {
  useAcceptInvitation,
  useValidateInvitation,
} from "@/shared/api/organizations"
import { currentRoute } from "@/shared/helpers/appLocation"
import { tokenFromRoute } from "@/shared/helpers/routeParams"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { ClaimForm } from "./ClaimForm"
import { verificationPendingSentence, waysInSentence } from "./waysIn"

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
  // Read once, safely: App.tsx renders this keyed on the route
  // (`<AcceptInvitationPage key={route} />`), so a *different* invitation link
  // opened in the same tab remounts a fresh instance of this component rather
  // than re-rendering this one with a stale token frozen in its initial
  // state. Without that key, a same-type re-render on a route change would keep
  // this state (and its token) exactly as it was.
  const [token] = useState(() => tokenFromRoute(currentRoute()))
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
  // Which button this browser pressed, not a server fact: both land on
  // `sign_in`, and only the copy differs.
  const passwordSubmitted = accept.variables?.password != null
  // Read defensively because a dashboard build that reaches several
  // deployments can be newer than the one answering, and a deployment older
  // than `next_step` sends neither field. The step then falls through to the
  // same branch as `use_another_way_in`, which names a way in rather than
  // ending at a sign-in screen the address cannot pass.
  const previewWaysIn = preview.data?.ways_in ?? []
  const acceptWaysIn = accept.data?.ways_in ?? []

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
          <PublicAuthLink to="/">Back to sign in</PublicAuthLink>
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
          {standing === "invited" && accept.data.next_step === "sign_in" ? (
            <>
              <p className="text-center text-xs text-muted">
                You're already signed in, so there is nothing left to set up.
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => goToPublicAuthPage("/")}
              >
                Go to the dashboard
              </Button>
            </>
          ) : accept.data.next_step === "use_another_way_in" &&
            preview.data?.next_step === "set_password" &&
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
                {accept.data.next_step === "verify_email"
                  ? `Verify ${preview.data?.email ?? "the invited address"} first: ${verificationPendingSentence(acceptWaysIn, deployment.oauth_providers)}`
                  : accept.data.next_step === "sign_in"
                    ? passwordSubmitted
                      ? `Your password is set. Sign in as ${preview.data?.email ?? "the invited address"} to get started.`
                      : "Sign in to get started."
                    : `${invitedEmail ?? "The invited address"} can't sign in yet. To get started, ${waysInSentence(acceptWaysIn, deployment.oauth_providers)}`}
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => {
                  if (invitedEmail) rememberSignInAddress(invitedEmail)
                  const organizationName =
                    accept.data.organization_name ||
                    preview.data?.organization_name
                  rememberSignInReason(
                    organizationName
                      ? `Sign in to join ${organizationName}.`
                      : "Sign in to accept your invitation.",
                  )
                  goToPublicAuthPage("/")
                }}
              >
                Go to sign in
              </Button>
              {accept.data.next_step !== "verify_email" &&
              accept.data.next_step !== "sign_in" &&
              acceptWaysIn.includes("password_reset") ? (
                <PublicAuthLink to="/recover-password">
                  Choose a password
                </PublicAuthLink>
              ) : null}
              {/* Whoever signed up may not be this invitee, who then takes the
                  address back by choosing a password through the mailbox. A
                  reset is always on offer here: this step is answered only
                  where a link reaches the mailbox. */}
              {accept.data.next_step === "verify_email" ? (
                <PublicAuthLink to="/recover-password">
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
          <PublicAuthLink to="/">Back to sign in</PublicAuthLink>
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
          ) : preview.data.next_step === "set_password" ? (
            <ClaimForm
              token={token}
              preview={preview.data}
              accept={accept}
              offersProviderSignIn={offersProviderSignIn}
            />
          ) : (
            <>
              {/* This link may not choose the address's password, either
                  because a signup already chose one or because another
                  organization holds the address too; say which way in exists
                  instead of ending at a sign-in screen it cannot pass. */}
              {preview.data.next_step === "sign_in" ? null : (
                <InfoBanner>
                  This link can't set a password for {preview.data.email}.
                  Accept the invitation, then{" "}
                  {preview.data.next_step === "verify_email"
                    ? verificationPendingSentence(
                        previewWaysIn,
                        deployment.oauth_providers,
                      )
                    : waysInSentence(previewWaysIn, deployment.oauth_providers)}
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
