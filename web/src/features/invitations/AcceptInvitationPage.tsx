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
 * sign in, accepting is one button and the next step is signing in.
 *
 * A session is treated as this browser's state rather than proof of who is
 * reading: accepting takes no identity, so a signed-in visitor still gets the
 * page the link was meant for.
 */

import { Button, Link } from "@heroui/react"
import { useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useAuth } from "@/features/auth/AuthContext"
import { LoginPageShell } from "@/features/auth/LoginPageShell"
import {
  goToPublicAuthPage,
  PublicAuthLink,
} from "@/features/auth/PublicAuthLayout"
import {
  useAcceptInvitation,
  useValidateInvitation,
} from "@/shared/api/organizations"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import { welcomeGuideHref } from "@/shared/helpers/welcomeGuide"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { ClaimForm } from "./ClaimForm"

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
  const { isAuthenticated } = useAuth()
  const deployment = useDeployment()
  // Absent on a hosted deployment, which serves no such page; see
  // `welcomeGuideHref`. The rule above it goes with it, since a card ending in
  // a bordered empty row reads as something that failed to load.
  const welcomeHref = welcomeGuideHref(deployment)
  // A provider-verified address resolves a rostered identity that has no
  // password (`adapters/identity_provider_adapter.py`), so where there is a
  // provider, setting a password here is one way in rather than the only one.
  const offersProviderSignIn = deployment.oauth_providers.length > 0

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
          {/* A refusal still owes a door: Back onto a spent token lands here,
              and a card whose only other link is the welcome guide strands
              whoever reads it. */}
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
            You're now a member of{" "}
            <strong>{accept.data.organization_name}</strong>, with the{" "}
            <strong>{accept.data.role}</strong> role.
          </p>
          {isAuthenticated && !accept.data.password_set ? (
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
          ) : (
            <>
              <p className="text-center text-xs text-muted">
                {accept.data.password_set
                  ? `Your password is set. Sign in as ${preview.data?.email ?? "the invited address"} to get started.`
                  : preview.data?.needs_password && offersProviderSignIn
                    ? "Sign in with one of the providers on the sign-in screen to get started."
                    : "Sign in to get started."}
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => goToPublicAuthPage("#/")}
              >
                Go to sign in
              </Button>
            </>
          )}
        </>
      ) : preview.isLoading ? (
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
          {preview.data.needs_password ? (
            <ClaimForm
              token={token}
              preview={preview.data}
              accept={accept}
              offersProviderSignIn={offersProviderSignIn}
            />
          ) : (
            <>
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

      {welcomeHref ? (
        <div className="flex border-t border-border pt-2">
          <Link
            href={welcomeHref}
            className="inline-flex min-h-11 items-center text-sm font-medium text-link hover:text-link-hover"
          >
            Open the welcome guide
          </Link>
        </div>
      ) : null}
    </LoginPageShell>
  )
}
