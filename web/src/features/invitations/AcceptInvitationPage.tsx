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
 * address has never signed in (`needs_password`), accepting also sets its
 * password, so the invitee can sign in straight away with no verification
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
import type { InvitationPreview } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { useAuth } from "@/features/auth/AuthContext"
import { AuthPasswordField, AuthTextField } from "@/features/auth/AuthFields"
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
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { welcomeGuideHref } from "@/shared/helpers/welcomeGuide"
import { useDeployment } from "@/shared/hooks/useDeployment"

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

// Accepts and sets the first password in one call, for an address that has
// never signed in. Kept apart from `SignupPage`: that form ends in a
// verification email, and this one ends ready to sign in.
function ClaimForm({
  token,
  preview,
  accept,
  offersProviderSignIn,
}: {
  token: string
  preview: InvitationPreview
  accept: ReturnType<typeof useAcceptInvitation>
  offersProviderSignIn: boolean
}) {
  const { terms_url } = useDeployment()
  const [fullName, setFullName] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isTermsAccepted, setIsTermsAccepted] = useState(false)

  const problem = newPasswordProblem(password, confirmPassword)
  const canSubmit =
    password !== "" &&
    confirmPassword !== "" &&
    problem === null &&
    (terms_url === null || isTermsAccepted)

  const submit = () => {
    if (!canSubmit || accept.isPending) return
    accept.mutate({
      token,
      password,
      full_name: fullName.trim() || null,
      // Present only where a document was actually shown, as on signup.
      ...(terms_url !== null ? { terms_accepted: true } : {}),
    })
  }

  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
    >
      <p className="text-xs text-muted">
        Set a password to sign in as {preview.email}.
      </p>
      <AuthTextField
        label="Full name (optional)"
        value={fullName}
        onChange={setFullName}
        autoComplete="name"
      />
      <AuthPasswordField
        label="Password"
        value={password}
        onChange={setPassword}
        autoComplete="new-password"
        description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
        errorMessage={problem ?? undefined}
      />
      <AuthPasswordField
        label="Confirm password"
        value={confirmPassword}
        onChange={setConfirmPassword}
        autoComplete="new-password"
      />
      {/* Beside the checkbox rather than inside its label, for the reason
          `SignupPage` gives: nested, the link only ticked the box. */}
      {terms_url !== null ? (
        <div className="flex flex-wrap items-center gap-x-1 text-caption">
          <Checkbox
            isSelected={isTermsAccepted}
            onChange={setIsTermsAccepted}
            ariaLabel="I accept the terms of service"
          >
            <span className="text-caption">I accept the</span>
          </Checkbox>
          <span>
            <a
              href={terms_url}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-link hover:text-link-hover"
            >
              terms of service
            </a>
            .
          </span>
        </div>
      ) : null}

      <ErrorBanner error={accept.error} />

      <Button
        type="submit"
        variant="primary"
        fullWidth
        isPending={accept.isPending}
        isDisabled={!canSubmit}
      >
        Accept and set password
      </Button>
      {offersProviderSignIn ? (
        <Button
          variant="ghost"
          fullWidth
          isDisabled={accept.isPending}
          onPress={() => accept.mutate({ token })}
        >
          Accept and sign in with a provider instead
        </Button>
      ) : null}
    </form>
  )
}
