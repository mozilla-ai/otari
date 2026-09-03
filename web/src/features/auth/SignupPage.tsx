import { Button } from "@heroui/react"
import { useState } from "react"

import { ApiError } from "@/shared/api/client"
import { useSignup } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { emailFromHash } from "@/shared/helpers/hashParams"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthEmailField, AuthPasswordField, AuthTextField } from "./AuthFields"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"

/**
 * `#/signup`: claim the identity an admin already put on this deployment's
 * roster, by giving it a password.
 *
 * Two departures from the platform's own signup page, both of them the
 * gateway's behavior rather than a design preference:
 *
 * - **It is not registration.** `POST /v1/auth/signup` only ever completes an
 *   identity `organization_service` already added or invited by address; it
 *   creates nothing from nothing. The copy says so, because a page that reads
 *   as "create an account" would leave someone who is not on the roster
 *   waiting for an email that is never sent.
 * - **The response says nothing about the address.** It is enumeration-safe:
 *   unknown, already claimed, and genuinely just claimed all answer the same
 *   sentence. So success navigates to `#/check-email`, which is written in the
 *   conditional the server's own message uses, and nothing here branches on
 *   what came back.
 *
 * The platform's Google and GitHub buttons, its newsletter opt-in, and its
 * terms checkbox are all left behind. The first two are #651 and a hosted
 * marketing concern; the third has nothing to link to on a deployment that
 * published no terms, so the request omits `terms_accepted` rather than
 * asserting an acceptance of a document that does not exist. `terms_url` makes
 * that conditional rather than always true, and the server has carried
 * `terms_accepted` and its `terms_accepted_at` column throughout, so offering
 * the checkbox where there is a document to accept is unwired rather than
 * impossible.
 *
 * `?email=…` prefills the address, which is how the accept-invitation page
 * hands an invitee straight here (otari#835). It arrives read-only, because
 * the invitation is bound to that address and claiming a different one would
 * answer with the same enumeration-safe sentence while doing nothing at all,
 * which is the failure this whole handoff exists to remove. The footer offers
 * the plain page for anyone who does need another address. Not a credential
 * and not treated as one: the token that proved anything was spent on the
 * accept, and `POST /v1/auth/signup` checks this address against the roster
 * itself.
 */
export function SignupPage({ hash }: { hash: string }) {
  const signup = useSignup()
  const { recordEvent } = useTelemetry()
  // Read straight from the prop rather than held in state: `PublicAuthPage` is
  // keyed on the whole hash, so a second link pasted into an open tab remounts
  // this page instead of re-rendering it with the first link's address.
  const invitedEmail = emailFromHash(hash)
  const [email, setEmail] = useState(() => invitedEmail ?? "")
  const [fullName, setFullName] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const problem = newPasswordProblem(password, confirmPassword)
  const complete =
    email.trim() !== "" && password !== "" && confirmPassword !== ""
  const canSubmit = complete && problem === null

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without canceling the request, so clearing mid-call would drop the
  // `isPending` that `submit` guards on and let a keystroke start a second one.
  const clearError = () => {
    if (signup.isPending) {
      return
    }
    signup.reset()
  }

  const submit = () => {
    if (!canSubmit || signup.isPending) {
      return
    }
    // The attempt, recorded before the request rather than alongside its
    // outcome, so a claim that never comes back is still a step in the funnel.
    // `password` is the only method this form offers: the platform's Google and
    // GitHub buttons wait on otari#651.
    recordEvent(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: "password",
    })
    signup.mutate(
      {
        email: email.trim(),
        password,
        full_name: fullName.trim() || null,
      },
      {
        onSuccess: () => {
          // Always verification-bound, and not a reading of the response: this
          // endpoint is enumeration-safe and says nothing about the address, so
          // the page navigates to check-email whatever came back.
          recordEvent(TELEMETRY_EVENTS.SIGNUP_SUCCESS, {
            authentication_method: "password",
            requires_verification: true,
          })
          goToPublicAuthPage("#/check-email?type=signup")
        },
        onError: (error) => {
          recordEvent(TELEMETRY_EVENTS.SIGNUP_FAILED, {
            authentication_method: "password",
            status: error instanceof ApiError ? error.status : undefined,
          })
        },
      },
    )
  }

  return (
    <PublicAuthLayout
      title="Claim your account"
      description="Set a password for the address an admin invited or added. You will confirm the address by email before your first sign-in."
      footer={
        <>
          <PublicAuthLink to="#/">
            Already have a password? Sign in
          </PublicAuthLink>
          <PublicAuthLink to="#/resend-verification">
            Need a new verification link?
          </PublicAuthLink>
        </>
      }
    >
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <AuthEmailField
          value={email}
          onChange={(next) => {
            setEmail(next)
            clearError()
          }}
          isReadOnly={invitedEmail !== null}
          description={
            invitedEmail
              ? "The address your invitation was sent to, which is the one it can claim."
              : "The address an admin added or invited. Another address has nothing to claim."
          }
        />
        {/* Directly under the field rather than in the footer: this is the way
            out of a prefill that is wrong for whoever is reading, and someone
            who has just tried to type over a read-only field is looking here,
            not three rows below the submit button. */}
        {invitedEmail ? (
          <PublicAuthLink to="#/signup">
            Claim a different address instead
          </PublicAuthLink>
        ) : null}
        {/* Optional, and the server treats it as such: it fills the name in
            only if the identity does not already have one, so leaving it blank
            never clears what an admin typed. */}
        <AuthTextField
          label="Full name (optional)"
          value={fullName}
          onChange={(next) => {
            setFullName(next)
            clearError()
          }}
          autoComplete="name"
        />
        <AuthPasswordField
          label="Password"
          value={password}
          onChange={(next) => {
            setPassword(next)
            clearError()
          }}
          autoComplete="new-password"
          description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
        />
        <AuthPasswordField
          label="Confirm password"
          value={confirmPassword}
          onChange={(next) => {
            setConfirmPassword(next)
            clearError()
          }}
          autoComplete="new-password"
        />

        {problem ? (
          <p role="alert" className="text-caption text-danger">
            {problem}
          </p>
        ) : null}
        <ErrorBanner error={signup.error} />

        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={signup.isPending}
          isDisabled={!canSubmit}
        >
          Claim account
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
