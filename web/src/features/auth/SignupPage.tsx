import { Button } from "@heroui/react"
import { useState } from "react"

import { useSignup } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"

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
 * marketing concern; the third has nothing to link to, since a self-hosted
 * deployment publishes no terms, so the request omits `terms_accepted` rather
 * than asserting an acceptance of a document that does not exist.
 */
export function SignupPage() {
  const signup = useSignup()
  const [email, setEmail] = useState("")
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
    signup.mutate(
      {
        email: email.trim(),
        password,
        full_name: fullName.trim() || null,
      },
      { onSuccess: () => goToPublicAuthPage("#/check-email?type=signup") },
    )
  }

  return (
    <PublicAuthLayout
      title="Claim your account"
      description="Set a password for the address your administrator added to this gateway. You will confirm the address by email before your first sign-in."
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
          description="The address your administrator added or invited. Another address has nothing to claim."
        />
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
          <p role="alert" className="text-sm text-danger">
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
