import { Button } from "@heroui/react"
import { useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { useSignup } from "@/shared/api/auth"
import { ApiError } from "@/shared/api/client"
import { emailFromHash } from "@/shared/helpers/hashParams"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthEmailField, AuthPasswordField, AuthTextField } from "./AuthFields"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"

/**
 * `#/signup`: set a password for an address, claiming or registering it.
 *
 * Which of the two is the deployment's own posture, published as `open_signup`
 * in the bootstrap (otari-ai#2100). Closed, the default, `POST /v1/auth/signup`
 * only ever completes an identity `organization_service` already added or
 * invited by address and creates nothing from nothing, so the copy says so:
 * a page reading as "create an account" would leave somebody who is not on the
 * roster waiting for an email that is never sent. Open, an address nobody has
 * added is registered with an organization of its own, and the same page is a
 * registration form. The form itself is identical either way; only the wording
 * moves, because the request is the same one.
 *
 * **The response says nothing about the address**, under either posture. It is
 * enumeration-safe: unknown, already claimed, and genuinely just claimed all
 * answer the same sentence. So success navigates to `#/check-email`, which is
 * written in the conditional the server's own message uses, and nothing here
 * branches on what came back.
 *
 * The platform's Google and GitHub buttons and its newsletter opt-in are left
 * behind: the first is #651 and the second a hosted marketing concern. Its
 * terms checkbox is here, conditionally, because the objection to it was that a
 * deployment publishing no terms has nothing to link to; `terms_url` answers
 * that, and the server has carried `terms_accepted` and its
 * `terms_accepted_at` column throughout. Where there is a document, accepting
 * it is required, which is what makes the recorded acceptance mean anything.
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
  const { open_signup, terms_url } = useDeployment()
  // Read straight from the prop rather than held in state: `PublicAuthPage` is
  // keyed on the whole hash, so a second link pasted into an open tab remounts
  // this page instead of re-rendering it with the first link's address.
  const invitedEmail = emailFromHash(hash)
  const [email, setEmail] = useState(() => invitedEmail ?? "")
  const [fullName, setFullName] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isTermsAccepted, setIsTermsAccepted] = useState(false)

  const problem = newPasswordProblem(password, confirmPassword)
  const complete =
    email.trim() !== "" &&
    password !== "" &&
    confirmPassword !== "" &&
    (terms_url === null || isTermsAccepted)
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
        // Present only where a document was actually shown. Omitted rather than
        // sent as `false` on a deployment that published none, so the column
        // records an acceptance of something rather than a decision about a
        // checkbox nobody saw. The box below is required, so reaching here with
        // terms published means it was ticked.
        ...(terms_url !== null ? { terms_accepted: true } : {}),
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
      title={open_signup ? "Create your account" : "Claim your account"}
      description={
        open_signup
          ? "Pick an address and a password. You will confirm the address by email before your first sign-in."
          : "Set a password for the address an admin invited or added. You will confirm the address by email before your first sign-in."
      }
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
              : open_signup
                ? "Where the verification link goes, and the address you will sign in with."
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

        {/* Rendered only where there is a document to read, which is what the
            deployment's `terms_url` says. Required rather than optional: an
            acceptance the form would have submitted either way records nothing.
            A plain anchor and not a router `Link`, because the target is an
            address an operator configured and is usually off this origin. */}
        {terms_url !== null ? (
          <Checkbox isSelected={isTermsAccepted} onChange={setIsTermsAccepted}>
            <span className="text-caption">
              I accept the{" "}
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
          </Checkbox>
        ) : null}

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
          {open_signup ? "Create account" : "Claim account"}
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
