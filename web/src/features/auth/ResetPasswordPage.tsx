import { Button } from "@heroui/react"
import { useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { useMailedLinkStatus, useResetPassword } from "@/shared/api/auth"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { AuthPasswordField } from "./AuthFields"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { CheckingLinkPanel, UnusableLinkPanel } from "./UnusableLinkPanel"

/**
 * `#/reset-password?token=…`: the page the mailed reset link lands on.
 *
 * Arrival only asks whether the link can still be used, which does not use it,
 * so a spent or expired link says so before a password is typed. The new
 * password is what completes the reset. No current password is asked for,
 * which is the entire point of a reset; the token in the link is the proof.
 *
 * No session is minted on success, and the sign-in screen is where the caller
 * goes next. The gateway revokes every other session this identity holds as
 * part of the reset, so a session opened before the account was recovered does
 * not outlive it.
 *
 * The same link sets a first password for an account somebody added or
 * invited, so where the deployment publishes terms the page offers them, as
 * on signup. Only an account that has accepted none needs them, and the page
 * cannot tell which this is, so the box is optional: the gateway refuses a
 * first claim without it, and that refusal shows beside the box.
 */
export function ResetPasswordPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const reset = useResetPassword()
  const linkStatus = useMailedLinkStatus("reset", token ?? "")
  const { terms_url } = useDeployment()
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isTermsAccepted, setIsTermsAccepted] = useState(false)

  const problem = newPasswordProblem(password, confirmPassword)
  const isComplete = password !== "" && confirmPassword !== ""
  const canSubmit = token !== null && isComplete && problem === null

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without canceling the request, so clearing mid-call would drop the
  // `isPending` that `submit` guards on and let a keystroke start a second one.
  const clearError = () => {
    if (reset.isPending) {
      return
    }
    reset.reset()
  }

  const submit = () => {
    if (!canSubmit || reset.isPending) {
      return
    }
    reset.mutate({
      token,
      new_password: password,
      ...(terms_url !== null && isTermsAccepted
        ? { terms_accepted: true }
        : {}),
    })
  }

  if (token === null) {
    return (
      <PublicAuthLayout
        title="Reset your password"
        footer={
          <>
            <PublicAuthLink to="#/recover-password">
              Request a new reset link
            </PublicAuthLink>
            <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
          </>
        }
      >
        <ErrorBanner
          error={
            new Error(
              "This link is missing its reset token, so there is no password to set.",
            )
          }
        />
      </PublicAuthLayout>
    )
  }

  if (reset.isSuccess) {
    return (
      <PublicAuthLayout
        title="Password updated"
        description="Sign in with your new password. Any other session this account held has ended."
        footer={<PublicAuthLink to="#/">Go to sign in</PublicAuthLink>}
      >
        <p role="status" className="text-center text-sm text-success">
          Your password has been reset.
        </p>
      </PublicAuthLayout>
    )
  }

  if (linkStatus.isLoading) {
    return <CheckingLinkPanel title="Set a new password" />
  }

  // A check that failed leaves the form up: submitting still gets the
  // gateway's own answer.
  if (linkStatus.data === "expired" || linkStatus.data === "invalid") {
    return (
      <UnusableLinkPanel
        status={linkStatus.data}
        renew={{ to: "#/recover-password", label: "Request a new reset link" }}
      />
    )
  }

  return (
    <PublicAuthLayout
      title="Set a new password"
      description="Choose the password you will sign in with from now on."
      footer={
        <>
          <PublicAuthLink to="#/recover-password">
            Request a new reset link
          </PublicAuthLink>
          <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
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
        <AuthPasswordField
          label="New password"
          value={password}
          onChange={(next) => {
            setPassword(next)
            clearError()
          }}
          autoComplete="new-password"
          description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
          errorMessage={problem ?? undefined}
        />
        <AuthPasswordField
          label="Confirm new password"
          value={confirmPassword}
          onChange={(next) => {
            setConfirmPassword(next)
            clearError()
          }}
          autoComplete="new-password"
        />
        {/* Beside the checkbox rather than inside its label, for the reason
            `SignupPage` gives: nested, the link only ticked the box. */}
        {terms_url !== null ? (
          <div className="flex flex-wrap items-center gap-x-1 text-caption">
            <Checkbox
              isSelected={isTermsAccepted}
              onChange={(next) => {
                setIsTermsAccepted(next)
                clearError()
              }}
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
        <ErrorBanner error={reset.error} />
        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={reset.isPending}
          isDisabled={!canSubmit}
        >
          Set password
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
