import { Button } from "@heroui/react"
import { useState } from "react"

import { useResetPassword } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"

import { AuthPasswordField } from "./AuthFields"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { tokenFromHash } from "./publicAuthPaths"

/**
 * `#/reset-password?token=…`: the page the mailed reset link lands on.
 *
 * Unlike `#/verify-email`, nothing fires on arrival: the token alone is not
 * the whole act here, and the new password is what completes it. No current
 * password is asked for, which is the entire point of a reset; the token in
 * the link is the proof.
 *
 * No session is minted on success, and the sign-in screen is where the caller
 * goes next. The gateway revokes every other session this identity holds as
 * part of the reset, so a session opened before the account was recovered does
 * not outlive it.
 */
export function ResetPasswordPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const reset = useResetPassword()
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const problem = newPasswordProblem(password, confirmPassword)
  const complete = password !== "" && confirmPassword !== ""
  const canSubmit = token !== null && complete && problem === null

  const submit = () => {
    if (!canSubmit || reset.isPending) {
      return
    }
    reset.mutate({ token, new_password: password })
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
          onChange={setPassword}
          autoComplete="new-password"
          autoFocus
          description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
        />
        <AuthPasswordField
          label="Confirm new password"
          value={confirmPassword}
          onChange={setConfirmPassword}
          autoComplete="new-password"
        />
        {problem ? (
          <p role="alert" className="text-sm text-danger">
            {problem}
          </p>
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
