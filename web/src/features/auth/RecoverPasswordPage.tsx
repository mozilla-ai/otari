import { Button } from "@heroui/react"
import { useState } from "react"

import { useRequestPasswordReset } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"

import { AuthEmailField } from "./AuthFields"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"

/**
 * `#/recover-password`: ask for a password-reset link.
 *
 * Confirms in place rather than navigating to `#/check-email`, which is where
 * signup and a verification resend both land. The two messages are different
 * ("a reset link" against "a verification link"), and one page that had to say
 * which flow sent it would need a third `?type=`; the platform's own recover
 * page makes the same call for the same reason.
 *
 * Enumeration-safe like its siblings, so the confirmation is written in the
 * conditional and nothing here reads the response.
 */
export function RecoverPasswordPage() {
  const request = useRequestPasswordReset()
  const [email, setEmail] = useState("")

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without canceling the request, so clearing mid-call would drop the
  // `isPending` that `submit` guards on and let a keystroke start a second one.
  const clearError = () => {
    if (request.isPending) {
      return
    }
    request.reset()
  }

  const submit = () => {
    if (email.trim() === "" || request.isPending) {
      return
    }
    request.mutate(email.trim())
  }

  if (request.isSuccess) {
    return (
      <PublicAuthLayout
        title="Check your email"
        description="If that address has a password on this gateway, a reset link is on its way."
        footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
      >
        <p className="text-center text-caption">
          The link is single-use and expires. Requesting another replaces the
          one before it.
        </p>
      </PublicAuthLayout>
    )
  }

  return (
    <PublicAuthLayout
      title="Reset your password"
      description="Enter the address you sign in with and we will mail a link to set a new password."
      footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
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
        />
        <ErrorBanner error={request.error} />
        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={request.isPending}
          isDisabled={email.trim() === ""}
        >
          Send reset link
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
