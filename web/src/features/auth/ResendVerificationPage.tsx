import { Button } from "@heroui/react"
import { useState } from "react"

import { useResendVerification } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"

import { AuthEmailField } from "./AuthFields"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"

/**
 * `#/resend-verification`: mail a fresh verification link.
 *
 * Reached from the check-email page, from a verification link that had already
 * expired, and from the sign-in screen, because the gateway's own refusal for
 * an unverified identity tells the caller to request a new one and would
 * otherwise be a dead end.
 *
 * Enumeration-safe like signup, so success goes to `#/check-email?type=resend`
 * without reading the response: the server sends the same sentence whether the
 * address is unknown, already verified, or genuinely just re-sent.
 */
export function ResendVerificationPage() {
  const resend = useResendVerification()
  const [email, setEmail] = useState("")

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without cancelling the request, so clearing mid-call would drop the
  // `isPending` that `submit` guards on and let a keystroke start a second one.
  const clearError = () => {
    if (resend.isPending) {
      return
    }
    resend.reset()
  }

  const submit = () => {
    if (email.trim() === "" || resend.isPending) {
      return
    }
    resend.mutate(email.trim(), {
      onSuccess: () => goToPublicAuthPage("#/check-email?type=resend"),
    })
  }

  return (
    <PublicAuthLayout
      title="Send a new verification link"
      description="Enter the address you signed up with and we will mail a fresh link."
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
        <ErrorBanner error={resend.error} />
        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={resend.isPending}
          isDisabled={email.trim() === ""}
        >
          Send link
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
