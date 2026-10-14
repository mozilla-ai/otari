import { Button } from "@heroui/react"
import { useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useMailedLinkStatus, useVerifyEmail } from "@/shared/api/auth"
import { ApiError } from "@/shared/api/client"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthPasswordField } from "./AuthFields"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { rememberSignInAddress } from "./signInAddress"
import { CheckingLinkPanel, UnusableLinkPanel } from "./UnusableLinkPanel"

/**
 * `#/verify-email?token=…`: the page the mailed verification link lands on.
 *
 * It asks for the password chosen at signup before verifying. Anyone can sign
 * up with an address they do not hold, and the link then reaches the
 * address's real owner: verifying on arrival would let that owner's click
 * activate a stranger's password. The gateway refuses a mismatch without
 * spending the link, so a typo costs nothing. Arrival only asks whether the
 * link can still be used, which does not use it.
 *
 * Only the first signup's password verifies, so someone who signed up twice is
 * told to use the one they chose first. Whoever forgot it, or never signed up,
 * is pointed at a password reset, which is how they take the address back.
 */
export function VerifyEmailPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const verify = useVerifyEmail()
  const linkStatus = useMailedLinkStatus("verify", token ?? "")
  const { recordEvent } = useTelemetry()
  const [password, setPassword] = useState("")

  const clearError = () => {
    if (verify.isPending) {
      return
    }
    verify.reset()
  }

  const submit = () => {
    if (token === null || password === "" || verify.isPending) {
      return
    }
    verify.mutate(
      { token, password },
      {
        onSuccess: (verified) => {
          // So the sign-in screen this page points to opens on the address.
          rememberSignInAddress(verified.email)
          recordEvent(TELEMETRY_EVENTS.EMAIL_VERIFICATION_SUCCESS)
        },
        // The gateway's wording is not recorded, only its status: it tells a
        // spent link from a wrong password for the person, and neither
        // sentence is this event's business.
        onError: (error) => {
          recordEvent(TELEMETRY_EVENTS.EMAIL_VERIFICATION_FAILED, {
            error_code: "verification_rejected",
            status: error instanceof ApiError ? error.status : undefined,
          })
        },
      },
    )
  }

  if (token === null) {
    return (
      <PublicAuthLayout
        title="Verify your email"
        footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
      >
        <ErrorBanner
          error={
            new Error(
              "This link is missing its verification token, so there is nothing to confirm.",
            )
          }
        />
      </PublicAuthLayout>
    )
  }

  if (verify.data) {
    return (
      <PublicAuthLayout
        title="Email verified"
        description={`${verify.data.email} is confirmed. You can sign in now.`}
        footer={<PublicAuthLink to="#/">Go to sign in</PublicAuthLink>}
      >
        <p role="status" className="text-center text-sm text-success">
          Verification complete.
        </p>
      </PublicAuthLayout>
    )
  }

  if (linkStatus.isLoading) {
    return <CheckingLinkPanel title="Verify your email" />
  }

  // A check that failed leaves the form up: submitting still gets the
  // gateway's own answer.
  if (linkStatus.data === "expired" || linkStatus.data === "invalid") {
    return (
      <UnusableLinkPanel
        status={linkStatus.data}
        renew={{
          to: "#/resend-verification",
          label: "Send a new verification link",
        }}
      />
    )
  }

  return (
    <PublicAuthLayout
      isBusy={verify.isPending}
      title="Verify your email"
      description="Enter the password you chose when you signed up to confirm this address. If you signed up more than once, use the password you chose first."
      footer={
        <>
          <PublicAuthLink to="#/recover-password">
            Forgot that password, or didn't sign up? Set a new one
          </PublicAuthLink>
          <PublicAuthLink to="#/resend-verification">
            Send a new verification link
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
          label="Password"
          value={password}
          onChange={(next) => {
            setPassword(next)
            clearError()
          }}
          autoComplete="current-password"
          canReveal
        />
        <ErrorBanner error={verify.error} />
        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={verify.isPending}
          isDisabled={password === ""}
        >
          Verify email
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
