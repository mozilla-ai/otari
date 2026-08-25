import { useEffect, useRef } from "react"

import { ApiError } from "@/shared/api/client"
import { useVerifyEmail } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"

/**
 * `#/verify-email?token=…`: the page the mailed verification link lands on.
 *
 * It verifies on arrival rather than behind a button, which is what the link
 * in the message already promised the recipient would happen. Safe to do
 * without a confirmation for the reason the accept-invitation page needs one
 * and this does not: accepting joins an organization, while this only confirms
 * an address the recipient's own mailbox already proved they hold.
 *
 * Verifying on arrival is what makes `useVerifyEmail` a query rather than a
 * mutation: the token is single-use, so the call has to happen exactly once
 * per token, and the query cache is what guarantees that through StrictMode's
 * development-only remount. See the hook for the rest.
 *
 * The one effect here records the outcome and does nothing else. The page has
 * no other moment to record from: the verification runs on arrival rather than
 * behind a control, so there is no handler, and TanStack Query v5 dropped the
 * `onSuccess` a query used to carry. It fires once per mounted page, which the
 * ref is what guarantees: the remount StrictMode performs in development runs
 * effects a second time against the already-settled cache, and a funnel counted
 * twice in development is a funnel nobody can read.
 */
export function VerifyEmailPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const verify = useVerifyEmail(token ?? "")
  const { recordEvent } = useTelemetry()

  const outcome = verify.data ? "verified" : verify.error ? "refused" : null
  const status =
    verify.error instanceof ApiError ? verify.error.status : undefined
  const recorded = useRef(false)
  useEffect(() => {
    if (outcome === null || recorded.current) {
      return
    }
    recorded.current = true
    if (outcome === "verified") {
      recordEvent(TELEMETRY_EVENTS.EMAIL_VERIFICATION_SUCCESS)
      return
    }
    // The gateway's wording is not recorded, only its status: a single-use
    // token that had already been spent and one that never existed answer the
    // same way, and neither sentence is this event's business.
    recordEvent(TELEMETRY_EVENTS.EMAIL_VERIFICATION_FAILED, {
      error_code: "verification_rejected",
      status,
    })
  }, [outcome, status, recordEvent])

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

  if (verify.error) {
    return (
      <PublicAuthLayout
        title="Verification failed"
        description="A verification link is single-use and expires. Request a fresh one and open the newest message."
        footer={
          <>
            <PublicAuthLink to="#/resend-verification">
              Send a new verification link
            </PublicAuthLink>
            <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
          </>
        }
      >
        <ErrorBanner error={verify.error} />
      </PublicAuthLayout>
    )
  }

  return (
    <PublicAuthLayout title="Verify your email">
      <p role="status" className="text-center text-sm text-muted">
        Confirming your address…
      </p>
    </PublicAuthLayout>
  )
}
