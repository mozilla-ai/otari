import { useVerifyEmail } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { tokenFromHash } from "@/shared/helpers/hashParams"

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
 * development-only remount. See the hook for the rest. There is deliberately
 * no effect here at all.
 */
export function VerifyEmailPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const verify = useVerifyEmail(token ?? "")

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
