import { useEffect } from "react"

import { useVerifyEmail } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"

import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { tokenFromHash } from "./publicAuthPaths"

/**
 * `#/verify-email?token=…`: the page the mailed verification link lands on.
 *
 * It verifies on arrival rather than behind a button, which is what the link
 * in the message already promised the recipient would happen. Safe to do
 * without a confirmation for the reason the accept-invitation page needs one
 * and this does not: accepting joins an organization, while this only confirms
 * an address the recipient's own mailbox already proved they hold.
 *
 * Fired once per token. The effect depends on the token alone (`mutate` is
 * stable across renders, and including the mutation object would loop), and
 * `App.tsx` keys this page on the hash, so a second link pasted over the first
 * in an open tab remounts instead of rendering a fresh verification on top of
 * the previous one's success or failure.
 */
export function VerifyEmailPage({ hash }: { hash: string }) {
  const token = tokenFromHash(hash)
  const verify = useVerifyEmail()
  const { mutate } = verify

  // biome-ignore lint/correctness/useExhaustiveDependencies: the token is the only input; `mutate` is stable and adding it would only invite the mutation object back into the list, which is not
  useEffect(() => {
    if (token) {
      mutate(token)
    }
  }, [token])

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

  if (verify.isSuccess) {
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

  if (verify.isError) {
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
