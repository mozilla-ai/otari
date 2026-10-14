import type { MailedLinkStatus } from "@/client"

import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"

/**
 * What a mailed reset or verification link answers on arrival once the
 * gateway says it can no longer be used, in place of a form whose only
 * outcome is a refusal after a password has been typed.
 *
 * `invalid` covers a link already used and one a newer link replaced, which
 * the gateway cannot tell apart: a used link's record is gone.
 */
export function UnusableLinkPanel({
  status,
  renew,
}: {
  status: Exclude<MailedLinkStatus, "valid">
  /** Where a fresh link is requested, and what that link is called. */
  renew: { to: string; label: string }
}) {
  return (
    <PublicAuthLayout
      title={
        status === "expired"
          ? "This link has expired"
          : "This link no longer works"
      }
      description={
        status === "expired"
          ? "Links like this stop working after a while. Request a new one and open the newest message."
          : "It has already been used, or a newer link replaced it."
      }
      footer={
        <>
          <PublicAuthLink to={renew.to}>{renew.label}</PublicAuthLink>
          <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
        </>
      }
    >
      <p className="text-center text-caption">
        Nothing was changed. If you already finished this step, sign in.
      </p>
    </PublicAuthLayout>
  )
}

/** What a mailed-link page shows while it asks the gateway about its link. */
export function CheckingLinkPanel({ title }: { title: string }) {
  return (
    <PublicAuthLayout title={title}>
      <p className="text-center text-sm text-muted" role="status">
        Checking your link…
      </p>
    </PublicAuthLayout>
  )
}
