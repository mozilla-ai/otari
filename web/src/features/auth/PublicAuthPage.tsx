import { useDeployment } from "@/shared/hooks/useDeployment"

import { CheckEmailPage } from "./CheckEmailPage"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import type { PublicAuthPath } from "./publicAuthPaths"
import { isPublicAuthPageAvailable } from "./publicAuthPaths"
import { RecoverPasswordPage } from "./RecoverPasswordPage"
import { ResendVerificationPage } from "./ResendVerificationPage"
import { ResetPasswordPage } from "./ResetPasswordPage"
import { SignupPage } from "./SignupPage"
import { VerifyEmailPage } from "./VerifyEmailPage"

/**
 * One of the six pages in front of a session, chosen by hash path.
 *
 * `App.tsx` mounts this ahead of its auth gate and keys it on the whole hash,
 * so following a second emailed link in an open tab remounts rather than
 * re-rendering: the token-reading pages take their token once, at mount, the
 * same hazard `AcceptInvitationPage`'s own key exists for.
 *
 * A page whose flow this deployment cannot run answers with a panel saying so,
 * rather than a form whose only outcome is a 503. That is the shell's own
 * answer to a gated-off destination (`AppShell`'s "not available here"),
 * applied here because the link to it is hidden on the sign-in screen and a
 * bookmark or an old message can still reach the URL.
 */
export function PublicAuthPage({
  path,
  hash,
}: {
  path: PublicAuthPath
  hash: string
}) {
  const { mail_ready } = useDeployment()

  if (!isPublicAuthPageAvailable(path, mail_ready)) {
    return <MailUnavailable />
  }

  switch (path) {
    case "/signup":
      return <SignupPage />
    case "/check-email":
      return <CheckEmailPage hash={hash} />
    case "/resend-verification":
      return <ResendVerificationPage />
    case "/recover-password":
      return <RecoverPasswordPage />
    case "/verify-email":
      return <VerifyEmailPage hash={hash} />
    case "/reset-password":
      return <ResetPasswordPage hash={hash} />
  }
}

function MailUnavailable() {
  return (
    <PublicAuthLayout
      title="Not available on this gateway"
      description="This flow works by emailing you a link, and this deployment is not configured to send mail."
      footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
    >
      <p className="text-center text-sm text-muted">
        An operator can turn it on by configuring outgoing mail and a public
        base URL for this gateway. Until then, ask whoever administers it to set
        your password for you.
      </p>
    </PublicAuthLayout>
  )
}
