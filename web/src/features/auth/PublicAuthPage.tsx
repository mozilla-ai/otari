import { useDeployment } from "@/shared/hooks/useDeployment"

import { CheckEmailPage } from "./CheckEmailPage"
import { OAuthCallbackPage } from "./OAuthCallbackPage"
import { oauthProviderLabel } from "./oauthProviders"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import type { PublicAuthPath } from "./publicAuthPaths"
import {
  isPublicAuthPageAvailable,
  oauthCallbackProvider,
} from "./publicAuthPaths"
import { RecoverPasswordPage } from "./RecoverPasswordPage"
import { ResendVerificationPage } from "./ResendVerificationPage"
import { ResetPasswordPage } from "./ResetPasswordPage"
import { SignupPage } from "./SignupPage"
import { VerifyEmailPage } from "./VerifyEmailPage"

/**
 * One of the eight pages in front of a session, chosen by hash path.
 *
 * `App.tsx` mounts this ahead of its auth gate and keys it on the whole hash,
 * so following a second emailed link in an open tab remounts rather than
 * re-rendering: the token-reading pages take their token once, at mount, the
 * same hazard `AcceptInvitationPage`'s own key exists for.
 *
 * A page whose flow this deployment cannot run answers with a panel saying so,
 * rather than a form whose only outcome is a 503. The two OAuth callbacks say
 * it about their own provider, because "this gateway sends no mail" is the
 * wrong sentence for a person a provider just redirected here. That is the shell's own
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
  const { mail_ready, oauth_providers } = useDeployment()
  const provider = oauthCallbackProvider(path)

  if (
    !isPublicAuthPageAvailable(path, {
      mailReady: mail_ready,
      oauthProviders: oauth_providers,
    })
  ) {
    return provider === null ? (
      <MailUnavailable />
    ) : (
      <ProviderUnavailable provider={provider} />
    )
  }

  switch (path) {
    case "/signup":
      return <SignupPage hash={hash} />
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
    case "/auth/google/callback":
    case "/auth/github/callback":
      // `provider` is non-null on these two arms by construction: it is parsed
      // from the same path this switch matched. Narrowed with a fallback rather
      // than an assertion, because a `!` here would be a claim the type system
      // cannot check and the fallback renders the same panel either way.
      return <OAuthCallbackPage provider={provider ?? ""} hash={hash} />
  }
}

function ProviderUnavailable({ provider }: { provider: string }) {
  const label = oauthProviderLabel(provider)
  return (
    <PublicAuthLayout
      title="Not available on this gateway"
      description={`This deployment is not configured to sign anyone in with ${label}.`}
      footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
    >
      <p className="text-center text-sm text-muted">
        An operator can turn it on by registering an OAuth client with {label}{" "}
        and giving this gateway its ID and secret. Until then, sign in with the
        credential the sign-in screen offers.
      </p>
    </PublicAuthLayout>
  )
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
