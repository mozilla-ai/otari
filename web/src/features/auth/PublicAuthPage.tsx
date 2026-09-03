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
      <MailUnavailable offersProviderSignIn={oauth_providers.length > 0} />
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
      <p className="text-sm text-muted">
        An operator can turn it on by registering an OAuth client with {label}{" "}
        and giving this gateway its ID and secret. Until then, sign in with the
        credential the sign-in screen offers.
      </p>
    </PublicAuthLayout>
  )
}

/**
 * `offersProviderSignIn` is what stops this panel from being wrong on a
 * deployment that configures OAuth but no mail, which is an ordinary shape: a
 * provider client is easier to register than an SMTP host. A provider-verified
 * address resolves a rostered identity that holds no password at all
 * (`adapters/identity_provider_adapter.py`), so there the visitor is not stuck
 * and should be sent to the button that works.
 *
 * What it never says any more is "ask an administrator to set your password":
 * `PUT /v1/auth/password` only ever acts on the caller's own identity, so no
 * endpoint on this deployment can do that for someone else.
 */
function MailUnavailable({
  offersProviderSignIn,
}: {
  offersProviderSignIn: boolean
}) {
  return (
    <PublicAuthLayout
      title="Not available on this gateway"
      description="This flow works by emailing you a link, and this deployment is not configured to send mail."
      footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
    >
      <p className="text-sm text-muted">
        {offersProviderSignIn
          ? "Sign in with one of the providers on the sign-in screen instead, which needs no mail. An operator can turn this flow on by configuring outgoing mail and a public base URL for this gateway."
          : "An operator can turn it on by configuring outgoing mail and a public base URL for this gateway."}
      </p>
    </PublicAuthLayout>
  )
}
