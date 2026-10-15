import { useState } from "react"

import { OAuthContinueRows } from "./OAuthContinueRows"
import { isOAuthProvider, oauthProviderLabel } from "./oauthProviders"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { TermsAcceptanceField } from "./TermsAcceptanceField"

/**
 * What the callback page shows when the gateway refused a sign-in that would
 * set up the person's account here because the published terms were not
 * accepted (`terms_not_accepted`).
 *
 * Reached from a provider button that asks for no terms, the sign-in screen's
 * above all: signing in there creates an account on an open deployment, and is
 * the first way in for somebody an organization added. The refusal kept
 * nothing, so accepting the terms here and going back to the provider is a
 * complete second attempt rather than a resumption.
 */
export function OAuthTermsStep({
  provider,
  termsUrl,
  returnHash,
}: {
  provider: string
  termsUrl: string
  /** The page the refused sign-in started from, which the second attempt keeps. */
  returnHash: string
}) {
  const [isAccepted, setIsAccepted] = useState(false)
  const label = oauthProviderLabel(provider)

  return (
    <PublicAuthLayout
      title="Accept the terms to continue"
      description={`Signing in with ${label} sets up your account on this gateway, which needs its terms of service accepted first.`}
      footer={<PublicAuthLink to={returnHash}>Back to sign in</PublicAuthLink>}
    >
      <div className="flex flex-col gap-3">
        <TermsAcceptanceField
          termsUrl={termsUrl}
          isAccepted={isAccepted}
          onChange={setIsAccepted}
        />
        {isOAuthProvider(provider) ? (
          <OAuthContinueRows
            providers={[provider]}
            termsAccepted
            returnHash={returnHash}
            isDisabled={!isAccepted}
          />
        ) : null}
      </div>
    </PublicAuthLayout>
  )
}
