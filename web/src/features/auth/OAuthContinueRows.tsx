import { useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { ApiError } from "@/shared/api/client"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthProviderRows } from "./AuthProviderButtons"
import { type OAuthProvider, oauthProviderLabel } from "./oauthProviders"
import { leaveForOAuthProvider } from "./oauthRoundTrip"

/**
 * Provider rows that leave for the provider straight away, for a page that
 * already holds what the round trip has to carry back.
 *
 * `termsAccepted` is that: the person ticked the published terms on this page
 * (or the one before it), so the callback sends the acceptance a sign-in that
 * creates their account needs. Failures to start are recorded the way the
 * sign-in screen records them; the callback page records the outcome.
 */
export function OAuthContinueRows({
  providers,
  termsAccepted,
  returnHash,
  isDisabled = false,
}: {
  providers: readonly OAuthProvider[]
  termsAccepted: boolean
  /** Where a successful sign-in lands; home when not given. */
  returnHash?: string
  isDisabled?: boolean
}) {
  const { recordEvent } = useTelemetry()
  const [pendingProvider, setPendingProvider] = useState<string>()
  // `undefined` while there is none: a failure can be any thrown value.
  const [error, setError] = useState<unknown>()

  const leave = async (provider: OAuthProvider) => {
    if (pendingProvider !== undefined) {
      return
    }
    setError(undefined)
    setPendingProvider(provider)
    try {
      const started = await leaveForOAuthProvider(provider, {
        termsAccepted,
        returnHash,
      })
      if (!started.isOk) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsStatusCode(started.status),
        })
        setError(
          new Error(
            started.message ??
              `${oauthProviderLabel(provider)} sign-in is not available on this gateway.`,
          ),
        )
        setPendingProvider(undefined)
      }
    } catch (caught) {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: provider,
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setError(caught)
      setPendingProvider(undefined)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <AuthProviderRows
        providers={providers}
        layout="stacked"
        verb="Sign in"
        pendingProvider={pendingProvider}
        isDisabled={isDisabled}
        onSelect={(provider) => void leave(provider)}
      />
      {error ? <ErrorBanner error={error} /> : null}
    </div>
  )
}
