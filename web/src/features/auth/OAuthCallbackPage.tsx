import { useEffect, useRef, useState } from "react"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { useAuth } from "@/features/auth/AuthContext"
import { ApiError, completeOAuthSignIn } from "@/shared/api/client"
import { useDeployment } from "@/shared/hooks/useDeployment"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { OAuthTermsStep } from "./OAuthTermsStep"
import { oauthProviderLabel } from "./oauthProviders"
import { takeOAuthReturnHash, takeOAuthState } from "./oauthRoundTrip"
import {
  PublicAuthLayout,
  PublicAuthLink,
  replacePublicAuthPage,
} from "./PublicAuthLayout"

/**
 * The page a provider's redirect lands on, which finishes an OAuth sign-in.
 *
 * Reached from `/auth/{provider}/callback`, an ordinary path the gateway
 * redirects into this hash route: a redirect URI may not carry a fragment, so
 * the provider cannot be pointed at a hash path directly. The query the
 * provider appended rides along and is read here, once, at mount.
 *
 * Four outcomes, and only the last one talks to the gateway:
 *
 * - The provider says the person declined, or it refused. Nothing was proven,
 *   so nothing is sent.
 * - The `state` does not match the one this tab stored. That is the CSRF check
 *   this tab can make, and the only one that can tell a callback apart by which
 *   tab began it; a mismatch means this redirect did not come from a flow this
 *   tab started. The gateway checks the same value against its own record, so a
 *   state that never came from an `/authorize` here is refused there even when
 *   no tab is involved. A tab that already holds a session goes to the
 *   dashboard instead, since that is Back returning to a finished sign-in.
 * - There is no code. A callback without one has nothing to spend.
 * - Otherwise the code is posted to the gateway, which exchanges it and sets
 *   the session cookie. On success this signs in exactly the way the password
 *   and passkey screens do, through `useAuth().login()`. The acceptance of the
 *   published terms goes with it when the page that started the flow took one.
 *
 * Not a form and not retryable: an authorization code is single-use, so a
 * failure sends the person back to the sign-in screen to start again rather
 * than offering a button that would spend a spent code. The one exception is a
 * sign-in that would create the account on a deployment that publishes terms
 * and arrived without their acceptance (`terms_not_accepted`): that asks for
 * the terms here and starts a fresh round trip with the provider, rather than
 * leaving somebody who did nothing wrong at a dead end.
 *
 * Both endings leave this page, and both have to: `DeploymentRoot` picks a
 * public auth path off the hash ahead of the auth gate, so this page keeps
 * rendering until the hash changes, whether or not a session was minted. The
 * failure ending offers a link; the success ending changes the hash itself,
 * since there is nothing left for the person to read. The one exception is a
 * sign-in that removed the account's earlier password, which the person is told
 * about before they go on, with links rather than a redirect.
 */
export function OAuthCallbackPage({
  provider,
  hash,
}: {
  provider: string
  hash: string
}) {
  const { isAuthenticated, login } = useAuth()
  const { recordEvent } = useTelemetry()
  const { terms_url } = useDeployment()
  // A failure keeps the page the sign-in started from, so trying again from
  // the sign-in screen still ends there.
  const [failure, setFailure] = useState<{
    message: string
    returnHash: string
  }>()
  const [needsTerms, setNeedsTerms] = useState(false)
  // Set when the sign-in removed the account's earlier password, holding the
  // page the notice continues to.
  const [passwordRemoved, setPasswordRemoved] = useState<{
    returnHash: string
  }>()
  // The effect below signs somebody in, so it must run once and not once per
  // render. React's development StrictMode mounts an effect twice on purpose,
  // and the second run would post a code the first already spent, turning every
  // successful development sign-in into a refusal.
  const startedRef = useRef(false)

  useEffect(() => {
    if (startedRef.current) {
      return
    }
    startedRef.current = true

    const label = oauthProviderLabel(provider)
    const params = new URLSearchParams(hash.split("?")[1] ?? "")
    const remembered = takeOAuthState()
    const returnHash = takeOAuthReturnHash()
    const providerError = params.get("error")
    const state = params.get("state")
    const code = params.get("code")

    const fail = (message: string) => setFailure({ message, returnHash })
    const refuse = (message: string, errorCode: string) => {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: provider,
        error_code: errorCode,
      })
      fail(message)
    }

    if (providerError) {
      // The provider's own `error_description` is deliberately not rendered: it
      // is attacker-influenceable text arriving in a URL, and the two outcomes
      // it distinguishes ("you pressed cancel" and "something went wrong") are
      // one instruction to the person reading this page.
      refuse(
        `${label} did not complete the sign-in. You may have declined, or the request expired.`,
        // Not the provider's string: an `error` from a URL is unbounded text,
        // and this value is recorded.
        "provider_error",
      )
      return
    }
    if (!state || state !== remembered.state) {
      // Back from the dashboard after a sign-in lands here with the state
      // already taken. Nothing is spent, and the session this tab holds is the
      // one that callback started, so the dashboard is where it belongs.
      if (isAuthenticated) {
        replacePublicAuthPage("#/")
        return
      }
      refuse(
        `That ${label} sign-in did not start in this tab, or it was already used, so it was not completed. Start again from the sign-in screen.`,
        "invalid_state",
      )
      return
    }
    if (!code) {
      refuse(
        `${label} sent this browser back without an authorization code.`,
        "missing_code",
      )
      return
    }

    void (async () => {
      try {
        const result = await completeOAuthSignIn(
          provider,
          code,
          state,
          remembered.termsAccepted,
        )
        if (result.isOk) {
          recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
            authentication_method: provider,
          })
          login()
          if (result.passwordRemoved) {
            setPasswordRemoved({ returnHash })
            return
          }
          // Signing in is not enough to stop this page rendering, and leaving
          // the hash alone strands the person here. `DeploymentRoot` matches a
          // public auth path *before* the auth gate, so an unchanged hash keeps
          // re-selecting this page, whose effect has already run. So this hands
          // the tab on to the page the sign-in started from, replacing this
          // entry so Back from there skips this page.
          replacePublicAuthPage(returnHash)
          return
        }
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsStatusCode(result.status),
        })
        fail(result.message ?? `${label} did not sign you in.`)
      } catch (caught) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsErrorCode(caught),
          status: caught instanceof ApiError ? caught.status : undefined,
        })
        setNeedsTerms(
          caught instanceof ApiError && caught.code === "terms_not_accepted",
        )
        fail(errorMessage(caught))
      }
    })()
  }, [provider, hash, isAuthenticated, login, recordEvent])

  if (passwordRemoved) {
    return (
      <PublicAuthLayout
        title="You're signed in"
        footer={
          <>
            <PublicAuthLink to="#/account">Set a new password</PublicAuthLink>
            <PublicAuthLink to={passwordRemoved.returnHash}>
              Continue
            </PublicAuthLink>
          </>
        }
      >
        <p className="text-center text-body" role="status">
          Signing in with {oauthProviderLabel(provider)} confirmed your email
          address, so the password set on this account before then was removed:
          whoever chose it may not have been you. To sign in with a password
          again, set a new one from Account settings.
        </p>
      </PublicAuthLayout>
    )
  }

  if (failure === undefined) {
    return (
      <PublicAuthLayout
        title={`Finishing your ${oauthProviderLabel(provider)} sign-in`}
        description="One moment."
      >
        {/* No spinner: the wait is a single request and a spinner that renders
            for 200ms is a flash rather than feedback. */}
        <p className="text-sm text-muted" role="status">
          Checking with the gateway…
        </p>
      </PublicAuthLayout>
    )
  }

  // Only where this tab knows which terms to show; a bootstrap that names none
  // falls back to the gateway's own wording below.
  if (needsTerms && terms_url !== null) {
    return (
      <OAuthTermsStep
        provider={provider}
        termsUrl={terms_url}
        returnHash={failure.returnHash}
      />
    )
  }

  return (
    <PublicAuthLayout
      title="That sign-in did not complete"
      footer={
        <PublicAuthLink to={failure.returnHash}>Back to sign in</PublicAuthLink>
      }
    >
      <p className="text-center text-body">{failure.message}</p>
    </PublicAuthLayout>
  )
}
