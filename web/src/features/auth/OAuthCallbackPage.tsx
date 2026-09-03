import { useEffect, useRef, useState } from "react"

import { useAuth } from "@/features/auth/AuthContext"
import { ApiError, completeOAuthSignIn } from "@/shared/api/client"
import { errorMessage } from "@/shared/components/ui"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { oauthProviderLabel } from "./oauthProviders"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"

/**
 * Where the CSRF `state` waits while the browser is away at the provider.
 *
 * `sessionStorage` and not `localStorage`: the value is scoped to the tab that
 * started the flow and dies with it, so a stale state cannot outlive the tab or
 * be read by a second one. The gateway keeps no copy at all, which is the same
 * reason PKCE stays off there (`src/gateway/services/oauth_service.py`): a
 * value minted at authorize time has nowhere server-side to live until the
 * exchange, so the browser holds it and this page is what checks it.
 */
const STATE_KEY = "otari.oauth.state"

/** Remember the state for the provider round trip about to start. */
export function rememberOAuthState(state: string): void {
  window.sessionStorage.setItem(STATE_KEY, state)
}

/** Take the remembered state, clearing it so one value answers one callback. */
function takeOAuthState(): string | null {
  const stored = window.sessionStorage.getItem(STATE_KEY)
  window.sessionStorage.removeItem(STATE_KEY)
  return stored
}

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
 * - The `state` does not match the one this tab stored. That is the CSRF check,
 *   and the whole reason the value made the round trip; a mismatch means this
 *   redirect did not come from a flow this tab started.
 * - There is no code. A callback without one has nothing to spend.
 * - Otherwise the code is posted to the gateway, which exchanges it and sets
 *   the session cookie. On success this signs in exactly the way the password
 *   and passkey screens do, through `useAuth().login()`.
 *
 * Not a form and not retryable: an authorization code is single-use, so a
 * failure sends the person back to the sign-in screen to start again rather
 * than offering a button that would spend a spent code.
 *
 * Both endings leave this page, and both have to: `DeploymentRoot` picks a
 * public auth path off the hash ahead of the auth gate, so this page keeps
 * rendering until the hash changes, whether or not a session was minted. The
 * failure ending offers a link; the success ending changes the hash itself,
 * since there is nothing left for the person to read.
 */
export function OAuthCallbackPage({
  provider,
  hash,
}: {
  provider: string
  hash: string
}) {
  const { login } = useAuth()
  const { recordEvent } = useTelemetry()
  const [failure, setFailure] = useState<string | null>(null)
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
    const expectedState = takeOAuthState()
    const providerError = params.get("error")
    const state = params.get("state")
    const code = params.get("code")

    const refuse = (message: string, errorCode: string) => {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: provider,
        error_code: errorCode,
      })
      setFailure(message)
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
    if (!state || state !== expectedState) {
      refuse(
        `That ${label} sign-in did not start in this tab, so it was not completed. Start again from the sign-in screen.`,
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
        const result = await completeOAuthSignIn(provider, code)
        if (result.ok) {
          recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
            authentication_method: provider,
          })
          login()
          // Signing in is not enough to stop this page rendering, and leaving
          // the hash alone strands the person here. `DeploymentRoot` matches a
          // public auth path *before* the auth gate, so an unchanged hash keeps
          // re-selecting this page, whose effect has already run: the pending
          // panel would sit there with no way out, and a reload would report a
          // state mismatch because the stored state is spent. So this hands the
          // tab back the way `AcceptInvitationPage` does when it is finished.
          goToPublicAuthPage("#/")
          return
        }
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsStatusCode(result.status),
        })
        setFailure(result.message ?? `${label} did not sign you in.`)
      } catch (caught) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsErrorCode(caught),
          status: caught instanceof ApiError ? caught.status : undefined,
        })
        setFailure(errorMessage(caught))
      }
    })()
  }, [provider, hash, login, recordEvent])

  if (failure === null) {
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

  return (
    <PublicAuthLayout
      title="That sign-in did not complete"
      footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
    >
      <p className="text-center text-body">{failure}</p>
    </PublicAuthLayout>
  )
}
