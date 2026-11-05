import { Button } from "@heroui/react"
import { useEffect, useRef, useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { useAuth } from "@/features/auth/AuthContext"
import { ApiError, completeSsoSignIn } from "@/shared/api/client"
import { useDeployment } from "@/shared/hooks/useDeployment"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import {
  PublicAuthLayout,
  PublicAuthLink,
  replacePublicAuthPage,
} from "./PublicAuthLayout"
import { rememberCompany } from "./rememberedCompany"
import {
  leaveForSsoConnection,
  takeSsoReturnHash,
  takeSsoState,
} from "./ssoRoundTrip"
import { TermsAcceptanceField } from "./TermsAcceptanceField"

/** What this page records a sign-in attempt under, across every outcome. */
const AUTHENTICATION_METHOD = "sso"

/**
 * The page an organization's own identity provider redirects to, which
 * finishes a sign-in through its connection.
 *
 * The callback path is fixed for every organization alike, so the
 * organization is not named in the URL. A tab holding no sign-in of its own
 * fails the ordinary state mismatch rather than resolving a wrong
 * organization.
 *
 * A refused sign-in is not retryable: an authorization code is single-use, so
 * it sends the person back to the sign-in screen to start again. The one
 * exception is a sign-in that would create the account on a deployment
 * publishing terms and arrived without their acceptance, which asks for them
 * here and starts a fresh round trip through the same connection.
 */
export function SsoCallbackPage({ hash }: { hash: string }) {
  const { isAuthenticated, login } = useAuth()
  const { recordEvent } = useTelemetry()
  const { terms_url } = useDeployment()
  const [slug, setSlug] = useState<string>()
  const [failure, setFailure] = useState<{
    message: string
    returnHash: string
  }>()
  const [needsTerms, setNeedsTerms] = useState(false)
  const [passwordRemoved, setPasswordRemoved] = useState<{
    returnHash: string
  }>()
  // React's development StrictMode mounts an effect twice on purpose, and a
  // code is single-use, so the second run would post a code the first already
  // spent, turning every successful development sign-in into a refusal.
  const startedRef = useRef(false)

  useEffect(() => {
    if (startedRef.current) {
      return
    }
    startedRef.current = true

    const params = new URLSearchParams(hash.split("?")[1] ?? "")
    const stored = takeSsoState()
    const returnHash = takeSsoReturnHash()
    setSlug(stored.slug)
    const providerError = params.get("error")
    const state = params.get("state")
    const code = params.get("code")
    const iss = params.get("iss")

    const fail = (message: string) => setFailure({ message, returnHash })
    const refuse = (message: string, errorCode: string) => {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: AUTHENTICATION_METHOD,
        error_code: errorCode,
      })
      fail(message)
    }

    if (providerError) {
      refuse(
        "Your company's sign-in did not complete. You may have declined, or the request expired.",
        "provider_error",
      )
      return
    }
    if (!state || state !== stored.state) {
      // Back from the dashboard after a sign-in lands here with the state
      // already taken. Nothing is spent, and the session this tab holds is the
      // one that callback started, so the dashboard is where it belongs.
      if (isAuthenticated) {
        replacePublicAuthPage("#/")
        return
      }
      refuse(
        "That sign-in did not start in this tab, or it was already used, so it was not completed. Start again from the sign-in screen.",
        "invalid_state",
      )
      return
    }
    if (!code) {
      refuse(
        "Your company's identity provider sent this browser back without an authorization code.",
        "missing_code",
      )
      return
    }

    void (async () => {
      try {
        const result = await completeSsoSignIn(
          code,
          state,
          stored.termsAccepted,
          iss,
        )
        if (result.isOk) {
          recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
            authentication_method: AUTHENTICATION_METHOD,
          })
          rememberCompany(stored.slug)
          login()
          if (result.passwordRemoved) {
            setPasswordRemoved({ returnHash })
            return
          }
          replacePublicAuthPage(returnHash)
          return
        }
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: AUTHENTICATION_METHOD,
          error_code: analyticsStatusCode(result.status),
        })
        fail(result.message ?? "Your company's sign-in did not complete.")
      } catch (caught) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: AUTHENTICATION_METHOD,
          error_code: analyticsErrorCode(caught),
          status: caught instanceof ApiError ? caught.status : undefined,
        })
        setNeedsTerms(
          caught instanceof ApiError && caught.code === "terms_not_accepted",
        )
        fail(errorMessage(caught))
      }
    })()
  }, [hash, isAuthenticated, login, recordEvent])

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
          Signing in through your company's connection confirmed your email
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
        title={
          slug
            ? `Finishing your sign-in with ${slug}`
            : "Finishing your company sign-in"
        }
        description="One moment."
      >
        <p className="text-sm text-muted" role="status">
          Checking your sign-in…
        </p>
      </PublicAuthLayout>
    )
  }

  // Only where this tab knows which terms to show and which connection to
  // retry; a bootstrap that names no terms falls back to the ordinary
  // refusal wording instead.
  if (needsTerms && terms_url !== null && slug !== undefined) {
    return (
      <SsoTermsStep
        slug={slug}
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

/**
 * What this page shows when a sign-in that would set up the person's account
 * here was refused because the published terms were not accepted.
 *
 * The refusal kept nothing, so accepting the terms here and going back to the
 * same connection is a complete second attempt rather than a resumption.
 */
function SsoTermsStep({
  slug,
  termsUrl,
  returnHash,
}: {
  slug: string
  termsUrl: string
  /** The page the refused sign-in started from, which the second attempt keeps. */
  returnHash: string
}) {
  const { recordEvent } = useTelemetry()
  const [isAccepted, setIsAccepted] = useState(false)
  const [isPending, setIsPending] = useState(false)
  const [error, setError] = useState<unknown>()

  const retry = async () => {
    if (isPending) {
      return
    }
    setError(undefined)
    setIsPending(true)
    try {
      const started = await leaveForSsoConnection(slug, {
        termsAccepted: true,
        returnHash,
      })
      if (!started.isOk) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: AUTHENTICATION_METHOD,
          error_code: analyticsStatusCode(started.status),
        })
        setError(
          new Error(
            started.message ?? "That company's sign-in is not available.",
          ),
        )
        setIsPending(false)
      }
    } catch (caught) {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: AUTHENTICATION_METHOD,
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setError(caught)
      setIsPending(false)
    }
  }

  return (
    <PublicAuthLayout
      title="Accept the terms to continue"
      description="Signing in through your company's connection sets up your account here, which needs its terms of service accepted first."
      footer={<PublicAuthLink to={returnHash}>Back to sign in</PublicAuthLink>}
    >
      <div className="flex flex-col gap-3">
        <TermsAcceptanceField
          termsUrl={termsUrl}
          isAccepted={isAccepted}
          onChange={setIsAccepted}
        />
        <Button
          type="button"
          variant="primary"
          fullWidth
          isDisabled={!isAccepted || isPending}
          onPress={() => void retry()}
          className="h-11"
        >
          {isPending ? "Redirecting…" : "Continue signing in"}
        </Button>
        {error ? <ErrorBanner error={error} /> : null}
      </div>
    </PublicAuthLayout>
  )
}
