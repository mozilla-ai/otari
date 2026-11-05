import { Button } from "@heroui/react"
import { useState } from "react"
import { ApiError } from "@/shared/api/client"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { leaveForSsoConnection } from "./ssoRoundTrip"

/** What this page records a sign-in attempt under, across every outcome. */
const AUTHENTICATION_METHOD = "sso"

/**
 * The page a direct link to one organization's own connection opens, naming
 * its slug in the URL rather than asking for an address first.
 *
 * Offers one neutral refusal for every slug it cannot start a sign-in with,
 * an unknown one and a disabled connection alike.
 *
 * NOTE: A control starts the round trip rather than an effect on mount. An
 * identity provider a visitor backs out of can return the browser to this
 * same page through its own history, where an effect here would restart the
 * round trip and loop.
 */
export function SsoAuthorizePage({ slug }: { slug: string }) {
  const { recordEvent } = useTelemetry()
  const [isPending, setIsPending] = useState(false)
  const [isRefused, setIsRefused] = useState(false)

  const start = async () => {
    setIsPending(true)
    try {
      const started = await leaveForSsoConnection(slug)
      if (!started.isOk) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: AUTHENTICATION_METHOD,
          error_code: analyticsStatusCode(started.status),
        })
        setIsPending(false)
        setIsRefused(true)
        return
      }
      // Leaving: the tab is already navigating away.
    } catch (caught) {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: AUTHENTICATION_METHOD,
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setIsPending(false)
      setIsRefused(true)
    }
  }

  if (isRefused) {
    return (
      <PublicAuthLayout
        title="That sign-in is not available"
        footer={<PublicAuthLink to="#/">Back to sign in</PublicAuthLink>}
      >
        <p className="text-center text-body">
          Sign in with one of the options on the sign-in screen instead.
        </p>
      </PublicAuthLayout>
    )
  }

  return (
    <PublicAuthLayout
      title="Sign in with your company"
      footer={<PublicAuthLink to="#/">Not your company?</PublicAuthLink>}
    >
      <div className="flex flex-col gap-3">
        <p className="text-center text-body">
          Continue signing in with{" "}
          <span className="font-medium text-foreground">{slug}</span>.
        </p>
        <Button
          type="button"
          variant="primary"
          fullWidth
          isDisabled={isPending}
          onPress={() => void start()}
          className="h-11"
        >
          {isPending ? "Continuing…" : `Continue with ${slug}`}
        </Button>
      </div>
    </PublicAuthLayout>
  )
}
