import { Button, Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { FiBriefcase } from "react-icons/fi"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { ApiError, lookupSso } from "@/shared/api/client"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthMethodRow } from "./AuthProviderButtons"
import {
  forgetRememberedCompany,
  readRememberedCompany,
} from "./rememberedCompany"
import { leaveForSsoConnection } from "./ssoRoundTrip"

/** What the row records a sign-in attempt under, across every outcome. */
const AUTHENTICATION_METHOD = "sso"

/**
 * The row that signs in through an organization's own connection, closed by
 * default like the credential row above it.
 *
 * Offered on every deployment, because no deployment-wide setting could gate
 * it: whether an address's organization has a connection is answered per
 * address rather than published in the bootstrap. A browser that already signed in this way once skips straight to
 * continuing with that organization; any other browser is asked for an
 * address to resolve one from.
 */
export function SsoEntry({ isDisabled = false }: { isDisabled?: boolean }) {
  const { recordEvent } = useTelemetry()
  const [remembered, setRemembered] = useState<string | null>(() =>
    readRememberedCompany(),
  )
  const [isOpen, setIsOpen] = useState(false)
  const [email, setEmail] = useState("")
  const [notFound, setNotFound] = useState(false)
  const [isPending, setIsPending] = useState(false)
  const [error, setError] = useState<unknown>(null)

  const close = () => {
    setIsOpen(false)
    setNotFound(false)
    setError(null)
    setEmail("")
  }

  const beginSignIn = async (slug: string) => {
    setIsPending(true)
    setError(null)
    try {
      const started = await leaveForSsoConnection(slug, {
        returnHash: window.location.hash,
      })
      if (!started.isOk) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: AUTHENTICATION_METHOD,
          error_code: analyticsStatusCode(started.status),
        })
        setIsPending(false)
        setError(
          new Error(
            started.message ?? "That company's sign-in is not available.",
          ),
        )
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
      // A remembered slug can go stale: the organization may have disabled or
      // removed its connection since the last sign-in. Falling back to the
      // address step rather than showing a bare error keeps this from being a
      // dead end.
      if (caught instanceof ApiError && caught.status === 404) {
        forgetRememberedCompany()
        setRemembered(null)
        setNotFound(true)
        return
      }
      setError(caught)
    }
  }

  const submitAddress = async () => {
    const address = email.trim()
    if (!address) {
      return
    }
    setIsPending(true)
    setError(null)
    setNotFound(false)
    try {
      const looked = await lookupSso(address)
      if (!looked.isOk) {
        setIsPending(false)
        setError(new Error(looked.message ?? "Could not look up that address."))
        return
      }
      if (!looked.found) {
        setIsPending(false)
        setNotFound(true)
        return
      }
      await beginSignIn(looked.slug)
    } catch (caught) {
      setIsPending(false)
      setError(caught)
    }
  }

  if (!isOpen) {
    return (
      <AuthMethodRow
        icon={FiBriefcase}
        isDisabled={isDisabled}
        onPress={() => setIsOpen(true)}
      >
        Sign in with your company (SSO)
      </AuthMethodRow>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {remembered !== null ? (
        <div className="flex flex-col gap-3">
          <p className="text-body">
            Continue signing in with{" "}
            <span className="font-medium text-foreground">{remembered}</span>.
          </p>
          <Button
            type="button"
            variant="primary"
            fullWidth
            isDisabled={isDisabled || isPending}
            onPress={() => void beginSignIn(remembered)}
            className="h-11"
          >
            {isPending ? "Continuing…" : `Continue with ${remembered}`}
          </Button>
          <Button
            type="button"
            variant="ghost"
            fullWidth
            isDisabled={isDisabled || isPending}
            onPress={() => {
              forgetRememberedCompany()
              setRemembered(null)
            }}
            className="h-11"
          >
            Not your company?
          </Button>
        </div>
      ) : notFound ? (
        <div className="flex flex-col gap-3">
          <p className="text-body">
            We could not find a company sign-in for that address.
          </p>
          <Button
            type="button"
            variant="ghost"
            fullWidth
            onPress={close}
            className="h-11"
          >
            Back to sign in
          </Button>
        </div>
      ) : (
        <form
          className="flex flex-col gap-3"
          noValidate
          onSubmit={(event) => {
            event.preventDefault()
            void submitAddress()
          }}
        >
          <TextField
            value={email}
            onChange={setEmail}
            type="email"
            isRequired
            validationBehavior="aria"
            className="flex flex-col gap-2"
          >
            <Label>Work email</Label>
            <Input
              placeholder="you@example.com"
              autoComplete="email"
              className="h-10 text-base"
            />
          </TextField>
          <Button
            type="submit"
            variant="primary"
            fullWidth
            isDisabled={isDisabled || isPending || email.trim() === ""}
            className="h-11"
          >
            {isPending ? "Looking up…" : "Continue"}
          </Button>
        </form>
      )}
      {error ? <ErrorBanner error={error} /> : null}
    </div>
  )
}
