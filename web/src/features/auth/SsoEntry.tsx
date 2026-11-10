import { Button, Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { FiBriefcase } from "react-icons/fi"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { ApiError, lookupSso } from "@/shared/api/client"
import { useBackForwardCacheRestore } from "@/shared/hooks/useBackForwardCacheRestore"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthMethodRow } from "./AuthProviderButtons"
import {
  forgetRememberedCompany,
  type RememberedCompany,
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
  const [remembered, setRemembered] = useState<RememberedCompany | null>(() =>
    readRememberedCompany(),
  )
  const [isOpen, setIsOpen] = useState(false)
  const [rememberedUnavailable, setRememberedUnavailable] = useState(false)
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

  const beginSignIn = async (slug: string, organizationName: string) => {
    setIsPending(true)
    setError(null)
    try {
      const started = await leaveForSsoConnection(slug, {
        organizationName,
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
      // removed its connection since the last sign-in. Dropping it leaves
      // `remembered` null, which renders the address step, rather than the
      // refusal `notFound` renders: that one answers an address, and this
      // visitor typed none.
      if (caught instanceof ApiError && caught.status === 404) {
        forgetRememberedCompany()
        setRemembered(null)
        setRememberedUnavailable(true)
        return
      }
      setError(caught)
    }
  }

  // The round trip leaves while the control is mid-flight, and a restore brings
  // that back, so it is cleared rather than left disabled for good.
  useBackForwardCacheRestore(() => setIsPending(false))

  const submitAddress = async () => {
    const address = email.trim()
    if (!address) {
      setError(new Error("Enter your work email."))
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
      await beginSignIn(looked.slug, looked.organizationName)
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
            <span className="font-medium text-foreground">
              {remembered.name}
            </span>
            .
          </p>
          <Button
            type="button"
            variant="primary"
            fullWidth
            isDisabled={isDisabled || isPending}
            onPress={() => void beginSignIn(remembered.slug, remembered.name)}
            className="h-11"
          >
            {isPending ? "Continuing…" : `Continue with ${remembered.name}`}
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
          {rememberedUnavailable ? (
            <p className="text-body">
              That company&rsquo;s sign-in is no longer available. Enter your
              work email.
            </p>
          ) : null}
          <TextField
            value={email}
            onChange={(next) => {
              setEmail(next)
              setError(null)
            }}
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
          {/* Never disabled for an empty box, as Login's own submit is not:
              a disabled primary button sits at 1.95:1 on the brand tint, so an
              empty address is refused on submit instead. */}
          <Button
            type="submit"
            variant="primary"
            fullWidth
            isDisabled={isDisabled || isPending}
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
