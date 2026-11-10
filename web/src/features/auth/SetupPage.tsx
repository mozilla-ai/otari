import { Button } from "@heroui/react"
import { useEffect, useRef, useState } from "react"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useSetPassword } from "@/shared/api/auth"
import { createSession } from "@/shared/api/client"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import {
  useDeployment,
  useRetireMasterKeySignIn,
} from "@/shared/hooks/useDeployment"

import { useAuth } from "./AuthContext"
import { AuthEmailField, AuthPasswordField } from "./AuthFields"
import { PublicAuthLayout, replacePublicAuthPage } from "./PublicAuthLayout"
import { captureSetupKey, forgetSetupKey } from "./setupKey"
import { rememberSignInNotice } from "./signInNotice"

const ALREADY_SET_UP =
  "This gateway is already set up, so the setup link no longer works. Sign in instead."
const NO_KEY =
  "That setup link is incomplete. Open the full link from the gateway's log, or sign in."
const KEY_REFUSED =
  "The gateway did not accept that setup link. Use the link from its most recent first start, or sign in."

/**
 * `#/setup?key=…`: where the first-run link in the gateway's log lands.
 *
 * The key in the link is the generated master key, so this is the existing
 * claim with nothing typed twice: the key signs in (`POST /v1/auth/session`),
 * and the address and password entered here claim the deployment
 * (`PUT /v1/auth/password`), which retires the key as a sign-in. A link that
 * cannot do that (the gateway is already claimed, or the key is missing or
 * refused) hands the sign-in screen one line saying so.
 */
export function SetupPage({ hash }: { hash: string }) {
  const { sign_in_methods } = useDeployment()
  const { isAuthenticated, login } = useAuth()
  const retireMasterKeySignIn = useRetireMasterKeySignIn()
  const setPassword = useSetPassword()
  const [key] = useState(() => captureSetupKey(hash))
  // Decided once, on arrival: claiming flips `sign_in_methods` under this page.
  const [refusal] = useState(() =>
    !sign_in_methods.includes("master_key")
      ? ALREADY_SET_UP
      : key === null
        ? NO_KEY
        : null,
  )
  const [email, setEmail] = useState("")
  const [password, setPasswordValue] = useState("")
  const [error, setError] = useState<unknown>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  // A refused password leaves the session this page opened in place, so a
  // second attempt goes straight to the password.
  const hasSession = useRef(false)

  useEffect(() => {
    if (refusal === null) {
      return
    }
    // A signed-in caller goes to the dashboard, which needs no explanation.
    if (!isAuthenticated) {
      rememberSignInNotice(refusal)
    }
    replacePublicAuthPage("#/")
  }, [refusal, isAuthenticated])

  if (refusal !== null || key === null) {
    return null
  }

  const problem = newPasswordProblem(password, "")
  const canSubmit = email.trim() !== "" && password !== "" && problem === null

  const submit = async () => {
    if (!canSubmit || isSubmitting) {
      return
    }
    setIsSubmitting(true)
    setError(null)
    try {
      if (!hasSession.current) {
        const session = await createSession({ masterKey: key })
        if (!session.isOk) {
          forgetSetupKey()
          rememberSignInNotice(KEY_REFUSED)
          replacePublicAuthPage("#/")
          return
        }
        hasSession.current = true
      }
      const result = await setPassword.mutateAsync({
        email: email.trim(),
        new_password: password,
      })
      forgetSetupKey()
      if (result.master_key_sign_in_retired) {
        retireMasterKeySignIn()
      }
      login()
      replacePublicAuthPage("#/")
    } catch (caught) {
      setError(caught)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <PublicAuthLayout
      title="Set up Otari"
      description="Choose the email and password you will sign in with from now on."
      isBusy={isSubmitting}
    >
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          void submit()
        }}
      >
        <AuthEmailField
          value={email}
          onChange={(next) => {
            setEmail(next)
            setError(null)
          }}
        />
        <AuthPasswordField
          label="Password"
          value={password}
          onChange={(next) => {
            setPasswordValue(next)
            setError(null)
          }}
          autoComplete="new-password"
          description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
          errorMessage={problem ?? undefined}
          canReveal
        />
        <ErrorBanner error={error} />
        <Button
          type="submit"
          variant="primary"
          fullWidth
          isPending={isSubmitting}
          isDisabled={!canSubmit}
        >
          Finish setup
        </Button>
      </form>
    </PublicAuthLayout>
  )
}
