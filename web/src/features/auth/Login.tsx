import { Button, Card, Input, Label, Link, TextField } from "@heroui/react"
import { useState } from "react"
import { useAuth } from "@/features/auth/AuthContext"
import type { SignInCredential } from "@/shared/api/client"
import { createSession } from "@/shared/api/client"
import { ErrorBanner } from "@/shared/components/ui"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { PublicAuthLink } from "./PublicAuthLayout"

/**
 * The sign-in screen, rendering whichever credential this deployment accepts.
 *
 * A standalone gateway takes the master key until an operator claims it by
 * setting a password, and email and password from then on
 * (mozilla-ai/otari-ai#1716). The gateway publishes which applies in the
 * bootstrap's `sign_in_methods`, so the form is chosen from that rather than
 * from a refusal: presenting the master-key box to a claimed deployment would
 * ask for the one credential its sign-in endpoint no longer takes.
 *
 * Below the form sit the ways in that are not a credential: claiming a rostered
 * identity, recovering a forgotten password, and asking for a fresh
 * verification link (otari#650, the pages in `publicAuthPaths.ts`). All three
 * start by sending a message, so all three are hidden on a deployment whose
 * bootstrap reports `mail_ready: false` rather than offered and then refused
 * with a 503, the way otari#648 already settled it for the invitation form.
 * Recovery is hidden on an unclaimed deployment as well: `master_key` is
 * published exactly while no identity holds a password, so there is nothing
 * yet for a reset link to reset, and the way back in is the master key against
 * `PUT /v1/auth/password` (see docs/access-control.md).
 *
 * The OAuth buttons and the passkey affordances are still #651's and #652's,
 * and are absent rather than disabled: their backends have not landed.
 */
export function Login() {
  const { login, isSigningOut } = useAuth()
  const { sign_in_methods, mail_ready } = useDeployment()
  const usesPassword = sign_in_methods.includes("password")
  // An empty list is the gateway saying it cannot mint a session at all right
  // now, which is what `/v1/bootstrap` answers when it cannot reach its
  // database. Falling through to a credential form would offer the operator a
  // box whose only possible outcome is a refusal, and on a claimed deployment
  // it would be the *master-key* box, whose refusal reads as "wrong key".
  const signInUnavailable = sign_in_methods.length === 0
  // Every flow below the form begins with an email, so none of them can work
  // on a gateway that cannot send one. The two recovery links need one thing
  // more: an identity that already holds a password, which is exactly what
  // `password` (rather than `master_key`) reports. Nothing has a reset link to
  // reset or a verification to redo before then, and a claimed-but-unverified
  // signup already flips the deployment to `password`, so this is not the
  // caller who needs a resend being left without one.
  const offersSignup = mail_ready
  const offersRecovery = mail_ready && usesPassword

  const [masterKey, setMasterKey] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<unknown>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const credential: SignInCredential | null = usesPassword
    ? email.trim() && password
      ? { email: email.trim(), password }
      : null
    : masterKey.trim()
      ? { masterKey: masterKey.trim() }
      : null

  const clearError = () => {
    if (error) {
      setError(null)
    }
  }

  const submit = async () => {
    // isSigningOut blocks a new sign-in until a prior sign-out's server-side
    // revocation has finished (or timed out): otherwise its expiring cookie
    // could land after this one mints a fresh session and clobber it (#557).
    if (!credential || isSubmitting || isSigningOut) {
      return
    }
    setIsSubmitting(true)
    setError(null)
    try {
      const result = await createSession(credential)
      if (result.ok) {
        login()
      } else {
        // The gateway's own wording, not a guess: it distinguishes a wrong
        // credential from a master key presented to a deployment that has
        // retired it as a sign-in, and only it knows which happened.
        setError(
          new Error(
            result.message ??
              (usesPassword
                ? "Incorrect email or password."
                : "Invalid master key."),
          ),
        )
      }
    } catch (caught) {
      setError(caught)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (signInUnavailable) {
    return (
      <div className="flex min-h-full items-center justify-center p-6">
        <Card className="w-full max-w-md">
          <Card.Content className="flex flex-col gap-4 p-7 text-center">
            <img src="/favicon.svg" alt="Otari" className="mx-auto h-12 w-12" />
            <h1 className="text-lg font-semibold text-foreground">
              Sign-in is unavailable
            </h1>
            <p className="text-sm text-muted">
              This gateway cannot start a session at the moment, which usually
              means it cannot reach its database. It reports which credentials
              it accepts once it recovers, so reload this page to try again.
            </p>
            <p className="text-sm text-muted">
              The management API is unaffected by this screen and still accepts
              the master key.
            </p>
          </Card.Content>
        </Card>
      </div>
    )
  }

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col gap-5 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">
                Otari Dashboard
              </h1>
              <p className="mt-1 text-sm text-muted">
                {usesPassword
                  ? "Sign in to browse models, set pricing, and manage settings."
                  : "Sign in with your master key to browse models, set pricing, and manage settings."}
              </p>
            </div>
          </div>

          <form
            className="flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault()
              void submit()
            }}
          >
            {usesPassword ? (
              <>
                <TextField
                  value={email}
                  onChange={(next) => {
                    setEmail(next)
                    clearError()
                  }}
                  type="email"
                  isRequired
                  className="flex flex-col gap-1"
                >
                  <Label className="text-sm font-medium text-foreground">
                    Email
                  </Label>
                  <Input
                    placeholder="you@example.com"
                    autoFocus
                    autoComplete="username"
                  />
                </TextField>
                <TextField
                  value={password}
                  onChange={(next) => {
                    setPassword(next)
                    clearError()
                  }}
                  type="password"
                  isRequired
                  className="flex flex-col gap-1"
                >
                  <Label className="text-sm font-medium text-foreground">
                    Password
                  </Label>
                  <Input autoComplete="current-password" />
                </TextField>
              </>
            ) : (
              <>
                <TextField
                  value={masterKey}
                  onChange={(next) => {
                    setMasterKey(next)
                    clearError()
                  }}
                  type="password"
                  isRequired
                  className="flex flex-col gap-1"
                >
                  <Label className="text-sm font-medium text-foreground">
                    Master key
                  </Label>
                  <Input
                    placeholder="otari-mk-… or your master key"
                    autoFocus
                    autoComplete="off"
                  />
                </TextField>
                <details className="text-xs text-muted">
                  <summary className="cursor-pointer font-medium text-link hover:text-link-hover">
                    First run? Where to find your key
                  </summary>
                  <p className="mt-2 leading-relaxed">
                    If you did not set <code>OTARI_MASTER_KEY</code>, Otari
                    generated one and printed it to the server logs on startup.
                    Look for the line <code>Your master key:</code> (for
                    example, run <code>docker logs &lt;container&gt;</code>) and
                    paste it above.
                  </p>
                </details>
              </>
            )}
            <ErrorBanner error={error} />
            <Button
              type="submit"
              variant="primary"
              fullWidth
              isDisabled={!credential || isSubmitting || isSigningOut}
            >
              {isSigningOut
                ? "Finishing sign-out…"
                : isSubmitting
                  ? "Signing in…"
                  : "Sign in"}
            </Button>
          </form>

          <p className="text-center text-xs text-muted">
            {usesPassword
              ? "Your password is sent once to this gateway and exchanged for a session cookie. It is never written to browser storage, and the cookie that replaces it cannot be read by this page."
              : "The key is sent once to this gateway and exchanged for a session cookie. It is never written to browser storage, and the cookie that replaces it cannot be read by this page."}
          </p>

          <div className="flex flex-col items-center border-t border-border pt-2 text-center">
            {offersSignup ? (
              <PublicAuthLink to="#/signup">
                Added to this gateway? Claim your account
              </PublicAuthLink>
            ) : null}
            {offersRecovery ? (
              <PublicAuthLink to="#/recover-password">
                Forgot your password?
              </PublicAuthLink>
            ) : null}
            {offersRecovery ? (
              <PublicAuthLink to="#/resend-verification">
                Need a new verification link?
              </PublicAuthLink>
            ) : null}
            {/* Not a `PublicAuthLink`: `/welcome` is a page the gateway
                serves, so this one really is a navigation and not a hash
                change. Sized to match the links above it. */}
            <Link
              href="/welcome"
              className="inline-flex min-h-11 items-center text-sm font-medium text-link hover:text-link-hover"
            >
              New to Otari? Open the welcome guide
            </Link>
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}
