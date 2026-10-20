import { useState } from "react"
import { FiMail } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PublicAuthFields } from "@/features/auth/overlayPublicAuthFields"
import { useSignup } from "@/shared/api/auth"
import { ApiError } from "@/shared/api/client"
import { emailFromHash } from "@/shared/helpers/hashParams"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { AuthEmailField, AuthPasswordField } from "./AuthFields"
import { AuthHelp } from "./AuthHelp"
import {
  AuthMethodRow,
  AuthOrRule,
  AuthProviderRows,
} from "./AuthProviderButtons"
import {
  type OAuthProvider,
  oauthProviderLabel,
  renderableOAuthProviders,
} from "./oauthProviders"
import { leaveForOAuthProvider } from "./oauthRoundTrip"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"
import { rememberSignInAddress } from "./signInAddress"
import { TermsAcceptanceField } from "./TermsAcceptanceField"

/**
 * `#/signup`: register an address, or claim one an admin already added.
 *
 * Which of the two is the deployment's own posture, published as `open_signup`
 * in the bootstrap (otari-ai#2100). Closed, the default, `POST /v1/auth/signup`
 * only ever completes an identity `organization_service` already added or
 * invited by address and creates nothing from nothing, so the copy says so:
 * a page reading as "create an account" would leave somebody who is not on the
 * roster waiting for an email that is never sent. Open, an address nobody has
 * added is registered with an organization of its own, and the same page is a
 * registration form.
 *
 * **Open, with a provider configured, the provider is the way in** and the
 * address form is folded behind "Sign up with email": full-width provider rows
 * on top, the form opening in place below them, the rows shrinking to a two-up
 * pair once it does. A closed deployment shows no provider here at all: an
 * OAuth sign-in on a closed deployment only admits an address already on the
 * roster, so offering it as a way to sign *up* would promise an account it
 * never creates. The sign-in page still offers it. With no provider configured
 * the form is simply open.
 *
 * **The response says nothing about the address**, under either posture. It is
 * enumeration-safe: unknown, already claimed, and genuinely just claimed all
 * answer the same sentence. So success navigates to `#/check-email`, which is
 * written in the conditional the server's own message uses, and nothing here
 * branches on what came back.
 *
 * Its terms checkbox is here, conditionally, because the objection to it was
 * that a deployment publishing no terms has nothing to link to; `terms_url`
 * answers that, and the server has carried `terms_accepted` and its
 * `terms_accepted_at` column throughout. Where there is a document, accepting
 * it is required, which is what makes the recorded acceptance mean anything.
 * It gates the provider rows as well as the address form, because a provider
 * sign-up creates the account just as the form does, and the gateway refuses
 * one without the acceptance (`terms_not_accepted`). With providers on offer
 * it therefore sits above them, the one box for every way in on the card; with
 * none it stays in the form, above the button it gates.
 *
 * `?email=…` prefills the address for a link that names an invited one
 * (otari#835); the accept-invitation page sets a first password itself, so
 * it no longer sends anyone here. It arrives read-only, because the invitation
 * is bound to that address and claiming a different one would answer with the
 * same enumeration-safe sentence while doing nothing at all, and it opens the
 * form, since a link naming an address is not asking to be signed up with a
 * provider. The footer offers the plain page for anyone who does need another
 * address. Not a credential and not treated as one: `POST /v1/auth/signup`
 * checks this address against the roster itself.
 */
export function SignupPage({ hash }: { hash: string }) {
  const signup = useSignup()
  const { recordEvent } = useTelemetry()
  const { open_signup, terms_url, oauth_providers } = useDeployment()
  const providers = open_signup ? renderableOAuthProviders(oauth_providers) : []
  // Read straight from the prop rather than held in state: `PublicAuthPage` is
  // keyed on the whole hash, so a second link pasted into an open tab remounts
  // this page instead of re-rendering it with the first link's address.
  const invitedEmail = emailFromHash(hash)
  const [isFormOpen, setIsFormOpen] = useState(
    () => providers.length === 0 || invitedEmail !== null,
  )
  // Whether the visitor opened the form themselves. Only then may it take
  // focus: a form that is open on arrival must not raise the soft keyboard.
  const [didOpenForm, setDidOpenForm] = useState(false)
  const [email, setEmail] = useState(() => invitedEmail ?? "")
  const [password, setPassword] = useState("")
  const [isTermsAccepted, setIsTermsAccepted] = useState(false)
  const [pendingProvider, setPendingProvider] = useState<string>()
  const [providerError, setProviderError] = useState<unknown>(null)

  const problem = newPasswordProblem(password, "")
  const isComplete =
    email.trim() !== "" &&
    password !== "" &&
    (terms_url === null || isTermsAccepted)
  const canSubmit = isComplete && problem === null
  const isBusy = signup.isPending || pendingProvider !== undefined
  const awaitsTerms = terms_url !== null && !isTermsAccepted

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without canceling the request, so clearing mid-call would drop the
  // `isPending` that `submit` guards on and let a keystroke start a second one.
  const clearError = () => {
    if (signup.isPending) {
      return
    }
    signup.reset()
  }

  const submit = () => {
    if (!canSubmit || isBusy) {
      return
    }
    // The attempt, recorded before the request rather than alongside its
    // outcome, so a claim that never comes back is still a step in the funnel.
    recordEvent(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: "password",
    })
    signup.mutate(
      {
        email: email.trim(),
        password,
        // Present only where a document was actually shown. Omitted rather than
        // sent as `false` on a deployment that published none, so the column
        // records an acceptance of something rather than a decision about a
        // checkbox nobody saw. The box below is required, so reaching here with
        // terms published means it was ticked.
        ...(terms_url !== null ? { terms_accepted: true } : {}),
      },
      {
        onSuccess: () => {
          // Always verification-bound, and not a reading of the response: this
          // endpoint is enumeration-safe and says nothing about the address, so
          // the page navigates to check-email whatever came back.
          recordEvent(TELEMETRY_EVENTS.SIGNUP_SUCCESS, {
            authentication_method: "password",
            requires_verification: true,
          })
          rememberSignInAddress(email.trim())
          goToPublicAuthPage("#/check-email?type=signup")
        },
        onError: (error) => {
          recordEvent(TELEMETRY_EVENTS.SIGNUP_FAILED, {
            authentication_method: "password",
            status: error instanceof ApiError ? error.status : undefined,
          })
        },
      },
    )
  }

  /**
   * Start an OAuth sign-up: ask the gateway for a consent URL, then leave.
   *
   * The same call the sign-in page makes, and the same callback finishes it:
   * the provider's account either resolves to an identity here or, on a
   * deployment with open signup, registers one. No success telemetry is
   * recorded here, since nothing has succeeded yet; the callback page records
   * the outcome once there is one.
   */
  const startProvider = async (provider: OAuthProvider) => {
    if (isBusy || awaitsTerms) {
      return
    }
    setProviderError(null)
    signup.reset()
    setPendingProvider(provider)
    recordEvent(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: provider,
    })
    try {
      const started = await leaveForOAuthProvider(provider, {
        termsAccepted: terms_url !== null,
      })
      if (!started.isOk) {
        recordEvent(TELEMETRY_EVENTS.SIGNUP_FAILED, {
          authentication_method: provider,
          status: started.status,
        })
        setProviderError(
          new Error(
            started.message ??
              `${oauthProviderLabel(provider)} sign-up is not available on this gateway.`,
          ),
        )
        setPendingProvider(undefined)
      }
    } catch (caught) {
      recordEvent(TELEMETRY_EVENTS.SIGNUP_FAILED, {
        authentication_method: provider,
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setProviderError(caught)
      setPendingProvider(undefined)
    }
  }

  return (
    <PublicAuthLayout
      isBusy={isBusy}
      title={open_signup ? "Create your account" : "Claim your account"}
      description={
        open_signup
          ? "One account for every model."
          : "Set a password for the address an admin invited or added. You will confirm the address by email before your first sign-in."
      }
      footer={
        <div className="otari-auth-actions flex flex-wrap items-center justify-between gap-x-4">
          <PublicAuthLink to="#/">Sign in instead</PublicAuthLink>
          <AuthHelp offersRecovery />
        </div>
      }
    >
      <div className="flex flex-col">
        {/* Whatever an edition puts above the address (the data region). Its
            own wrapper so the gap below it exists only when it rendered. */}
        <div className="pb-3 empty:hidden">
          <PublicAuthFields page="signup" isBusy={isBusy} />
        </div>

        {providers.length > 0 ? (
          <>
            {terms_url !== null ? (
              <div className="pb-3">
                <TermsAcceptanceField
                  termsUrl={terms_url}
                  isAccepted={isTermsAccepted}
                  onChange={setIsTermsAccepted}
                />
              </div>
            ) : null}
            <AuthProviderRows
              providers={providers}
              layout={isFormOpen ? "two-up" : "stacked"}
              verb="Sign up"
              pendingProvider={pendingProvider}
              isDisabled={signup.isPending || awaitsTerms}
              onSelect={(provider) => void startProvider(provider)}
            />
            {providerError ? (
              <div className="pt-3">
                <ErrorBanner error={providerError} />
              </div>
            ) : null}
            <AuthOrRule />
          </>
        ) : null}

        {isFormOpen ? (
          <form
            className="flex flex-col gap-3"
            aria-busy={signup.isPending}
            onSubmit={(event) => {
              event.preventDefault()
              submit()
            }}
          >
            <AuthEmailField
              value={email}
              onChange={(next) => {
                setEmail(next)
                clearError()
              }}
              isReadOnly={invitedEmail !== null}
              autoFocus={didOpenForm}
              description={
                invitedEmail
                  ? "The address your invitation was sent to, which is the one it can claim."
                  : open_signup
                    ? undefined
                    : "The address an admin added or invited. Another address has nothing to claim."
              }
            />
            {/* Directly under the field rather than in the footer: this is the
                way out of a prefill that is wrong for whoever is reading, and
                someone who has just tried to type over a read-only field is
                looking here, not three rows below the submit button. */}
            {invitedEmail ? (
              <PublicAuthLink to="#/signup">
                Claim a different address instead
              </PublicAuthLink>
            ) : null}
            <AuthPasswordField
              label="Password"
              value={password}
              onChange={(next) => {
                setPassword(next)
                clearError()
              }}
              autoComplete="new-password"
              canReveal
              description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
              errorMessage={problem ?? undefined}
            />

            {terms_url !== null && providers.length === 0 ? (
              <TermsAcceptanceField
                termsUrl={terms_url}
                isAccepted={isTermsAccepted}
                onChange={setIsTermsAccepted}
              />
            ) : null}

            <ErrorBanner error={signup.error} />

            {/* Not `isPending`, and not `isDisabled`: both land on the product's
                one disabled treatment, which reads as refused. A submit in
                flight is working, so the fill stays, the press is blocked by
                hand (`pointer-events`, plus the guard in `submit` for the
                keyboard) and the form's `aria-busy` says what is happening,
                the way `FormDialog` does it. */}
            <Button
              type="submit"
              variant="primary"
              fullWidth
              isDisabled={!canSubmit || pendingProvider !== undefined}
              className={`h-11 ${signup.isPending ? "pointer-events-none" : ""}`}
            >
              {signup.isPending
                ? open_signup
                  ? "Creating account…"
                  : "Claiming account…"
                : open_signup
                  ? "Create account"
                  : "Claim account"}
            </Button>
          </form>
        ) : (
          <AuthMethodRow
            icon={FiMail}
            isDisabled={isBusy}
            onPress={() => {
              setDidOpenForm(true)
              setIsFormOpen(true)
            }}
          >
            Sign up with email
          </AuthMethodRow>
        )}
      </div>
    </PublicAuthLayout>
  )
}
