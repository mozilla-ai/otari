import { useState } from "react"
import { FiMail } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { PublicAuthFields } from "@/features/auth/overlayPublicAuthFields"
import { useSignup } from "@/shared/api/auth"
import { ApiError, startOAuthSignIn } from "@/shared/api/client"
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
import { rememberOAuthState } from "./OAuthCallbackPage"
import {
  type OAuthProvider,
  oauthProviderLabel,
  renderableOAuthProviders,
} from "./oauthProviders"
import {
  goToPublicAuthPage,
  PublicAuthLayout,
  PublicAuthLink,
} from "./PublicAuthLayout"

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
 * It applies to the address form only: an OAuth sign-in records no acceptance.
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
    if (isBusy) {
      return
    }
    setProviderError(null)
    signup.reset()
    setPendingProvider(provider)
    recordEvent(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: provider,
    })
    try {
      const started = await startOAuthSignIn(provider)
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
        return
      }
      rememberOAuthState(started.state)
      window.location.assign(started.authorizationUrl)
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
            <AuthProviderRows
              providers={providers}
              layout={isFormOpen ? "two-up" : "stacked"}
              verb="Sign up"
              pendingProvider={pendingProvider}
              isDisabled={signup.isPending}
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

            {/* Rendered only where there is a document to read, which is what
                the deployment's `terms_url` says. Required rather than
                optional: an acceptance the form would have submitted either way
                records nothing. A plain anchor and not a router `Link`, because
                the target is an address an operator configured and is usually
                off this origin, and beside the control rather than inside its
                label, which is the only arrangement that lets the terms be
                read: HTML exempts an interactive descendant from a label's own
                activation, but react-aria presses the label from a
                document-level handler that knows no such exemption and that
                nothing on the anchor can stop, so nested the link only ticked
                the box (otari-ai#2146). `ariaLabel` carries the sentence the
                visible label no longer holds in full. */}
            {terms_url !== null ? (
              <div className="flex flex-wrap items-center gap-x-1 text-caption">
                {/* 44px target from a pseudo-element bleed of 13 above and 12
                    below the 19px row, so the box is easy to hit on a phone
                    without `hasTouchTarget`, which would add 25px of height to
                    the card. The bleed is inside the form's own 12px gaps, so
                    it overlaps no neighbour. */}
                <Checkbox
                  isSelected={isTermsAccepted}
                  onChange={setIsTermsAccepted}
                  ariaLabel="I accept the terms of service"
                  className="relative before:absolute before:inset-x-0 before:-top-[13px] before:-bottom-[12px]"
                >
                  <span className="text-caption">I accept the</span>
                </Checkbox>
                <span>
                  <a
                    href={terms_url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-link hover:text-link-hover"
                  >
                    terms of service
                  </a>
                  .
                </span>
              </div>
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
