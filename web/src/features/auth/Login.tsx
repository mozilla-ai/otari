import { Button, Input, Label, Link, TextField } from "@heroui/react"
import { useState } from "react"
import { useAuth } from "@/features/auth/AuthContext"
import type { SignInCredential } from "@/shared/api/client"
import {
  ApiError,
  createSession,
  signInWithPasskey,
  startOAuthSignIn,
} from "@/shared/api/client"
import { errorMessage } from "@/shared/components/ui"
import {
  PasskeyCancelledError,
  supportsPasskeys,
} from "@/shared/helpers/webauthn"
import { useDeployment } from "@/shared/hooks/useDeployment"
import {
  analyticsErrorCode,
  analyticsStatusCode,
} from "@/shared/telemetry/errorCode"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { rememberOAuthState } from "./OAuthCallbackPage"
import {
  OAUTH_PROVIDER_ICONS,
  oauthProviderLabel,
  renderableOAuthProviders,
} from "./oauthProviders"
import { AuthPageShell, PublicAuthLink } from "./PublicAuthLayout"

/** Which box an error belongs beside. */
type CredentialField = "email" | "password" | "masterKey"

/**
 * `aria` rather than react-aria's default `native`, which puts a `required`
 * attribute on the input and lets the browser cancel the submit event outright.
 * That is a correct refusal announced in the wrong place: an unstyleable bubble
 * saying "Please fill out this field", gone on the next scroll, instead of the
 * message this form renders on the field's own label row. The field still
 * carries `aria-required`, which is the part a screen reader reads.
 */
const VALIDATION = "aria" as const

// Mirrors the gateway's `core/addresses.py` shape check. Keep the two in sync
// so the sign-in screen neither locks out a stored address nor accepts one the
// gateway will refuse.
const EMAIL_PATTERN = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

const ERROR_IDS: Record<CredentialField, string> = {
  email: "login-email-error",
  password: "login-password-error",
  masterKey: "login-master-key-error",
}

/**
 * The page frame: optically centered, and stable while the card grows.
 *
 * `items-center` gave the second at the cost of the first. Centering measures
 * the card, so anything that grows it moves its top edge up by half the growth,
 * and opening the first-run disclosure walked the submit button out from under
 * a pointer already resting on it. A flat top offset fixed that and left the
 * card sitting high with the page empty below it.
 *
 * So the offset is half the viewport minus a **constant** half-height, rather
 * than minus the card's real one. 17.5rem is the figure that best fits half of
 * what the card measures at rest across its branches, taken off the running
 * page rather than guessed: 537px for the master key with one footer link,
 * 581px with two, 589px for a password, 721px for a password with all four
 * links. Every one of those lands within 14px of true center, except the
 * tallest, which nearly fills a 900px window anyway. Because the figure is a
 * constant rather than the content's own height, the offset does not move when
 * the disclosure opens or a refusal wraps; the card grows downward from a
 * fixed top edge. `max()` floors it on a window shorter than
 * the card, where the page scrolls instead.
 *
 * `vh`, deliberately, not `dvh`: the dynamic unit shrinks when a phone's soft
 * keyboard opens, which would re-center the card at the exact moment someone is
 * typing into it.
 */

/**
 * The same frame for the unavailable state, whose card is around 351px rather
 * than 537px. Sharing the form's figure would leave this one sitting about
 * 110px high, which is the complaint the computed offset exists to answer.
 */

/** The column's own stack. The band's padding is on `AuthPageShell`. */
const CARD = "flex flex-col gap-6"

/**
 * The same, for the two states with no form in them. Left-aligned like
 * everything else in the column: centering a paragraph inside a left-pinned
 * band was the card pattern's habit, not this one's.
 */
const CARD_FLAT = "flex flex-col gap-4"

/** The screen's one page-defining line. */
const HEADING = "text-display"

/**
 * An inline code chip. The guide used to be a tinted block, which read as a
 * second surface inside a card that already is one; a chip keeps the two names
 * an operator has to copy inside the sentence that explains them.
 * `bg-surface-alt` is the registered utility for `--color-surface-muted`:
 * `bg-surface-muted` is declared nowhere and would compile to nothing at all.
 */
const CODE_CHIP =
  "bg-surface-alt px-1 py-px font-mono text-xs whitespace-nowrap text-foreground"

/** Sits on the label row beside a refusal. */
function AlertIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className="h-3.5 w-3.5 shrink-0"
    >
      <circle cx="12" cy="12" r="9.25" />
      <path d="M12 7.5v6" strokeLinecap="round" />
      <path d="M12 16.6h.01" strokeLinecap="round" strokeWidth="2.4" />
    </svg>
  )
}

/**
 * The reveal toggle's two states. Drawn rather than set as a glyph so it takes
 * `currentColor` and renders the same face on every platform.
 */
function EyeIcon({ isCrossedOut }: { isCrossedOut: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      aria-hidden="true"
      className="h-4.5 w-4.5"
    >
      <path
        d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="2.75" />
      {isCrossedOut ? (
        <path d="M4 20 20 4" strokeLinecap="round" strokeWidth="1.8" />
      ) : null}
    </svg>
  )
}

/** The disclosure's caret, rotating with its `<details open>`. */
function DisclosureCaret() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      aria-hidden="true"
      className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90 motion-reduce:transition-none"
    >
      <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/**
 * The row above a credential box: label and required marker grouped left, the
 * field's refusal right, both on the label's existing 20px line. This is where
 * an `<ErrorBanner>` used to go, between the last field and the button, and it
 * inserted about 46px right where the pointer already was. The card's fixed
 * top edge keeps the page stable when a gateway instruction wraps below this
 * row, rather than hiding the instruction a person needs to act on.
 */
function LabelRow({
  label,
  error,
  errorId,
}: {
  label: string
  error: unknown
  errorId: string
}) {
  const message = error ? errorMessage(error) : null

  return (
    <div className="flex min-h-5 flex-wrap items-center justify-between gap-x-3">
      <span className="flex shrink-0 items-center">
        {/* `text-foreground` holds the label at its resting ink while the field
            is invalid: HeroUI turns any `.label` under `[data-invalid]` red, and
            a red label beside a red message says the same thing twice. */}
        <Label className="text-foreground">{label}</Label>
        {/* HeroUI draws the asterisk from `[data-required] > .label`, a
            direct-child selector this row's wrapper breaks, so the marker is
            explicit. `isRequired` stays on the TextField, which is what carries
            `aria-required` to the input. */}
        <span
          aria-hidden="true"
          className="ms-0.5 text-sm font-medium text-danger"
        >
          *
        </span>
      </span>
      {message ? (
        <span
          id={errorId}
          role="alert"
          className="flex min-w-0 items-center gap-1 text-sm text-danger"
        >
          <AlertIcon />
          <span title={message}>{message}</span>
        </span>
      ) : null}
    </div>
  )
}

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
 * published exactly while the operator identity holds no password (otari#702),
 * so there is nothing yet for the operator to reset, and the way back in is the
 * master key against `PUT /v1/auth/password` (see docs/access-control.md). A
 * member who signed up on a deployment its operator never claimed is the one
 * case this screen does not serve: it offers the master-key box, and they sign
 * in by calling `POST /v1/auth/session` until the operator claims it.
 *
 * A passkey signs in beside the form rather than instead of it (otari#652),
 * offered only when the gateway publishes `passkey` *and* this browser can run
 * the ceremony. OAuth sits beside both (otari#651), one button per provider in
 * the bootstrap's `oauth_providers`, which lists only the providers an operator
 * configured: a provider nobody set up is absent rather than rendered disabled,
 * and a deployment that configured none carries no OAuth affordance at all.
 *
 * Three decisions here are load-bearing rather than cosmetic, and each carries
 * its own note where it is made: the card is anchored instead of centered, a
 * refusal is rendered on a label row instead of in a banner, and the submit
 * button is never disabled for an empty box. All three are about the moment a
 * pointer is already resting on that button.
 */
export function Login() {
  const { login, isSigningOut } = useAuth()
  const { recordEvent } = useTelemetry()
  const { sign_in_methods, mail_ready, maintenance_mode, oauth_providers } =
    useDeployment()
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
  // Two independent conditions, and both have to hold. The gateway publishes
  // `passkey` only while some credential could actually answer, and a browser
  // that cannot run the ceremony would turn the button into a dead end.
  const offersPasskey =
    sign_in_methods.includes("passkey") && supportsPasskeys()
  // Narrowed rather than rendered straight from the bootstrap: the gateway's
  // provider vocabulary is open (an overlay may bind an adapter for a
  // connection this build never named), and a provider with no label here would
  // become a button with no name.
  const oauthProviders = renderableOAuthProviders(oauth_providers)

  const [masterKey, setMasterKey] = useState("")
  const [isKeyVisible, setIsKeyVisible] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<unknown>(null)
  const [errorField, setErrorField] = useState<CredentialField | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  // Separate from `isSubmitting` because the two say different things while
  // they are true: the form's button reads "Signing in…", and this one has to
  // say the browser is waiting on an authenticator, which is a wait the person
  // has to act on rather than one they watch.
  const [isPasskeyPending, setIsPasskeyPending] = useState(false)
  // Which provider button was pressed, so only that one reads "Redirecting…".
  // The navigation that follows leaves this page, so this is never cleared on
  // success; it clears on the refusal path, where the person stays here.
  const [pendingProvider, setPendingProvider] = useState<string | null>(null)

  const clearError = () => {
    if (error) {
      setError(null)
      setErrorField(null)
    }
  }

  const fail = (field: CredentialField, message: string) => {
    setErrorField(field)
    setError(new Error(message))
  }

  /**
   * A refusal this form made itself, before any request went out.
   *
   * Recorded separately from a refusal the gateway made, because they are
   * different steps of the same funnel: this one is a form that could not be
   * sent, and `LOGIN_FAILED` is a credential the gateway would not take. The
   * reason is a code from a fixed set, never the box's contents.
   */
  const failValidation = (
    field: CredentialField,
    message: string,
    reason: string,
  ) => {
    recordEvent(TELEMETRY_EVENTS.FORM_VALIDATION_FAILED, {
      form_name: "login",
      errors: [reason],
    })
    fail(field, message)
  }

  /**
   * The credential, or `null` with the missing box already named. Emptiness is
   * checked here rather than by disabling the button: a disabled primary button
   * is white on the brand tint at 1.95:1, and an empty form is this screen's
   * resting state, so that unreadable pairing was the first thing an operator
   * saw on every visit. Submitting says which box to fill instead.
   */
  /**
   * Which credential an attempt actually presented, read off the credential
   * itself rather than off what the deployment offers.
   *
   * `sign_in_methods.includes("password")` answers the same question only while
   * that list has exactly two values. #652 adds `passkey`, and on a deployment
   * publishing `["password", "passkey"]` a passkey sign-in would report itself
   * as a password one with nothing failing. The platform's own vocabulary for
   * this property, minus the OAuth values whose buttons are still #651's.
   */
  const authenticationMethod = (credential: SignInCredential) =>
    "masterKey" in credential ? "master_key" : "password"

  // The third value the other two are drawn from `SignInCredential` for. A
  // passkey sign-in carries no credential object to derive it from: the whole
  // point is that nothing is typed, so the method is named rather than read.
  const PASSKEY_METHOD = "passkey"

  const readCredential = (): SignInCredential | null => {
    if (usesPassword) {
      if (!email.trim()) {
        failValidation("email", "Enter your email.", "email_required")
        return null
      }
      if (!EMAIL_PATTERN.test(email.trim())) {
        failValidation(
          "email",
          "Enter a valid email address.",
          "email_invalid_format",
        )
        return null
      }
      if (!password) {
        failValidation("password", "Enter your password.", "password_required")
        return null
      }
      return { email: email.trim(), password }
    }
    if (!masterKey.trim()) {
      failValidation(
        "masterKey",
        "Enter your master key.",
        "master_key_required",
      )
      return null
    }
    return { masterKey: masterKey.trim() }
  }

  const submit = async () => {
    // isSigningOut blocks a new sign-in until a prior sign-out's server-side
    // revocation has finished (or timed out): otherwise its expiring cookie
    // could land after this one mints a fresh session and clobber it (#557).
    //
    // isPasskeyPending is the same hazard from the other direction. A ceremony
    // in flight has the system sheet open over this page, but the form is still
    // live behind it and Enter still submits, so without this a passkey and a
    // password sign-in race and whichever cookie lands second wins.
    // `pendingProvider` blocks this for the same reason `isPasskeyPending`
    // does, and it matters more: an OAuth attempt ends in a navigation away
    // from this page, so a credential submitted while one is in flight can
    // mint a session that the provider's callback then replaces with a
    // session for whichever identity that account resolves to.
    if (isSubmitting || isSigningOut || isPasskeyPending || pendingProvider) {
      return
    }
    setError(null)
    setErrorField(null)
    const credential = readCredential()
    if (!credential) {
      return
    }
    setIsSubmitting(true)
    try {
      const result = await createSession(credential)
      if (result.ok) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
          authentication_method: authenticationMethod(credential),
        })
        login()
      } else {
        // The status, not a bucket of our own: a 401 is a wrong credential and
        // a 403 is a master key presented to a deployment that has retired it,
        // a distinction this screen already calls load-bearing, and collapsing
        // the two would throw it away in the one place it is measurable.
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: authenticationMethod(credential),
          error_code: analyticsStatusCode(result.status),
        })
        // The gateway's own wording, not a guess: it distinguishes a wrong
        // credential from a master key presented to a deployment that has
        // retired it as a sign-in, and only it knows which happened. It is
        // about the credential rather than one box, so it lands on the last row
        // above the button, where the operator's eye already is.
        fail(
          usesPassword ? "password" : "masterKey",
          result.message ??
            (usesPassword
              ? "Incorrect email or password."
              : "Invalid master key."),
        )
      }
    } catch (caught) {
      // The gateway's message is not recorded, only its status: a refusal's
      // wording is the one part of it that can carry something the operator
      // typed.
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: authenticationMethod(credential),
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setErrorField(usesPassword ? "password" : "masterKey")
      setError(caught)
    } finally {
      setIsSubmitting(false)
    }
  }

  /**
   * Sign in with a passkey: options, the browser ceremony, then the assertion.
   *
   * A dismissed prompt clears back to the resting state and says nothing. It is
   * not a refused credential, and the person who pressed Escape does not need
   * the screen to tell them what they just did.
   *
   * A refusal lands on the credential row above the button, where the form's
   * own refusals land, for the same reason: it is about the credential rather
   * than about one box.
   */
  const submitPasskey = async () => {
    if (isSubmitting || isSigningOut || isPasskeyPending || pendingProvider) {
      return
    }
    setError(null)
    setErrorField(null)
    setIsPasskeyPending(true)
    try {
      const result = await signInWithPasskey()
      if (result.ok) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
          authentication_method: PASSKEY_METHOD,
        })
        login()
      } else {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: PASSKEY_METHOD,
          error_code: analyticsStatusCode(result.status),
        })
        fail(
          usesPassword ? "password" : "masterKey",
          result.message ?? "That passkey did not sign you in.",
        )
      }
    } catch (caught) {
      // A dismissed prompt is recorded as its own outcome rather than as a
      // failure or not at all: it is the most common way this button ends, and
      // counting it as a failure would make the passkey path look broken while
      // dropping it would hide how often people back out of the sheet.
      if (caught instanceof PasskeyCancelledError) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: PASSKEY_METHOD,
          error_code: "passkey_cancelled",
        })
        return
      }
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: PASSKEY_METHOD,
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setErrorField(usesPassword ? "password" : "masterKey")
      setError(caught)
    } finally {
      setIsPasskeyPending(false)
    }
  }

  /**
   * Start an OAuth sign-in: ask the gateway for a consent URL, then leave.
   *
   * The `state` the gateway mints is stored before the navigation and compared
   * on the way back, in `OAuthCallbackPage`. That is the CSRF check, and it
   * happens in the browser because this deployment keeps nothing between the
   * two requests; see `src/gateway/services/oauth_service.py`.
   *
   * `window.location.assign` rather than a router navigation, because the
   * destination is the provider's own origin: this really is leaving the app.
   * No success telemetry is recorded here. Nothing has succeeded yet, and the
   * callback page records the outcome once there is one.
   */
  const submitOAuth = async (provider: string) => {
    if (isSubmitting || isSigningOut || isPasskeyPending || pendingProvider) {
      return
    }
    setError(null)
    setErrorField(null)
    setPendingProvider(provider)
    try {
      const started = await startOAuthSignIn(provider)
      if (!started.ok) {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: provider,
          error_code: analyticsStatusCode(started.status),
        })
        fail(
          usesPassword ? "password" : "masterKey",
          started.message ??
            `${oauthProviderLabel(provider)} sign-in is not available on this gateway.`,
        )
        setPendingProvider(null)
        return
      }
      rememberOAuthState(started.state)
      window.location.assign(started.authorizationUrl)
    } catch (caught) {
      recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: provider,
        error_code: analyticsErrorCode(caught),
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setErrorField(usesPassword ? "password" : "masterKey")
      setError(caught)
      setPendingProvider(null)
    }
  }

  if (signInUnavailable) {
    return (
      <AuthPageShell>
        <div className={CARD_FLAT}>
          <h1 className={HEADING}>Otari sign-in is unavailable</h1>
          <p className="text-sm text-muted">
            This gateway cannot start a session at the moment, which usually
            means it cannot reach its database. It reports which credentials it
            accepts once it recovers, so reload this page to try again.
          </p>
          <p className="text-sm text-muted">
            The management API is unaffected by this screen and still accepts
            the master key.
          </p>
        </div>
      </AuthPageShell>
    )
  }

  // An operator has frozen sign-ins to redeploy this gateway. Rendering the
  // form instead would offer a credential whose only outcome is a 503, and one
  // whose refusal reads as "wrong key" to anyone who does not already know a
  // freeze is on. Read from the bootstrap, so this is what the page shows on a
  // fresh load; a tab that was already open when the freeze started still has
  // the form, and submitting it renders the gateway's own 503 wording on the
  // label row, the same way every other refusal arrives.
  //
  // Checked after `signInUnavailable` because the two cannot both be true: the
  // database failure that empties `sign_in_methods` is also what makes the
  // bootstrap report no freeze, and "cannot reach its database" is the more
  // actionable of the two if they ever did collide.
  if (maintenance_mode) {
    return (
      <AuthPageShell>
        <div className={CARD_FLAT}>
          <h1 className={HEADING}>Otari is under maintenance</h1>
          <p className="text-sm text-muted">
            This gateway is not starting new dashboard sessions while it is
            being updated. It should be back shortly, so reload this page to try
            again.
          </p>
          <p className="text-sm text-muted">
            The API is unaffected by this screen and still serves requests, and
            the management API still accepts the master key.
          </p>
        </div>
      </AuthPageShell>
    )
  }

  return (
    <AuthPageShell>
      <div className={CARD}>
        {/* The mark is on the bar now, so the title stands on its own and the
              column starts at its left edge like every other column in the
              product. */}
        <div className="flex flex-col gap-1.5">
          <h1 className={HEADING}>Otari Dashboard</h1>
          <p className="text-sm text-pretty text-muted">
            {usesPassword
              ? "Sign in to browse models, set pricing, and manage settings."
              : "Sign in with your master key to browse models, set pricing, and manage settings."}
          </p>
        </div>

        {/* Section boundaries read 24px on screen throughout. A 44px row
              carries 12px of invisible padding above and below its own text, so
              the column next to one runs at 12px to land on the same 24px: the
              master-key branch has the disclosure's summary row in it, and the
              password branch has no such row and spaces at the full 24px. */}
        <form
          className={`flex flex-col ${usesPassword ? "gap-6" : "gap-3"}`}
          noValidate
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
                validationBehavior={VALIDATION}
                isInvalid={errorField === "email"}
                className="flex flex-col gap-2"
              >
                <LabelRow
                  label="Email"
                  error={errorField === "email" ? error : null}
                  errorId={ERROR_IDS.email}
                />
                {/* 16px, not the 14px HeroUI drops to from `sm:` up: under
                      16px iOS Safari zooms the page on focus. No autoFocus,
                      which on a page load raises the soft keyboard over half a
                      phone screen and makes a focus ring the resting state. */}
                <Input
                  placeholder="you@example.com"
                  autoComplete="username"
                  aria-describedby={
                    errorField === "email" ? ERROR_IDS.email : undefined
                  }
                  className="h-10 text-base"
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
                validationBehavior={VALIDATION}
                isInvalid={errorField === "password"}
                className="flex flex-col gap-2"
              >
                <LabelRow
                  label="Password"
                  error={errorField === "password" ? error : null}
                  errorId={ERROR_IDS.password}
                />
                <Input
                  autoComplete="current-password"
                  aria-describedby={
                    errorField === "password" ? ERROR_IDS.password : undefined
                  }
                  className="h-10 text-base"
                />
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
                type={isKeyVisible ? "text" : "password"}
                isRequired
                validationBehavior={VALIDATION}
                isInvalid={errorField === "masterKey"}
                className="flex flex-col gap-2"
              >
                <LabelRow
                  label="Master key"
                  error={errorField === "masterKey" ? error : null}
                  errorId={ERROR_IDS.masterKey}
                />
                <div className="relative">
                  <Input
                    placeholder="otari-mk-…"
                    autoComplete="off"
                    aria-describedby={
                      errorField === "masterKey"
                        ? ERROR_IDS.masterKey
                        : undefined
                    }
                    fullWidth
                    className="h-10 pr-10 font-mono text-base"
                  />
                  {/* A 40-character key pasted into a masked box cannot be
                        checked against the one in the logs, which is the whole
                        reason to fail a sign-in twice. The visible target is
                        the 36px slot inside a 44px field; `before` carries the
                        44px touch floor past it, rather than a hover fill the
                        height of the whole field doing it. */}
                  <Button
                    type="button"
                    variant="ghost"
                    isIconOnly
                    size="sm"
                    aria-label={
                      isKeyVisible ? "Hide master key" : "Show master key"
                    }
                    onPress={() => setIsKeyVisible((shown) => !shown)}
                    className="absolute top-1 right-1 h-9 w-9 text-muted before:absolute before:-inset-1"
                  >
                    <EyeIcon isCrossedOut={isKeyVisible} />
                  </Button>
                </div>
              </TextField>
              <details className="group">
                <summary className="flex min-h-11 cursor-pointer list-none items-center gap-2 text-sm font-medium text-link hover:text-link-hover [&::-webkit-details-marker]:hidden">
                  <DisclosureCaret />
                  First run? Where to find your key
                </summary>
                {/* The 12px tail is what keeps the gap to the button reading
                      24px once this is open, since the 12px it sits at closed
                      is the summary row's invisible padding doing that job. */}
                <p className="pb-3 text-caption">
                  No <code className={CODE_CHIP}>OTARI_MASTER_KEY</code> set?
                  Otari printed one to the server logs on startup. Find it with{" "}
                  <code className={CODE_CHIP}>
                    docker logs &lt;container&gt;
                  </code>
                  .
                </p>
              </details>
            </>
          )}
          {/* Disabled only while a prior sign-out is still revoking (#557),
                or while another credential is already mid-flight. Never for an
                empty box: see readCredential. */}
          <Button
            type="submit"
            variant="primary"
            fullWidth
            isDisabled={
              isSubmitting ||
              isSigningOut ||
              isPasskeyPending ||
              pendingProvider !== null
            }
            className="h-11"
          >
            {isSigningOut
              ? "Finishing sign-out…"
              : isSubmitting
                ? "Signing in…"
                : "Sign in"}
          </Button>
        </form>

        {offersPasskey || oauthProviders.length > 0 ? (
          <div className="flex flex-col gap-3">
            {/* A rule with the word on it, rather than a bare divider: these
                  are alternatives to the form above, not a second step of it,
                  and an unlabeled line reads as the latter. One rule for both
                  groups, however many buttons follow it, because they are all
                  the same alternative: another way to prove the same thing. */}
            <div
              className="flex items-center gap-3 text-xs text-muted"
              aria-hidden
            >
              <span className="h-px flex-1 bg-border" />
              or
              <span className="h-px flex-1 bg-border" />
            </div>
            {offersPasskey ? (
              <Button
                type="button"
                variant="secondary"
                fullWidth
                isDisabled={
                  isSubmitting || isSigningOut || pendingProvider !== null
                }
                onPress={() => void submitPasskey()}
                className="h-11"
              >
                {isPasskeyPending
                  ? "Waiting for your passkey…"
                  : "Use a passkey"}
              </Button>
            ) : null}
            {oauthProviders.map((provider) => {
              const Mark = OAUTH_PROVIDER_ICONS[provider]
              const isRedirecting = pendingProvider === provider
              return (
                <Button
                  key={provider}
                  type="button"
                  variant="secondary"
                  fullWidth
                  isDisabled={
                    isSubmitting ||
                    isSigningOut ||
                    isPasskeyPending ||
                    pendingProvider !== null
                  }
                  onPress={() => void submitOAuth(provider)}
                  className="h-11"
                >
                  {/* The mark is decorative: the label beside it already
                        names the provider, so announcing it again would read
                        the button's own text twice. Dropped while redirecting,
                        so the row does not keep a logo beside a label that no
                        longer names a provider to press. */}
                  {isRedirecting ? null : (
                    <Mark className="text-xl" aria-hidden />
                  )}
                  {isRedirecting
                    ? "Redirecting…"
                    : `Sign in with ${oauthProviderLabel(provider)}`}
                </Button>
              )
            })}
          </div>
        ) : null}

        {/* Under a rule of its own: what becomes of the credential is a
                note about the form above rather than a step of it, and the
                separator is what says so now that no card edge does.
                Left-aligned with the column, like everything else in it. */}
        <div className="flex flex-col gap-3 border-t border-border pt-5">
          <p className="text-xs text-muted">
            Your{" "}
            <a
              href="/welcome"
              className="font-medium text-link hover:text-link-hover"
            >
              master key
            </a>{" "}
            is sent once and exchanged for a session cookie. It is never stored
            in the browser.
          </p>
          {/* The rows
                themselves take no gap, because each is 44px around a 20px line
                and so already sits 24px from its neighbor's text. */}
          <div className="flex flex-col">
            {/* Deployment-neutral wording (otari#835): "this gateway" read as
                  a self-hosted process on a hosted control plane, where the same
                  screen is the sign-in for an invited tenant. */}
            {offersSignup ? (
              <PublicAuthLink to="#/signup">
                Invited or added by an admin? Set your password
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
        </div>
      </div>
    </AuthPageShell>
  )
}
