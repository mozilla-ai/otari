import { Button, Card, Input, Label, Link, TextField } from "@heroui/react"
import { useState } from "react"
import { useAuth } from "@/features/auth/AuthContext"
import type { SignInCredential } from "@/shared/api/client"
import { ApiError, createSession } from "@/shared/api/client"
import { errorMessage } from "@/shared/components/ui"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

import { PublicAuthLink } from "./PublicAuthLayout"

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
const PAGE =
  "flex min-h-full items-start justify-center px-4 pt-[max(2rem,calc(50vh-17.5rem))] pb-16"

/**
 * The same frame for the unavailable state, whose card is around 351px rather
 * than 537px. Sharing the form's figure would leave this one sitting about
 * 110px high, which is the complaint the computed offset exists to answer.
 */
const PAGE_FLAT =
  "flex min-h-full items-start justify-center px-4 pt-[max(2rem,calc(50vh-11.5rem))] pb-16"

/**
 * Card padding, on top of the 16px `<Card>` itself contributes. Read on screen
 * the top and bottom are equal: the bottom's smaller figure is completed by the
 * 12px of invisible padding under the 44px row the last footer link sits in.
 */
const CARD = "flex flex-col gap-6 px-6 pt-6 pb-3 sm:px-8 sm:pt-8 sm:pb-5"

/**
 * The same card for the unavailable state, which ends in a paragraph rather
 * than a 44px row, so it has no invisible 12px to borrow and pays the padding
 * itself.
 */
const CARD_FLAT = "flex flex-col gap-6 px-6 py-6 text-center sm:px-8 sm:py-8"

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
  "rounded bg-surface-alt px-1 py-px font-mono text-xs whitespace-nowrap text-foreground"

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
 * The OAuth buttons and the passkey affordances are still #651's and #652's,
 * and are absent rather than disabled: their backends have not landed.
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
  const { sign_in_methods, mail_ready, maintenance_mode } = useDeployment()
  const usesPassword = sign_in_methods.includes("password")
  // Which credential this attempt presented, recorded on every outcome so the
  // two sign-ins a deployment can offer stay separable in the funnel. The
  // platform's own vocabulary for this property, minus the OAuth values whose
  // buttons are still otari#651's and #652's.
  const authenticationMethod = usesPassword ? "password" : "master_key"
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
  const [isKeyVisible, setIsKeyVisible] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<unknown>(null)
  const [errorField, setErrorField] = useState<CredentialField | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

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
    if (isSubmitting || isSigningOut) {
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
          authentication_method: authenticationMethod,
        })
        login()
      } else {
        recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED, {
          authentication_method: authenticationMethod,
          error_code: "credential_rejected",
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
        authentication_method: authenticationMethod,
        error_code: "request_failed",
        status: caught instanceof ApiError ? caught.status : undefined,
      })
      setErrorField(usesPassword ? "password" : "masterKey")
      setError(caught)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (signInUnavailable) {
    return (
      <div className={PAGE_FLAT}>
        <Card className="w-full max-w-md">
          <Card.Content className={CARD_FLAT}>
            {/* Grouped so the mark sits 16px from the heading it belongs to,
                the way it does on the form. alt="" because the <h1> under it
                already names the product. */}
            <div className="flex flex-col items-center gap-4">
              <img src="/favicon.svg" alt="" className="h-10 w-11" />
              <h1 className={HEADING}>Otari sign-in is unavailable</h1>
            </div>
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
      <div className={PAGE_FLAT}>
        <Card className="w-full max-w-md">
          <Card.Content className={CARD_FLAT}>
            <div className="flex flex-col items-center gap-4">
              <img src="/favicon.svg" alt="" className="h-10 w-11" />
              <h1 className={HEADING}>Otari is under maintenance</h1>
            </div>
            <p className="text-sm text-muted">
              This gateway is not starting new dashboard sessions while it is
              being updated. It should be back shortly, so reload this page to
              try again.
            </p>
            <p className="text-sm text-muted">
              The API is unaffected by this screen and still serves requests,
              and the management API still accepts the master key.
            </p>
          </Card.Content>
        </Card>
      </div>
    )
  }

  return (
    <div className={PAGE}>
      <Card className="w-full max-w-md">
        <Card.Content className={CARD}>
          <div className="flex flex-col items-center gap-4 text-center">
            {/* Decorative: the <h1> beside it says "Otari Dashboard". */}
            <img src="/favicon.svg" alt="" className="h-10 w-11" />
            <div className="flex flex-col gap-1.5">
              <h1 className={HEADING}>Otari Dashboard</h1>
              <p className="text-sm text-pretty text-muted">
                {usesPassword
                  ? "Sign in to browse models, set pricing, and manage settings."
                  : "Sign in with your master key to browse models, set pricing, and manage settings."}
              </p>
            </div>
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
                    className="h-11 text-base"
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
                    className="h-11 text-base"
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
                      className="h-11 pr-11 font-mono text-base"
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
                    Otari printed one to the server logs on startup. Find it
                    with{" "}
                    <code className={CODE_CHIP}>
                      docker logs &lt;container&gt;
                    </code>
                    .
                  </p>
                </details>
              </>
            )}
            {/* Disabled only while a prior sign-out is still revoking (#557).
                Never for an empty box: see readCredential. */}
            <Button
              type="submit"
              variant="primary"
              fullWidth
              isDisabled={isSubmitting || isSigningOut}
              className="h-11"
            >
              {isSigningOut
                ? "Finishing sign-out…"
                : isSubmitting
                  ? "Signing in…"
                  : "Sign in"}
            </Button>
          </form>

          <div className="flex flex-col items-center gap-3">
            <p className="text-center text-xs text-balance text-muted">
              Sent once and exchanged for a session cookie. Never stored in the
              browser.
            </p>
            {/* No rule above these: at 1.38:1 the border was a line nobody
                could see, separating two things nobody was confusing. The rows
                themselves take no gap, because each is 44px around a 20px line
                and so already sits 24px from its neighbor's text. */}
            {/* `text-center` because a link long enough to wrap on a phone
                ("Claim your account" does at 390px) would otherwise rag left
                out of the lane the single-line rows sit in. */}
            <div className="flex flex-col items-center text-center">
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
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}
