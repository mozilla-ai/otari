import {
  Button,
  Card,
  Description,
  Input,
  Label,
  TextField,
} from "@heroui/react"
import { useState } from "react"

import { useSetPassword } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { useDeployment } from "@/shared/hooks/useDeployment"

// The client half of the server's password policy
// (`gateway.services.password_service`), not a second authority: it disables a
// save the gateway would refuse instead of round-tripping to a 400. The ceiling
// is counted in bytes because bcrypt's is, so an accented character spends more
// than one and a character count would let a 72-character password through to
// the refusal this exists to pre-empt.
const MIN_PASSWORD_LENGTH = 8
const MAX_PASSWORD_BYTES = 72

function passwordByteLength(password: string): number {
  return new TextEncoder().encode(password).length
}

/**
 * Why the new password cannot be saved yet, or null when it can.
 *
 * Returns the empty-field case as null rather than as a complaint: a form
 * nobody has typed in yet is not wrong, it is unfinished, and the disabled Save
 * already says so.
 */
function newPasswordProblem(password: string, confirm: string): string | null {
  if (password === "") {
    return null
  }
  if (password.length < MIN_PASSWORD_LENGTH) {
    return `At least ${MIN_PASSWORD_LENGTH} characters.`
  }
  if (passwordByteLength(password) > MAX_PASSWORD_BYTES) {
    return `At most ${MAX_PASSWORD_BYTES} bytes; accented characters count for more than one.`
  }
  if (confirm !== "" && confirm !== password) {
    return "The two passwords do not match."
  }
  return null
}

interface PasswordFieldProps {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete: "current-password" | "new-password"
  autoFocus?: boolean
  description?: string
}

function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
  autoFocus,
  description,
}: PasswordFieldProps) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      type="password"
      isRequired
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <Input autoComplete={autoComplete} autoFocus={autoFocus} />
      {description ? (
        // HeroUI's Description renders through the TextField's "description"
        // slot, so it reaches the input as aria-describedby; a raw span does
        // not, and a policy the field states only to sighted users is a policy
        // half the people typing into it cannot read.
        <Description className="text-xs text-muted">{description}</Description>
      ) : null}
    </TextField>
  )
}

/**
 * The password this identity signs in to the dashboard with: set it for the
 * first time, or change it.
 *
 * One endpoint (`PUT /v1/auth/password`) behind two forms, because the two acts
 * ask for different things and mean different things to the operator reading
 * them:
 *
 * - **Claiming.** First boot leaves the operator identity with no address and
 *   no password, and the master key as the dashboard login. Supplying an
 *   address and a password is the single act that retires master-key sign-in on
 *   this deployment (mozilla-ai/otari-ai#1716). No current password is asked
 *   for, because there is none.
 * - **Changing.** From then on the identity has both, and the server requires
 *   the current password from a cookie-authenticated caller. This dashboard is
 *   always cookie-authenticated, so that field is always required here.
 *
 * Which of the two applies is read from the bootstrap's `sign_in_methods`
 * rather than probed: `master_key` is published exactly while no identity on
 * this deployment holds a password. That context is populated once per page
 * load and cannot be refetched, so a claim that succeeds in this tab leaves it
 * stale; the response is what this card believes from then on, which is why the
 * mode is state seeded from the bootstrap rather than the bootstrap read
 * directly.
 */
export function PasswordCard() {
  const { sign_in_methods } = useDeployment()
  const setPassword = useSetPassword()

  // Seeded once, then owned locally: claiming in this tab flips it without a
  // reload, and re-reading the bootstrap would flip it back.
  const [isClaimed, setClaimed] = useState(
    () => !sign_in_methods.includes("master_key"),
  )
  const [email, setEmail] = useState("")
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  // What the last successful call did, kept because neither fact survives it
  // otherwise: the address comes back in the response and is the only way this
  // page ever learns one (the management API exposes no "who am I" route yet),
  // and whether it was the claim cannot be read off `isClaimed` afterwards,
  // since claiming is what sets that.
  const [outcome, setOutcome] = useState<{
    email: string
    claimed: boolean
  } | null>(null)

  const problem = newPasswordProblem(newPassword, confirmPassword)
  const unchanged =
    isClaimed && newPassword !== "" && newPassword === currentPassword
  const complete = isClaimed
    ? currentPassword !== "" && newPassword !== "" && confirmPassword !== ""
    : email.trim() !== "" && newPassword !== "" && confirmPassword !== ""
  // Deliberately not gated on `isPending`: that is the Button's own prop, which
  // keeps it focusable and announces it busy, where `isDisabled` would drop
  // focus out of the form mid-request. Double submission is stopped in
  // `submit` instead, which is where it has to be anyway (a form submits on
  // Enter, not only through the button).
  const canSubmit = complete && problem === null && !unchanged

  // A refusal and the line reporting the last success both describe a call that
  // is no longer the one being made, so typing clears them together.
  const clearResult = () => {
    setOutcome(null)
    setPassword.reset()
  }

  const submit = () => {
    if (!canSubmit || setPassword.isPending) {
      return
    }
    setPassword.mutate(
      isClaimed
        ? { current_password: currentPassword, new_password: newPassword }
        : { email: email.trim(), new_password: newPassword },
      {
        onSuccess: (result) => {
          setOutcome({ email: result.email, claimed: !isClaimed })
          setClaimed(true)
          setEmail("")
          setCurrentPassword("")
          setNewPassword("")
          setConfirmPassword("")
        },
      },
    )
  }

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-foreground">
        {isClaimed ? "Dashboard password" : "Claim this deployment"}
      </h2>
      <Card>
        <Card.Content className="flex flex-col gap-4 px-5 py-5">
          <p className="max-w-3xl text-sm text-muted">
            {isClaimed
              ? "The password you sign in to this dashboard with. Changing it ends every other session this identity holds; this one stays signed in."
              : "This gateway still signs in with its master key. Set an address and a password to sign in as yourself from now on. The master key stays the credential for the management API, and it can still reset this password if you forget it."}
          </p>

          {outcome ? (
            <p
              role="status"
              aria-live="polite"
              className="max-w-3xl text-sm text-success"
            >
              {outcome.claimed
                ? `Saved. Sign in as ${outcome.email} from now on: the master key no longer signs in to this dashboard, and it stays the credential for the management API.`
                : `Saved. Your other sessions have ended; sign in as ${outcome.email} next time.`}
            </p>
          ) : null}

          <form
            className="flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault()
              submit()
            }}
          >
            {isClaimed ? (
              <PasswordField
                label="Current password"
                value={currentPassword}
                onChange={(next) => {
                  setCurrentPassword(next)
                  clearResult()
                }}
                autoComplete="current-password"
              />
            ) : (
              <TextField
                value={email}
                onChange={(next) => {
                  setEmail(next)
                  clearResult()
                }}
                type="email"
                isRequired
                className="flex max-w-md flex-col gap-1"
              >
                <Label className="text-sm font-medium text-foreground">
                  Email
                </Label>
                {/* autoComplete="username" and not "email": this is the handle
                    the sign-in form will ask for, so a password manager should
                    file it against the credential it is being set beside. */}
                <Input
                  placeholder="you@example.com"
                  autoComplete="username"
                  autoFocus
                />
                <Description className="text-xs text-muted">
                  Changing this address later is not supported yet, so pick the
                  one you will keep.
                </Description>
              </TextField>
            )}

            <PasswordField
              label="New password"
              value={newPassword}
              onChange={(next) => {
                setNewPassword(next)
                clearResult()
              }}
              autoComplete="new-password"
              description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
            />
            <PasswordField
              label="Confirm new password"
              value={confirmPassword}
              onChange={(next) => {
                setConfirmPassword(next)
                clearResult()
              }}
              autoComplete="new-password"
            />

            {problem ? (
              <p role="alert" className="text-sm text-danger">
                {problem}
              </p>
            ) : null}
            {unchanged ? (
              <p role="alert" className="text-sm text-danger">
                The new password cannot be the one you already use.
              </p>
            ) : null}
            <ErrorBanner error={setPassword.error} />

            <div>
              <Button
                type="submit"
                variant="primary"
                isPending={setPassword.isPending}
                isDisabled={!canSubmit}
              >
                {isClaimed ? "Change password" : "Set password"}
              </Button>
            </div>
          </form>
        </Card.Content>
      </Card>
    </section>
  )
}
