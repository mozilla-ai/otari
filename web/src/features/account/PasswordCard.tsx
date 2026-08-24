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
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import {
  useDeployment,
  useRetireMasterKeySignIn,
} from "@/shared/hooks/useDeployment"

interface PasswordFieldProps {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete: "current-password" | "new-password"
  description?: string
}

function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
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
      <Input autoComplete={autoComplete} />
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
 * rather than probed: `master_key` is published exactly while this deployment's
 * operator identity holds no password (otari#702).
 *
 * That is a fact about the operator and not about the reader, and the gap shows
 * here: a member who signed up on a deployment its operator never claimed is
 * shown the claim form, and no input to it can succeed (their own address wants
 * a `current_password` the form does not render; any other address is refused
 * as a change). The copy below says so up front rather than letting them find
 * out by submitting, which is as far as this can go without a route for asking
 * what the *signed-in* identity holds; the management API has none, and
 * inferring it from a refusal would be guessing at a 400. Reaching this state
 * at all means having gone around the sign-in screen, which offers such a
 * member no password form either. That context cannot be refetched, so a
 * successful claim reports itself through `useRetireMasterKeySignIn` and the
 * provider serves the corrected value from then on. This card therefore reads
 * the context on every render and keeps no mode of its own: the fact belongs to
 * the deployment, not to this component, and the account menu's session line
 * and the sign-in screen a later sign-out lands on read the same one.
 */
export function PasswordCard() {
  const { sign_in_methods } = useDeployment()
  const retireMasterKeySignIn = useRetireMasterKeySignIn()
  const setPassword = useSetPassword()

  // Read from the context on every render rather than seeded into local state.
  // A claim corrects the context (see `useRetireMasterKeySignIn`), so this card
  // switching forms, the account menu's session line, and the sign-in screen a
  // later sign-out lands on all move at once, and navigating away and back does
  // not return to a claim form for a deployment already claimed.
  const isClaimed = !sign_in_methods.includes("master_key")
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
  //
  // Never while one is in flight. `reset()` returns the observer to idle
  // without canceling the request, so resetting mid-call would clear the
  // `isPending` that `submit` guards on and let a keystroke reopen the form to
  // a second, concurrent password change.
  const clearResult = () => {
    if (setPassword.isPending) {
      return
    }
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
          // The server's own assertion, not an inference from which form was
          // submitted: it answers this on a change as well, and it is the fact
          // the rest of the tab has to act on.
          if (result.master_key_sign_in_retired) {
            retireMasterKeySignIn()
          }
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
              : "This gateway still signs in with its master key. Set an address and a password to sign in as yourself from now on. The master key stays the credential for the management API, and it can still reset this password if you forget it. Claiming is the operator's to do: if your own account already has a password, this form will refuse it, and your password changes once they have claimed."}
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
                {/* No autoFocus: this is a page, not a dialog, and focusing
                    a field on mount raises the soft keyboard over the
                    explanation above it before the operator has asked to
                    type. */}
                <Input placeholder="you@example.com" autoComplete="username" />
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
