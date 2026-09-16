import { Description, Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { Section } from "@/design-system/layout/Section"
import { useSetPassword } from "@/shared/api/auth"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useRetireMasterKeySignIn } from "@/shared/hooks/useDeployment"

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
      className="flex flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <Input autoComplete={autoComplete} />
      {description ? (
        // HeroUI's Description renders through the TextField's "description"
        // slot, so it reaches the input as aria-describedby; a raw span does
        // not, and a policy the field states only to sighted users is a policy
        // half the people typing into it cannot read.
        <FieldMessages>
          <Description className="text-muted">{description}</Description>
        </FieldMessages>
      ) : null}
    </TextField>
  )
}

/**
 * What the signed-in identity asks for, which is the whole of what the form has
 * to render.
 *
 * Both flags are the caller's own state and neither is the deployment's, which
 * is the correction this card exists for (mozilla-ai/otari-ai#2099). They are
 * also exactly the two conditions `PUT /v1/auth/password` branches on, so a
 * form built from them cannot ask for a field the gateway ignores or omit one
 * it requires.
 */
interface PasswordFormShape {
  /** The identity has no sign-in address, so this call has to supply one. */
  needsEmail: boolean
  /** The identity holds a password, so replacing it means proving the old one. */
  needsCurrentPassword: boolean
}

/**
 * The password this identity signs in to the dashboard with: set it for the
 * first time, or change it.
 *
 * One endpoint (`PUT /v1/auth/password`) behind three readings of one form, and
 * which one applies is read off the *caller*, not off the deployment. That is
 * the correction: `sign_in_methods` describes the gateway, and the question here
 * is what the person signed in right now holds, which `caller.has_password` and
 * `caller.email` on the membership context answer.
 *
 * - **Claiming.** First boot leaves the operator identity with no address and no
 *   password, and the master key as the dashboard login. Supplying an address
 *   and a password is the single act that retires master-key sign-in on this
 *   deployment (mozilla-ai/otari-ai#1716).
 * - **Setting a first password.** Somebody who signs in through Google, GitHub
 *   or a passkey holds no password, and neither does a roster entry nobody has
 *   claimed. They have an address already, so only a new password is asked for.
 *   Keying this off the deployment is what used to strand them: a claimed
 *   deployment showed everybody the change form, which asks for a current
 *   password they could never supply and offered no other way through.
 * - **Changing.** From then on the server requires the current password from a
 *   cookie-authenticated caller.
 *
 * The form is a dialog rather than three fields sitting open on the page. It is
 * a credential change reached by deliberate act, it is the one thing on this
 * page that is not safe to half-fill and wander away from, and the card can then
 * say in a line what the account currently signs in with.
 *
 * A successful call moves two things this component does not own. The address
 * and the new `has_password` are seated back onto the membership context by
 * `useSetPassword`, so this card re-reads its own shape rather than keeping a
 * mode of its own; and a claim reports itself through `useRetireMasterKeySignIn`,
 * because the bootstrap is a context read once per load and no invalidation
 * reaches it.
 */
export function PasswordCard() {
  const context = useOrganizationContext()
  const caller = context.data?.caller
  const [isOpen, setIsOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so a draft credential is
  // cleared on the way in rather than left in memory on the way out
  // (`SpendCeilingsCard` is the pattern).
  const [openCount, setOpenCount] = useState(0)
  // What the last successful call did. Kept on the card rather than in the
  // dialog, which is remounted: the line reporting it is what the page shows in
  // the dialog's place.
  const [outcome, setOutcome] = useState<{
    email: string
    claimed: boolean
  } | null>(null)

  const shape: PasswordFormShape | null = caller
    ? {
        needsEmail: !caller.email,
        needsCurrentPassword: caller.has_password,
      }
    : null

  return (
    <Section
      aria-labelledby="account-password-title"
      className="border-t border-border pt-6 pb-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="account-password-title" className="text-title">
        Dashboard password
      </h2>

      {context.isPending && !context.data ? (
        <Skeleton
          className="h-20 w-full max-w-md"
          ariaLabel="Loading your sign-in details"
        />
      ) : !shape ? (
        // No guessed form. Which of the three applies is a fact about the
        // signed-in identity, and a form built on the wrong guess is what this
        // card was fixing: the change form asks for a password an OAuth
        // sign-in never had, and the claim form's address is refused for
        // anybody who already holds one.
        <>
          <p className="max-w-3xl text-sm text-muted">
            Who is signed in could not be read, so there is nothing here to
            change yet.
          </p>
          <ErrorBanner error={context.error} />
        </>
      ) : (
        <>
          <p className="max-w-3xl text-sm text-muted">{summaryFor(shape)}</p>

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

          <div>
            <Button
              variant="primary"
              onPress={() => {
                // The saved line describes a call that is not the one about to
                // be made, so it goes with the form that replaces it.
                setOutcome(null)
                setOpenCount((count) => count + 1)
                setIsOpen(true)
              }}
            >
              {actionFor(shape)}
            </Button>
          </div>

          <PasswordDialog
            key={openCount}
            isOpen={isOpen}
            onOpenChange={setIsOpen}
            shape={shape}
            onSaved={setOutcome}
          />
        </>
      )}
    </Section>
  )
}

function summaryFor({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "The password you sign in to this dashboard with. Changing it ends every other session this identity holds; this one stays signed in."
  }
  if (needsEmail) {
    return "This gateway still signs in with its master key. Set an address and a password to sign in as yourself from now on. The master key stays the credential for the management API, and it can still reset this password if you forget it."
  }
  return "You have no dashboard password. You signed in another way, through a connected account or a passkey, and that keeps working: a password is a second way in, and it is what lets you sign in where those are not available."
}

function descriptionFor({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "You stay signed in here; every other session this identity holds ends."
  }
  if (needsEmail) {
    return "The address and password this deployment signs in with from now on. The master key stops being a dashboard login."
  }
  return "A second way in, beside however you sign in now. Every other session this identity holds ends."
}

function actionFor({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "Change password"
  }
  return needsEmail ? "Claim this deployment" : "Set a password"
}

/**
 * The form itself, and the mutation behind it, below the card's key on the
 * dialog: a draft credential is cleared on the way in rather than left in
 * memory on the way out, and a refused save cannot greet the next open.
 */
function PasswordDialog({
  isOpen,
  onOpenChange,
  shape,
  onSaved,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  shape: PasswordFormShape
  onSaved: (outcome: { email: string; claimed: boolean }) => void
}) {
  const retireMasterKeySignIn = useRetireMasterKeySignIn()
  const setPassword = useSetPassword()
  const [email, setEmail] = useState("")
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const { needsEmail, needsCurrentPassword } = shape
  const problem = newPasswordProblem(newPassword, confirmPassword)
  const unchanged =
    needsCurrentPassword &&
    newPassword !== "" &&
    newPassword === currentPassword
  const complete =
    newPassword !== "" &&
    confirmPassword !== "" &&
    (!needsEmail || email.trim() !== "") &&
    (!needsCurrentPassword || currentPassword !== "")
  const canSubmit = complete && problem === null && !unchanged

  // A refusal describes a call that is no longer the one being made, so typing
  // clears it. Never while one is in flight: `reset()` returns the observer to
  // idle without canceling the request, so resetting mid-call would clear the
  // `isPending` that `submit` guards on and let a keystroke open a second,
  // concurrent password change.
  const clearError = () => {
    if (!setPassword.isPending) {
      setPassword.reset()
    }
  }

  const submit = () => {
    if (!canSubmit || setPassword.isPending) {
      return
    }
    setPassword.mutate(
      {
        new_password: newPassword,
        ...(needsCurrentPassword ? { current_password: currentPassword } : {}),
        ...(needsEmail ? { email: email.trim() } : {}),
      },
      {
        onSuccess: (result) => {
          onSaved({ email: result.email, claimed: needsEmail })
          // The server's own assertion, not an inference from which form was
          // submitted: it answers this on a change as well, and it is the fact
          // the rest of the tab has to act on.
          if (result.master_key_sign_in_retired) {
            retireMasterKeySignIn()
          }
          onOpenChange(false)
        },
      },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      // The object rather than the act, which the submit names: all three
      // readings of this form mint or replace one password, and the claim is
      // the one that supplies the address it belongs to as well.
      title="Dashboard password"
      description={descriptionFor(shape)}
      submitLabel={actionFor(shape)}
      onSubmit={submit}
      isPending={setPassword.isPending}
      error={setPassword.error}
      isDirty={
        email !== "" ||
        currentPassword !== "" ||
        newPassword !== "" ||
        confirmPassword !== ""
      }
      isSubmitDisabled={!canSubmit}
    >
      {needsCurrentPassword ? (
        <PasswordField
          label="Current password"
          value={currentPassword}
          onChange={(next) => {
            setCurrentPassword(next)
            clearError()
          }}
          autoComplete="current-password"
        />
      ) : null}

      {needsEmail ? (
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
          <Label className="text-body">Email</Label>
          {/* autoComplete="username" and not "email": this is the handle the
              sign-in form will ask for, so a password manager should file it
              against the credential it is being set beside. */}
          <Input placeholder="you@example.com" autoComplete="username" />
          <FieldMessages>
            <Description className="text-muted">
              Changing this address later is not supported yet, so pick the one
              you will keep.
            </Description>
          </FieldMessages>
        </TextField>
      ) : null}

      <PasswordField
        label="New password"
        value={newPassword}
        onChange={(next) => {
          setNewPassword(next)
          clearError()
        }}
        autoComplete="new-password"
        description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
      />
      <PasswordField
        label="Confirm new password"
        value={confirmPassword}
        onChange={(next) => {
          setConfirmPassword(next)
          clearError()
        }}
        autoComplete="new-password"
      />

      {problem ? (
        <p role="alert" className="text-caption text-danger">
          {problem}
        </p>
      ) : null}
      {unchanged ? (
        <p role="alert" className="text-caption text-danger">
          The new password cannot be the one you already use.
        </p>
      ) : null}
    </FormDialog>
  )
}
