import { Description, Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { useSetPassword } from "@/shared/api/auth"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useRetireMasterKeySignIn } from "@/shared/hooks/useDeployment"

import {
  type PasswordFormShape,
  passwordAction,
  passwordDialogDescription,
} from "./passwordForm"

/**
 * A credential field, rather than `SecretField`.
 *
 * That one suppresses password managers on purpose, because the secrets it
 * collects are write-only provider keys nobody should be offered back. This is
 * a sign-in password, so the opposite is wanted: `autoComplete` is what lets a
 * manager file it against the address it belongs to.
 */
function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
  description,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete: "current-password" | "new-password"
  description?: string
}) {
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
 * The password form, and the mutation behind it.
 *
 * Below the card's key on this component, which is the house shape for a
 * dialog's draft (`SpendCeilingDialog` says the same): a draft credential is
 * cleared on the way in rather than left in memory on the way out, and a
 * refused save cannot greet the next open.
 *
 * Which fields it renders is `shape` and nothing else, so this component never
 * asks what kind of deployment it is on. See `passwordForm.ts`.
 */
export function PasswordDialog({
  isOpen,
  onOpenChange,
  shape,
  onSaved,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  shape: PasswordFormShape
  /** Called with what landed, so the card can report it once this closes. */
  onSaved: (outcome: { email: string; claimed: boolean }) => void
}) {
  const retireMasterKeySignIn = useRetireMasterKeySignIn()
  const setPassword = useSetPassword()
  const [email, setEmail] = useState("")
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const { needsEmail, needsCurrentPassword, claimsDeployment } = shape
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
          onSaved({ email: result.email, claimed: claimsDeployment })
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
      description={passwordDialogDescription(shape)}
      submitLabel={passwordAction(shape)}
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
