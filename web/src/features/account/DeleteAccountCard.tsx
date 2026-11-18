import { Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Section } from "@/design-system/layout/Section"
import { useAuth } from "@/features/auth/AuthContext"
import { useDeleteAccount, useSessionCaller } from "@/shared/api/auth"

/**
 * Deleting the signed-in account, for good.
 *
 * The confirmation is typed rather than clicked, so a stray click cannot delete
 * the account: its address, and its password where it has one. The server refuses the last owner of an organization other
 * people use, and that refusal is shown as it comes back.
 */
export function DeleteAccountCard() {
  const caller = useSessionCaller()
  const { logout } = useAuth()
  const deleteAccount = useDeleteAccount()
  const [isOpen, setIsOpen] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  if (!caller?.email) {
    return null
  }
  const accountEmail = caller.email
  const needsPassword = caller.has_password
  const isConfirmed =
    email.trim().toLowerCase() === accountEmail.toLowerCase() &&
    (!needsPassword || password !== "")

  const open = () => {
    setEmail("")
    setPassword("")
    deleteAccount.reset()
    setIsOpen(true)
  }

  const confirm = () => {
    if (!isConfirmed || deleteAccount.isPending) {
      return
    }
    deleteAccount.mutate(
      {
        confirm_email: email.trim(),
        ...(needsPassword ? { current_password: password } : {}),
      },
      { onSuccess: () => void logout() },
    )
  }

  return (
    <Section
      aria-labelledby="account-delete-title"
      className="border-t border-border pt-6 pb-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="account-delete-title" className="text-title">
        Delete account
      </h2>
      <div>
        <Button variant="danger" onPress={open}>
          Delete account
        </Button>
      </div>

      <ConfirmDialog
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        heading="Delete your account?"
        confirmLabel="Delete account"
        isPending={deleteAccount.isPending}
        isConfirmDisabled={!isConfirmed}
        error={deleteAccount.error}
        onConfirm={confirm}
        body={
          <>
            <p className="text-sm">
              This signs you out and permanently deletes your sign-in, keys,
              files, traces and Playground conversations. Usage stays in your
              organizations&rsquo; totals without your name. An organization
              only you use is renamed, and its keys and stored credentials are
              removed. This cannot be undone.
            </p>
            <TextField
              value={email}
              onChange={setEmail}
              type="email"
              isRequired
              className="flex flex-col gap-1"
            >
              <Label className="text-body">
                Type {accountEmail} to confirm
              </Label>
              <Input autoComplete="off" />
            </TextField>
            {needsPassword ? (
              <TextField
                value={password}
                onChange={setPassword}
                type="password"
                isRequired
                className="flex flex-col gap-1"
              >
                <Label className="text-body">Password</Label>
                <Input autoComplete="current-password" />
              </TextField>
            ) : null}
          </>
        }
      />
    </Section>
  )
}
