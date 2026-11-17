import { Input, Label, TextField } from "@heroui/react"
import { useState } from "react"

import type { DeploymentUser } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { useHandOverBootstrapOperator } from "@/shared/api/deployment"
import { useOrganizationContext } from "@/shared/api/organizations"

import { accountLabel } from "./accounts"

/**
 * Hand the bootstrap operator role to another operator.
 *
 * Asks for the caller's own password when their account has one, as the
 * gateway does from a session: the role is the deployment's recovery point,
 * and only the master key can take it back, so a session alone is not enough
 * to give it away. Mount it keyed on the account, so a typed password does not
 * outlive the dialog.
 */
export function BootstrapHandoverDialog({
  account,
  onClose,
}: {
  /** The account taking the role; the dialog is closed while unset. */
  account: DeploymentUser | undefined
  onClose: () => void
}) {
  const handOver = useHandOverBootstrapOperator()
  const needsPassword =
    useOrganizationContext().data?.caller?.has_password ?? false
  const [password, setPassword] = useState("")
  const close = () => {
    handOver.reset()
    onClose()
  }

  return (
    <FormDialog
      isOpen={account !== undefined}
      onOpenChange={(open) => {
        if (!open) close()
      }}
      title="Bootstrap operator"
      description={
        account ? (
          <>
            Make <strong>{accountLabel(account)}</strong> the account the master
            key acts as.
          </>
        ) : null
      }
      submitLabel="Make bootstrap operator"
      isPending={handOver.isPending}
      error={handOver.error}
      isDirty={password !== ""}
      onSubmit={() => {
        if (!account) return
        handOver.mutate(
          { id: account.id, currentPassword: needsPassword ? password : null },
          { onSuccess: close },
        )
      }}
    >
      <p className="text-body">
        Their account becomes the one that cannot be deactivated or lose
        operator access, and only they or the master key can hand the role on.
        You stay an operator, which they can remove.
      </p>
      <p className="text-body text-muted">
        The master key keeps working and acts as their account from now on,
        including resetting its password. If anyone who should no longer have it
        holds the master key, rotate it in Settings.
      </p>
      {needsPassword ? (
        <TextField
          value={password}
          onChange={setPassword}
          type="password"
          isRequired
          className="flex flex-col gap-1"
        >
          <Label className="text-body">Your current password</Label>
          <Input autoComplete="current-password" />
        </TextField>
      ) : null}
    </FormDialog>
  )
}
