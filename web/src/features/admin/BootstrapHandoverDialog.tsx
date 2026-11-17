import { Input, Label, TextField } from "@heroui/react"
import { useState } from "react"

import type { DeploymentUser } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { useProfile } from "@/shared/api/auth"
import { useHandOverBootstrapOperator } from "@/shared/api/deployment"

import { accountLabel } from "./accounts"

/**
 * Hand the bootstrap operator role to another operator.
 *
 * Asks for the caller's own password, as the gateway does from a session: the
 * role is the deployment's recovery point, and only the master key can take it
 * back, so a session alone is not enough to give it away. An account with no
 * password cannot hand over from here at all, so the dialog says so instead.
 * Mount it only while open, keyed on the account, so a typed password does not
 * outlive it.
 */
export function BootstrapHandoverDialog({
  account,
  onClose,
}: {
  /** The account taking the role. */
  account: DeploymentUser
  onClose: () => void
}) {
  const handOver = useHandOverBootstrapOperator()
  // From the profile rather than the organization context, which a caller with
  // no organization does not have. Unknown until it answers, and while it has
  // failed, which the dialog reports with a way to ask again.
  const profile = useProfile()
  const hasPassword = profile.data?.has_password
  const [password, setPassword] = useState("")
  const close = () => {
    handOver.reset()
    onClose()
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) close()
      }}
      title="Bootstrap operator"
      description={
        <>
          Make <strong>{accountLabel(account)}</strong> the account the master
          key acts as.
        </>
      }
      submitLabel="Make bootstrap operator"
      isPending={handOver.isPending}
      error={handOver.error ?? (profile.isError ? profile.error : undefined)}
      isDirty={password !== ""}
      isSubmitDisabled={hasPassword !== true}
      footerStart={
        profile.isError ? (
          <Button
            size="sm"
            isPending={profile.isFetching}
            onPress={() => void profile.refetch()}
          >
            Try again
          </Button>
        ) : undefined
      }
      onSubmit={() => {
        handOver.mutate(
          { id: account.id, currentPassword: password },
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
      {hasPassword === false ? (
        <InfoBanner>
          Your account has no password, so it cannot hand over the role from the
          dashboard. Set one in Account settings, or run{" "}
          <code>otari operator handover</code> with the master key.
        </InfoBanner>
      ) : null}
      {hasPassword ? (
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
