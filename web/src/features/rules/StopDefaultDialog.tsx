import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { stopDefaultWords, type TeamDefault } from "@/features/rules/rules"
import { useDeleteWorkspaceBudgetDefault } from "@/shared/api/workspaces"

/**
 * Stops a workspace handing its per-person limit to newcomers. The people
 * already under it keep their ceilings; setting a new amount takes an
 * organization budget, which only owners and admins write.
 */
export function StopDefaultDialog({
  teamDefault,
  peopleCount,
  onClose,
}: {
  teamDefault: TeamDefault
  peopleCount: number
  onClose: () => void
}) {
  const remove = useDeleteWorkspaceBudgetDefault()
  return (
    <ConfirmDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      heading="Stop giving this limit to people who join later?"
      body={stopDefaultWords(peopleCount)}
      confirmLabel="Stop giving this limit"
      isPending={remove.isPending}
      error={remove.error}
      onConfirm={() =>
        remove.mutate(
          {
            workspaceId: teamDefault.workspaceId,
            defaultId: teamDefault.policyId,
          },
          { onSuccess: onClose },
        )
      }
    />
  )
}
