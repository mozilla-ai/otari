import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { type LimitTarget, useLimitWriter } from "@/features/spend/limits"

/** Removes a ceiling the page can write, through the limit writer's remove. */
export function RemoveLimitDialog({
  name,
  target,
  onClose,
}: {
  name: string
  target: LimitTarget
  onClose: () => void
}) {
  const writer = useLimitWriter()
  return (
    <ConfirmDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      heading={`Remove ${name}?`}
      body="Requests will no longer be limited by it."
      confirmLabel="Remove limit"
      isPending={writer.isPending}
      error={writer.error}
      onConfirm={() => {
        void writer.remove(target).then((isDone) => {
          if (isDone) onClose()
        })
      }}
    />
  )
}
