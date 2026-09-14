import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import type { PlaygroundComparison } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Dialog } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { formatDateTime } from "@/shared/helpers/format"

/** Which model a row's reader preferred, in the words the page uses. */
const PREFERENCE_LABEL: Record<string, string> = {
  model_a: "A preferred",
  model_b: "B preferred",
  tie: "Tie",
}

/**
 * The caller's own rated comparisons, with a delete on each.
 *
 * No load action, unlike the transcript history next door, and that asymmetry is
 * the point: a comparison is a judgment recorded at a moment, not a conversation
 * to resume, so there is nothing to put back on screen. The answers it stored
 * are not shown either, and the API does not send them: a dozen rows carrying
 * two full answers each would move megabytes to draw a list.
 */
export function ComparisonHistoryDialog({
  isOpen,
  onOpenChange,
  comparisons,
  onDelete,
  isDeleting,
  deleteError,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  comparisons: readonly PlaygroundComparison[]
  onDelete: (comparisonId: string) => void
  isDeleting: boolean
  /** The last delete's failure, shown in the confirm dialog. */
  deleteError?: unknown
}) {
  const [pendingDelete, setPendingDelete] = useState<
    PlaygroundComparison | undefined
  >(undefined)

  return (
    <>
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        title="Saved comparisons"
        description="Ratings you recorded in this workspace. Only you can see them."
        actions={<Button onPress={() => onOpenChange(false)}>Close</Button>}
      >
        {comparisons.length === 0 ? (
          <EmptyMessage>No saved comparisons yet.</EmptyMessage>
        ) : (
          <ul className="flex flex-col">
            {comparisons.map((comparison) => (
              <li
                key={comparison.id}
                className="flex items-center gap-2 border-border-subtle border-b py-3 last:border-b-0"
              >
                <div className="min-w-0 flex-1 px-2">
                  <span className="block truncate text-emphasis">
                    {comparison.user_question}
                  </span>
                  <span className="block truncate text-caption">
                    {comparison.model_a} vs {comparison.model_b} ·{" "}
                    {PREFERENCE_LABEL[comparison.preference] ??
                      comparison.preference}{" "}
                    · {formatDateTime(comparison.created_at)}
                  </span>
                </div>
                <IconButton
                  label="Delete comparison"
                  variant="danger"
                  onPress={() => setPendingDelete(comparison)}
                  isDisabled={isDeleting}
                >
                  <FiTrash2 aria-hidden className="size-4" />
                </IconButton>
              </li>
            ))}
          </ul>
        )}
      </Dialog>

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(next) => {
          if (!next) setPendingDelete(undefined)
        }}
        heading="Delete this comparison?"
        body="This permanently deletes the saved rating and both answers. It cannot be undone."
        confirmLabel="Delete"
        isPending={isDeleting}
        error={deleteError}
        onConfirm={() => {
          if (!pendingDelete) return
          onDelete(pendingDelete.id)
          setPendingDelete(undefined)
        }}
      />
    </>
  )
}
