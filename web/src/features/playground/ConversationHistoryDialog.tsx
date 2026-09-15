import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import type { PlaygroundConversation } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Dialog } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { formatDateTime } from "@/shared/helpers/format"

/**
 * The caller's own saved transcripts. Press a row to load it, or delete it.
 *
 * Each row is a plain button with the delete beside it rather than a selectable
 * list item, because the two actions have to be distinguishable: in a react-aria
 * collection a press anywhere in a row selects it, so a delete inside one loads
 * the transcript as well as asking to remove it.
 *
 * Deleting is confirmed, like every other destructive action in the product, and
 * the confirmation names the transcript: "delete this conversation" over a list
 * of ten is not enough to tell somebody which one they pressed.
 */
export function ConversationHistoryDialog({
  isOpen,
  onOpenChange,
  conversations,
  onLoad,
  onDelete,
  isDeleting,
  deleteError,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  conversations: readonly PlaygroundConversation[]
  onLoad: (conversationId: string) => void
  onDelete: (conversationId: string) => void
  isDeleting: boolean
  /** The last delete's failure, shown in the confirm dialog. */
  deleteError?: unknown
}) {
  const [pendingDelete, setPendingDelete] = useState<
    PlaygroundConversation | undefined
  >(undefined)

  return (
    <>
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        title="Conversation history"
        description="Transcripts you saved in this workspace. Only you can see them."
        actions={<Button onPress={() => onOpenChange(false)}>Close</Button>}
      >
        {conversations.length === 0 ? (
          <EmptyMessage>No saved conversations yet.</EmptyMessage>
        ) : (
          <ul className="flex flex-col">
            {conversations.map((conversation) => (
              <li
                key={conversation.id}
                className="flex items-center gap-2 border-border-subtle border-b last:border-b-0"
              >
                <button
                  type="button"
                  className="min-h-11 min-w-0 flex-1 rounded-md px-2 py-3 text-left transition-colors hover:bg-surface-subtle"
                  onClick={() => onLoad(conversation.id)}
                >
                  <span className="block truncate text-emphasis">
                    {conversation.title}
                  </span>
                  <span className="block truncate text-caption">
                    {conversation.model} · {conversation.message_count} turns ·{" "}
                    {formatDateTime(conversation.created_at)}
                  </span>
                </button>
                <IconButton
                  label={`Delete conversation "${conversation.title}"`}
                  variant="danger"
                  onPress={() => setPendingDelete(conversation)}
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
        heading="Delete this conversation?"
        body={
          pendingDelete
            ? `This permanently deletes "${pendingDelete.title}". It cannot be undone.`
            : ""
        }
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
