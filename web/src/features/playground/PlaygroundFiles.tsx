import { useState } from "react"
import { FiFile, FiFolder, FiPaperclip, FiTrash2 } from "react-icons/fi"

import type { PlaygroundFile } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { formatDate, formatFileSize } from "@/shared/helpers/format"

/**
 * The caller's uploads in this workspace: attach one to the next question
 * again, or delete it.
 *
 * Uploads outlive the chat that made them, so this is where they are managed.
 * Removing a chip from the composer only takes a file off one question.
 */
export function PlaygroundFiles({
  isOpen,
  onOpenChange,
  files,
  isLoading,
  hasMore,
  isLoadingMore,
  onLoadMore,
  error,
  attachedIds,
  canAttach,
  onAttach,
  onDelete,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  files: readonly PlaygroundFile[]
  isLoading: boolean
  /** Whether older uploads remain past the ones listed. */
  hasMore: boolean
  isLoadingMore: boolean
  onLoadMore: () => void
  error?: unknown
  /** Files already on the next question, which are not offered twice. */
  attachedIds: ReadonlySet<string>
  /** False while a reply is streaming or no model is chosen. */
  canAttach: boolean
  onAttach: (file: PlaygroundFile) => void
  onDelete: (id: string) => Promise<void>
}) {
  const [pendingDelete, setPendingDelete] = useState<PlaygroundFile>()
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<unknown>()

  return (
    <>
      <Button
        className="min-h-11 shrink-0 md:min-h-9"
        onPress={() => onOpenChange(true)}
      >
        <FiFolder aria-hidden className="size-4" /> Files
      </Button>
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        title="Files"
        description="Files you uploaded in this workspace. Only you can see them."
        size="lg"
        actions={<Button onPress={() => onOpenChange(false)}>Close</Button>}
      >
        <DialogSection>
          <div className="max-h-[min(28rem,55dvh)] overflow-y-auto">
            {error ? (
              <ErrorBanner error={error} />
            ) : isLoading ? (
              <p role="status" className="p-4 text-caption">
                Loading files…
              </p>
            ) : files.length === 0 ? (
              <p className="p-4 text-caption">No uploaded files yet.</p>
            ) : (
              <ul>
                {files.map((file) => (
                  <li
                    key={file.id}
                    className="flex items-center gap-2 border-t border-border-subtle px-4 py-2 first:border-t-0"
                  >
                    <FiFile
                      aria-hidden
                      className="size-4 shrink-0 text-muted"
                    />
                    <div className="flex min-h-11 min-w-0 flex-1 flex-col justify-center gap-0.5">
                      <span
                        className="w-full truncate text-emphasis"
                        title={file.filename}
                      >
                        {file.filename}
                      </span>
                      <span className="w-full truncate text-caption text-subtle">
                        {formatFileSize(file.bytes)} ·{" "}
                        {formatDate(
                          new Date(file.created_at * 1000).toISOString(),
                        )}
                      </span>
                    </div>
                    <Button
                      size="sm"
                      className="shrink-0"
                      aria-label={`Attach ${file.filename}`}
                      isDisabled={!canAttach || attachedIds.has(file.id)}
                      onPress={() => onAttach(file)}
                    >
                      <FiPaperclip aria-hidden className="size-3.5" />
                      {attachedIds.has(file.id) ? "Attached" : "Attach"}
                    </Button>
                    <IconButton
                      isIconOnly
                      size="sm"
                      label={`Delete ${file.filename}`}
                      className="shrink-0"
                      onPress={() => {
                        setDeleteError(undefined)
                        setPendingDelete(file)
                      }}
                    >
                      <FiTrash2 aria-hidden className="size-3.5" />
                    </IconButton>
                  </li>
                ))}
              </ul>
            )}
            {hasMore && !error ? (
              <div className="border-t border-border-subtle px-4 py-2">
                <Button
                  size="sm"
                  isPending={isLoadingMore}
                  onPress={onLoadMore}
                >
                  Load more files
                </Button>
              </div>
            ) : null}
          </div>
        </DialogSection>
      </Dialog>
      <ConfirmDialog
        isOpen={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open && !isDeleting) {
            setPendingDelete(undefined)
            setDeleteError(undefined)
          }
        }}
        heading="Delete this file?"
        body={`This permanently deletes "${pendingDelete?.filename ?? ""}". Saved conversations still list it, but cannot send it again.`}
        confirmLabel="Delete permanently"
        isPending={isDeleting}
        error={deleteError}
        onConfirm={() => {
          if (!pendingDelete) return
          setIsDeleting(true)
          void onDelete(pendingDelete.id)
            .then(() => setPendingDelete(undefined))
            .catch(setDeleteError)
            .finally(() => setIsDeleting(false))
        }}
      />
    </>
  )
}
