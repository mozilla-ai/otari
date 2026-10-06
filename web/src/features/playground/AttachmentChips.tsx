import { FiPaperclip } from "react-icons/fi"

import { DismissChip } from "@/design-system/indicators/DismissChip"
import { formatFileSize } from "@/shared/helpers/format"

import type { PendingAttachment } from "./hooks/usePlayground"

function AttachmentLabel({
  filename,
  detail,
}: {
  filename: string
  detail: string
}) {
  return (
    <>
      <FiPaperclip aria-hidden className="size-4 shrink-0" />
      <span className="min-w-0 truncate">{filename}</span>
      <span className="shrink-0 text-subtle">{detail}</span>
    </>
  )
}

/** The files on the next question, each removable. Renders nothing when there are none. */
export function PendingAttachmentChips({
  attachments,
  onRemove,
}: {
  attachments: readonly PendingAttachment[]
  onRemove: (key: string) => void
}) {
  if (attachments.length === 0) return null
  return (
    <ul
      aria-label="Attached files"
      className="flex flex-wrap items-center gap-2"
    >
      {attachments.map((attachment) => (
        <li key={attachment.key} className="min-w-0">
          <DismissChip
            value={
              <AttachmentLabel
                filename={attachment.filename}
                detail={
                  attachment.status === "uploading"
                    ? "Uploading…"
                    : formatFileSize(attachment.bytes)
                }
              />
            }
            dismissLabel={`Remove ${attachment.filename}`}
            onDismiss={() => onRemove(attachment.key)}
          />
        </li>
      ))}
    </ul>
  )
}

/** The files a sent question carried. */
export function SentAttachmentList({
  attachments,
}: {
  attachments: readonly { filename: string; bytes: number }[]
}) {
  return (
    <ul
      aria-label="Attached files"
      className="flex flex-wrap justify-end gap-x-4 gap-y-1 text-mono-caption text-foreground"
    >
      {attachments.map((attachment, index) => (
        <li
          // A question can carry one file twice, so the position disambiguates.
          key={`${attachment.filename}-${index}`}
          className="inline-flex min-w-0 items-center gap-1.5"
        >
          <AttachmentLabel
            filename={attachment.filename}
            detail={formatFileSize(attachment.bytes)}
          />
        </li>
      ))}
    </ul>
  )
}
