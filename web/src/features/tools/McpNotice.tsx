import type { ReactNode } from "react"
import { FiX } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Dot } from "@/design-system/indicators/Dot"

export type McpNoticeTone = "info" | "warning" | "success"

const DOT: Record<McpNoticeTone, string> = {
  info: "bg-text-subtle",
  warning: "bg-warning",
  success: "bg-success",
}

/**
 * One quiet line on an MCP page for what the page itself does not show: a
 * check that could not run, a sign-in that failed or was canceled, an admin's
 * answer to a request. Always dismissible, with at most one action beside it.
 */
export function McpNotice({
  tone = "warning",
  message,
  action,
  onDismiss,
}: {
  tone?: McpNoticeTone
  message: string
  /** The one thing to do about it, as a small button. */
  action?: ReactNode
  onDismiss: () => void
}) {
  return (
    <div
      role="status"
      className="flex items-center gap-3 border-y border-border py-1 text-sm text-muted"
    >
      <Dot className={DOT[tone]} />
      <p className="min-w-0 flex-1 break-words">{message}</p>
      {action ? <span className="shrink-0">{action}</span> : null}
      <IconButton label="Dismiss" onPress={onDismiss}>
        <FiX aria-hidden className="size-4" />
      </IconButton>
    </div>
  )
}
