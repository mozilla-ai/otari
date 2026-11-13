import { FiUser, FiUsers } from "react-icons/fi"

import type { McpCredentialMode } from "@/client"
import { CREDENTIAL_MODE_COPY } from "@/features/tools/mcpServerForm"

const ICON = { shared: FiUsers, per_user: FiUser } as const

/**
 * "Shared" or "Personal", the one way a server's credential mode is shown: an
 * icon and the word, in the same neutral ink for both. The mode is a kind and
 * not a state, so it carries no dot: a dot beside a server means its health.
 */
export function McpModeLabel({
  mode,
  isCaps = false,
  isLong = false,
}: {
  mode: McpCredentialMode
  /** The page heading's uppercase form. */
  isCaps?: boolean
  /** "Shared account" and "Personal accounts", where the one word alone would not say what is shared. */
  isLong?: boolean
}) {
  const Icon = ICON[mode]
  const copy = CREDENTIAL_MODE_COPY[mode]
  const label = isLong ? copy.longLabel : copy.label
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap ${isCaps ? "text-mono-caption text-muted" : "text-sm"}`}
    >
      <Icon aria-hidden className="size-3.5 shrink-0 text-muted" />
      {isCaps ? label.toUpperCase() : label}
    </span>
  )
}
