import type { ReactNode } from "react"

import { CopyableValue } from "@/design-system/actions/CopyField"

// `copyValue` adds a copy control for the fields that hold an opaque identifier
// (a request id, an api key id): they are what an operator pastes into a log
// search or a support thread, and a mistyped character makes them useless.
// Such a value keeps to one line, ellipsized rather than wrapped, with the
// whole of it as its title; `isId` sets it in mono.
export function DetailField({
  label,
  copyValue,
  copyLabel,
  isId = false,
  children,
}: {
  label: string
  copyValue?: string | null
  /** Overrides the copy control's name where the column heading would misname the
      value: "API key" holds a key id, never key material. */
  copyLabel?: string
  isId?: boolean
  children: ReactNode
}) {
  const valueClass = isId ? "text-mono-caption" : "text-body break-all"
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-overline">{label}</span>
      {copyValue ? (
        // A 20px line like every other value, so the 32px copy button overhangs
        // it instead of making this field taller than its neighbors.
        <span className="flex h-5 min-w-0 items-center">
          <CopyableValue
            value={copyValue}
            label={copyLabel ?? label.toLowerCase()}
            className={isId ? "text-mono-caption" : "text-body"}
            isTruncated
          >
            <span title={copyValue}>{children}</span>
          </CopyableValue>
        </span>
      ) : (
        <span className={valueClass}>{children}</span>
      )}
    </div>
  )
}
