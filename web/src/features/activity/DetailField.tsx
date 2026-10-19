import type { ReactNode } from "react"

import { CopyButton } from "@/design-system/actions/CopyButton"
import { CopyableValue } from "@/design-system/actions/CopyField"

// `copyValue` adds a copy control for the fields that hold an opaque identifier
// (a request id, an api key id): they are what an operator pastes into a log
// search or a support thread, and a mistyped character makes them useless.
// An id (`isId`, set in mono) keeps to one line, ellipsized with the whole of it
// as its title. A name with a copy control wraps like any other value.
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
  const valueClass = isId ? "text-mono-caption" : "text-body wrap-anywhere"
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-overline">{label}</span>
      {copyValue && !isId ? (
        <span className={valueClass}>
          {children}{" "}
          {/* Inline after the last word, set off by that word space, its 32px
              pulled into the 20px line so a one-line name keeps the height of
              its neighbors. */}
          <span className="-my-1.5 inline-flex align-middle">
            <CopyButton
              value={copyValue}
              label={copyLabel ?? label.toLowerCase()}
            />
          </span>
        </span>
      ) : copyValue ? (
        // A 20px line like every other value, so the 32px copy button overhangs
        // it instead of making this field taller than its neighbors.
        <span className="flex h-5 min-w-0 items-center">
          <CopyableValue
            value={copyValue}
            label={copyLabel ?? label.toLowerCase()}
            className="text-mono-caption"
            isTruncated
          >
            <span title={typeof children === "string" ? children : copyValue}>
              {children}
            </span>
          </CopyableValue>
        </span>
      ) : (
        <span className={valueClass}>{children}</span>
      )}
    </div>
  )
}
