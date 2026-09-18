import type { ReactNode } from "react"

import { CopyableValue } from "@/design-system/actions/CopyField"

// `copyValue` adds a copy control for the fields that hold an opaque identifier
// (a request id, an api key id): they are what an operator pastes into a log
// search or a support thread, and a mistyped character makes them useless.
export function DetailField({
  label,
  copyValue,
  copyLabel,
  children,
}: {
  label: string
  copyValue?: string | null
  /** Overrides the copy control's name where the column heading would misname the
      value: "API key" holds a key id, never key material. */
  copyLabel?: string
  children: ReactNode
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-overline">{label}</span>
      {copyValue ? (
        <CopyableValue
          value={copyValue}
          label={copyLabel ?? label.toLowerCase()}
          className="text-body break-all"
        >
          {children}
        </CopyableValue>
      ) : (
        <span className="text-body break-all">{children}</span>
      )}
    </div>
  )
}
