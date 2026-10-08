import { type ReactNode, useState } from "react"

import { DisclosureRow } from "@/design-system/navigation/DisclosureRow"

/** Rows most readers never touch, folded under one row at the end of a group. */
export function AdvancedRows({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <DisclosureRow
      label="Advanced"
      isOpen={isOpen}
      onToggle={() => setIsOpen((open) => !open)}
    >
      <div className="flex flex-col divide-y divide-border-subtle">
        {children}
      </div>
    </DisclosureRow>
  )
}
