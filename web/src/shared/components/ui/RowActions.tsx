import type { ReactNode } from "react"

export interface RowActionsProps {
  children: ReactNode
}

/**
 * Presentational cluster for a table row's trailing action buttons.
 * Purely layout — callers own the buttons and their handlers.
 */
export const RowActions = ({ children }: RowActionsProps) => (
  <div className="flex items-center gap-1 justify-end">{children}</div>
)
