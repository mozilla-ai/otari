import type { ReactNode } from "react"

/**
 * Where the layout has a place for something no part of Otari records or
 * serves yet: a sentence saying what would be there, instead of a look-alike.
 */
export function NotYet({ children }: { children: ReactNode }) {
  return (
    <p className="border-l-2 border-border py-1 pl-3 text-sm text-muted">
      {children}
    </p>
  )
}
