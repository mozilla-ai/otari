import type { ReactNode } from "react"

/** One labelled fact in a rule's summary row. */
export function Fact({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-overline">{label}</span>
      <span className="text-body">{children}</span>
    </div>
  )
}
