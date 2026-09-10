import type { ReactNode } from "react"

export function PageHeader({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3">
      <div>
        <h1 className="text-display">{title}</h1>
        {description ? (
          // max-w-prose because this ran to the full container width: 968px at
          // 14px is ~138 characters per line, roughly twice a comfortable
          // measure, and it is the same paragraph on every page.
          <p className="mt-1 max-w-prose text-sm text-muted">{description}</p>
        ) : null}
      </div>
      {/* The primary action sits on its own left-aligned row under the heading,
          so it stays near the sidebar the operator just came from rather than
          across the page at the top right. Wrapped so the button keeps its
          natural size instead of stretching in this flex column. */}
      {action ? <div className="flex flex-wrap gap-2">{action}</div> : null}
    </div>
  )
}
