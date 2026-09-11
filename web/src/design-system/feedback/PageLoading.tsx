import { Spinner } from "@heroui/react"

// A full-width loading placeholder for a page (or section) whose content is
// gated on a first fetch. Without it, config pages that render nothing until
// `data` arrives (Settings, Tools & Guardrails, the Overview index) flash a bare
// header over blank space, which reads as broken. `role="status"` announces the
// wait (and its label) to assistive tech.
export function PageLoading({ label = "Loading…" }: { label?: string }) {
  return (
    <div
      role="status"
      className="flex items-center justify-center gap-2 px-4 py-10 text-sm text-muted"
    >
      {/* The spinner is its own role="status" live region as of HeroUI 3.2.4,
          which would nest one status inside another and announce a bare
          "Loading" alongside this label. Hide it; the label below carries the
          announcement for the region. */}
      <Spinner size="sm" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
