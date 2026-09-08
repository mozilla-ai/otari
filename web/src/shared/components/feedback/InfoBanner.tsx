import type { ReactNode } from "react"
import { Dot } from "@/shared/components/indicators/Dot"

export function InfoBanner({
  tone = "info",
  children,
}: {
  tone?: "info" | "warning"
  children: ReactNode
}) {
  // A fact stated between rules, not a tinted box. An informational banner here
  // is almost always a ceiling ("this deployment has no sandbox", "an admin sets
  // this"), which is a fact about the deployment rather than a problem with it,
  // so it reads on the muted rung behind a subtle dot. A caution keeps the
  // danger dot and the same muted prose: the dot says "worth noticing" and the
  // words say what.
  return (
    <div className="flex items-start gap-3 border-y border-border py-3 text-sm text-muted">
      <Dot
        className={`mt-2 ${tone === "warning" ? "bg-danger" : "bg-text-subtle"}`}
      />
      {/* Capped for the reason the page header and the settings rows are: a
          banner is a full-bleed row now, and its sentence would otherwise run
          the width of the page, which is roughly twice a readable measure on a
          wide viewport. The rule spans the page; the words do not. */}
      <div className="min-w-0 max-w-prose">{children}</div>
    </div>
  )
}
