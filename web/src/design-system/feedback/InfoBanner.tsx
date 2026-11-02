import type { ReactNode } from "react"
import { FiX } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { Dot } from "@/design-system/indicators/Dot"

export function InfoBanner({
  tone = "info",
  onDismiss,
  dismissLabel = "Dismiss",
  children,
}: {
  tone?: "info" | "warning"
  /**
   * Offers a close control, for the one kind of banner the reader may put away:
   * a line that is not a fact about the deployment and is therefore not
   * something that stays true until fixed. A standing condition takes none,
   * because hiding a ceiling does not lift it. The caller owns what dismissal
   * means, including remembering it.
   */
  onDismiss?: () => void
  /** The close control's accessible name. Says what pressing it does. */
  dismissLabel?: string
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
      {onDismiss ? (
        // Pushed to the row's far end, so it sits where a close control sits
        // everywhere else in the product rather than trailing the sentence.
        // Dialog's close geometry: a small square whose pseudo-element widens
        // the target to the 44px floor without making the band taller than its
        // line of prose, and a negative margin that centers it on that line.
        <Button
          aria-label={dismissLabel}
          isIconOnly
          size="sm"
          className="relative -my-1 ml-auto shrink-0 before:absolute before:-inset-1.5 before:content-['']"
          onPress={onDismiss}
        >
          <FiX aria-hidden className="size-4 text-muted" />
        </Button>
      ) : null}
    </div>
  )
}
