import { Description, Label } from "@heroui/react"
import type { ReactNode } from "react"
import type { IconType } from "react-icons"

/**
 * A row menu item's glyph, its label and the line under it, in the danger ink
 * when it is hard to undo. The line sits under the label rather than beside
 * it: a sentence saying what an action does, or why it is refused, does not
 * fit on the label's line at the menu's width.
 */
export function RowMenuItemContent({
  icon: Icon,
  label,
  hint,
  isRed,
}: {
  icon: IconType | undefined
  label: ReactNode
  hint: ReactNode
  isRed: boolean
}) {
  return (
    <>
      {Icon ? (
        <Icon
          aria-hidden
          className={`mt-0.5 size-3.5 shrink-0 ${isRed ? "text-danger" : ""}`}
        />
      ) : null}
      <span className="flex min-w-0 flex-col gap-0.5">
        <Label
          className={`text-body ${isRed ? "text-danger" : "text-foreground"}`}
        >
          {label}
        </Label>
        {hint ? (
          <Description className="text-caption">{hint}</Description>
        ) : null}
      </span>
    </>
  )
}
