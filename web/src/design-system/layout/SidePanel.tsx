import { Drawer } from "@heroui/react"
import { type ReactNode, useState } from "react"
import {
  FiChevronDown,
  FiChevronUp,
  FiMaximize2,
  FiMinimize2,
  FiX,
} from "react-icons/fi"

import { IconButton } from "../actions/IconButton"

/**
 * A record opened over the page from its right edge, for a list whose records
 * are read in depth one after another: an agent session, a request.
 *
 * Over the page rather than beside it, so the list keeps its full width while
 * nothing is open, and the open record gets most of the screen rather than the
 * part a split view leaves. The page stays visible at the left edge, undimmed,
 * so it is clear where the panel came from; pressing there closes it, as Escape
 * does. Previous and next walk the list without closing, which is what keeps
 * reading several records from becoming open, close, open.
 *
 * It takes 60% of the viewport, enough for a tree beside the step it opens;
 * expand gives it all of it, and below `md` it always has all of it. The widths
 * are `.otari-side-panel` in `design-system.css`, because HeroUI sizes the
 * drawer's dialog with a rule a utility class does not outrank.
 *
 * Presentational: which record is open, and which are before and after it,
 * belong to the caller.
 */
export function SidePanel({
  isOpen,
  onClose,
  label,
  heading,
  onPrevious,
  onNext,
  children,
}: {
  isOpen: boolean
  onClose: () => void
  /** Names the panel, as "Session" names an open session. */
  label: string
  /** The record's identity in the panel's top bar, such as its id and a copy control. */
  heading: ReactNode
  /** Opens the record above this one, or is absent at the top of the list. */
  onPrevious?: () => void
  /** Opens the record below this one, or is absent at the end of the list. */
  onNext?: () => void
  children: ReactNode
}) {
  const [isExpanded, setIsExpanded] = useState(false)
  return (
    <Drawer
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
    >
      {/* Driven from state, so the trigger slot HeroUI expects is filled and
          hidden, as `Dialog` does with its own. */}
      <Drawer.Trigger aria-hidden className="hidden">
        {label}
      </Drawer.Trigger>
      <Drawer.Backdrop variant="transparent" isDismissable>
        <Drawer.Content placement="right">
          <Drawer.Dialog
            aria-label={label}
            className={`otari-side-panel ${
              isExpanded ? "otari-side-panel--expanded" : ""
            } flex h-full flex-col border-border border-l bg-surface p-0 shadow-xl`}
          >
            <header className="flex min-h-14 shrink-0 items-center justify-between gap-2 border-border border-b px-4">
              <div className="flex min-w-0 items-center gap-2">{heading}</div>
              <div className="flex shrink-0 items-center">
                <IconButton
                  label="Previous"
                  isDisabled={!onPrevious}
                  onPress={onPrevious}
                >
                  <FiChevronUp aria-hidden className="size-4" />
                </IconButton>
                <IconButton label="Next" isDisabled={!onNext} onPress={onNext}>
                  <FiChevronDown aria-hidden className="size-4" />
                </IconButton>
                <IconButton
                  label={isExpanded ? "Restore width" : "Expand"}
                  className="hidden md:inline-flex"
                  onPress={() => setIsExpanded((value) => !value)}
                >
                  {isExpanded ? (
                    <FiMinimize2 aria-hidden className="size-4" />
                  ) : (
                    <FiMaximize2 aria-hidden className="size-4" />
                  )}
                </IconButton>
                <IconButton label="Close" onPress={onClose}>
                  <FiX aria-hidden className="size-4" />
                </IconButton>
              </div>
            </header>
            {/* `min-h-0` lets the body scroll inside the panel instead of
                growing it past the viewport. */}
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
              {children}
            </div>
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>
    </Drawer>
  )
}
