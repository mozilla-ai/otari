import type { ReactNode } from "react"
import { useId } from "react"
import { FiChevronRight } from "react-icons/fi"

/**
 * A settings row that opens in place, for detail that belongs to the row rather
 * than beside it.
 *
 * The whole row is the button, which is what keeps the target at the row's own
 * size: a chevron is a 16px glyph, and making it the control puts a 16px target
 * in a 44px row. No press transform either, unlike an action button: a row is a
 * surface, and a surface that shrinks under the finger reads as a misfire.
 *
 * Two children of a `SettingsGroup`, not one, so the group's own divider draws
 * the line between the row and the panel it opened.
 */
export function DisclosureRow({
  label,
  help,
  trailing,
  isOpen,
  onToggle,
  chevronPlacement = "end",
  children,
}: {
  label: ReactNode
  help?: ReactNode
  /** A count or a status, at the row's end. */
  trailing?: ReactNode
  isOpen: boolean
  onToggle: () => void
  /**
   * `start` puts the chevron before the label, as a tree does, for a row whose
   * panel holds rows like the ones around it: the open state then reads where
   * the eye starts the row rather than at its far end.
   */
  chevronPlacement?: "start" | "end"
  children: ReactNode
}) {
  const panelId = useId()
  const chevron = (
    <FiChevronRight
      aria-hidden="true"
      className={`h-4 w-4 shrink-0 text-subtle transition-transform duration-150 ease-out motion-reduce:transition-none ${
        isOpen ? "rotate-90" : ""
      }`}
    />
  )
  return (
    <>
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={onToggle}
        className={`flex min-h-11 w-full items-center px-4 py-3 text-left transition-colors duration-150 ease-out hover:bg-surface-subtle focus-visible:otari-focus-ring motion-reduce:transition-none max-sm:flex-wrap max-sm:gap-y-1 ${
          chevronPlacement === "start" ? "gap-x-2" : "gap-x-6"
        }`}
      >
        {chevronPlacement === "start" ? chevron : null}
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-emphasis">{label}</span>
          {help ? (
            <span className="text-caption text-subtle">{help}</span>
          ) : null}
        </span>
        {/* On a phone the status takes its own line under the help, with the
            chevron kept on the label's line, so the label keeps its width.
            `contents` hands both to the button's wrapping row there without
            moving either in the DOM. */}
        <span className="flex shrink-0 items-center gap-3 max-sm:contents">
          {trailing ? (
            <span
              className={`flex items-center max-sm:order-last max-sm:basis-full ${
                chevronPlacement === "start" ? "max-sm:pl-6" : ""
              }`}
            >
              {trailing}
            </span>
          ) : null}
          {chevronPlacement === "end" ? chevron : null}
        </span>
      </button>
      {/* Rendered whether or not it is open, so `aria-controls` above always
          names an element that exists. `hidden` rather than unmounting: an
          attribute pointing at nothing is worse than a hidden panel, and the
          rows inside cost nothing when the browser is not painting them. */}
      <div id={panelId} hidden={!isOpen}>
        {children}
      </div>
    </>
  )
}
