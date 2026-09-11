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
  children,
}: {
  label: ReactNode
  help?: ReactNode
  /** A count or a status, to the left of the chevron. */
  trailing?: ReactNode
  isOpen: boolean
  onToggle: () => void
  children: ReactNode
}) {
  const panelId = useId()
  return (
    <>
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={onToggle}
        className="flex min-h-11 w-full items-center gap-6 px-4 py-3 text-left transition-colors duration-150 ease-out hover:bg-surface-subtle focus-visible:otari-focus-ring motion-reduce:transition-none"
      >
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-emphasis">{label}</span>
          {help ? (
            <span className="text-caption text-subtle">{help}</span>
          ) : null}
        </span>
        <span className="flex shrink-0 items-center gap-3">
          {trailing}
          <FiChevronRight
            aria-hidden="true"
            className={`h-4 w-4 text-subtle transition-transform duration-150 ease-out motion-reduce:transition-none ${
              isOpen ? "rotate-90" : ""
            }`}
          />
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
