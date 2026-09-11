import type { ReactNode, RefObject } from "react"
import type { IconType } from "react-icons"
import {
  Tooltip,
  type TooltipTriggerProps,
} from "@/design-system/overlays/Tooltip"

export type RowActionProps = {
  onPress: () => void
  isDanger?: boolean
  isDisabled?: boolean
  /**
   * Forwarded to the button so a caller can move focus onto it. The two-step
   * confirm is the only caller: its swap unmounts a focused control, and a ref
   * is the only way to hand the caret to whatever replaced it.
   */
  ref?: RefObject<HTMLButtonElement | null>
  /**
   * Replaces the name the label would give, which is how a row action says
   * which row it acts on and why it is refused: a disabled control takes no
   * focus, so a tooltip reaches a pointer and nothing else, and the reason has
   * to be in the name.
   */
  ariaLabel?: string
} & (
  | {
      /**
       * The glyph this action wears instead of its word. The component sizes it
       * and hides it from assistive tech, so every lane matches and no call
       * site can forget either.
       */
      icon: IconType
      /** What the glyph means: the accessible name and the tooltip. */
      label: string
      children?: never
    }
  | {
      icon?: never
      label?: never
      children: ReactNode
    }
)

/**
 * A row's action: a glyph, or 13px of muted text, with no border, fill or box.
 *
 * A row has three or four of these and boxing each one turned the last lane
 * into a control panel, louder than the row it acts on.
 *
 * **Pass `icon` and `label` wherever a glyph says it.** A lane of words reads as
 * prose until the pointer is already on it, and four of them repeated down a
 * table is four words to re-read per row (otari-ai#2123). `label` is not
 * optional beside an icon: it is the accessible name and the tooltip, so the
 * word is still there for anyone who needs it, and a glyph with no name is
 * anonymous to a screen reader and unreachable by speech input.
 *
 * The text form stays for what no glyph says: the armed half of a confirm,
 * whose whole job is to name the consequence, and a lane whose actions have no
 * convention to borrow.
 *
 * `isDanger` is for the ARMED state and nothing else, which is worth spelling
 * out because it was got wrong at six sites on the first pass: a destructive
 * action at rest is the same muted ink as its neighbors, and the color arrives
 * only once the next click commits something. A row that paints Remove red
 * before anybody has touched it spends the color on a state nothing is in, and
 * by the time it means something the reader has stopped seeing it. Same rule
 * the spend figure follows: the color marks what you are about to do, never
 * the control that offers it.
 *
 * An action whose confirmation is a dialog rather than an inline arm stays
 * muted throughout: the dialog is where the danger lives.
 *
 * Shared once two pages had spelled it. Pair with `RowActionRow`, which sets the
 * 16px between them.
 */
export function RowAction({
  onPress,
  icon: Icon,
  label,
  isDanger,
  isDisabled,
  ariaLabel,
  ref,
  children,
}: RowActionProps) {
  const name = ariaLabel ?? label
  // `trigger` is what the tooltip hands its control: the hover and focus
  // handlers that open it, the `aria-describedby` that names it, and a ref to
  // position it against. Every one of them has to reach this button, so they
  // are spread first and only the props this component owns are written after.
  //
  // Its `className` is the exception and is dropped: `.tooltip__trigger` is
  // HeroUI's presentation for a trigger it expects to have built itself, and
  // two thirds of it are wrong here. It sets `display: inline-block`, which
  // un-centers a glyph in a flex box, and it answers `:focus-visible` with a
  // box-shadow ring after zeroing `outline-style`, which motion-and-access.md
  // forbids in as many words. Nothing in it is behavioral, so nothing is lost.
  const render = ({
    className: _triggerClass,
    ref: triggerRef,
    ...trigger
  }: TooltipTriggerProps = {}) => (
    <button
      {...trigger}
      // Two refs on one element: the caller's, and the trigger's. A callback is
      // the only shape that serves both, and dropping the trigger's would
      // leave the tooltip anchored at the top left of the page.
      ref={(node) => {
        if (ref) ref.current = node
        if (typeof triggerRef === "function") triggerRef(node)
        else if (triggerRef) triggerRef.current = node
      }}
      type="button"
      disabled={isDisabled}
      aria-label={name}
      // A disabled control takes neither hover nor focus, so react-aria's
      // tooltip never opens on one and a refused glyph would be nameless to a
      // pointer. The native attribute is the only tooltip it has left, and it
      // is not set on a pressable action: there the browser's tooltip and the
      // product's would open on the same hover and say the same thing.
      title={Icon !== undefined && isDisabled ? name : undefined}
      onClick={onPress}
      className={`text-caption transition-colors motion-reduce:transition-none disabled:opacity-(--disabled-opacity) ${
        Icon
          ? // 32px of visual, 44px of target, the same pair `CopyButton` uses in
            // these very rows: the pseudo-element is the device
            // motion-and-access.md names for keeping a small glyph reachable,
            // and 6px each way is under half `RowActionRow`'s 16px pitch, so no
            // two of these overlap.
            "relative flex size-8 shrink-0 items-center justify-center before:absolute before:-inset-1.5 before:content-['']"
          : "whitespace-nowrap"
      } ${isDanger ? "text-danger" : "hover:text-foreground"}`}
    >
      {Icon ? <Icon aria-hidden="true" className="h-3.5 w-3.5" /> : children}
    </button>
  )
  if (Icon === undefined) return render()
  // The tooltip is a repetition of the accessible name, which is the one job
  // overlays.md leaves it: it spells out a glyph that is already named rather
  // than saying anything only a pointer would learn. Through the function form,
  // so the trigger's props land on this button rather than on a wrapper the
  // library would also call a button.
  return <Tooltip content={label}>{render}</Tooltip>
}

/** The lane those sit in: right-aligned, 16px apart. */
export function RowActionRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center justify-end gap-4">{children}</div>
}
