import { Dropdown, Tooltip as HeroTooltip } from "@heroui/react"
import {
  Children,
  createContext,
  Fragment,
  isValidElement,
  type ReactNode,
  type RefObject,
  use,
  useRef,
  useState,
} from "react"
import type { IconType } from "react-icons"
import { FiMoreHorizontal } from "react-icons/fi"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import {
  TOOLTIP_OPEN_DELAY_MS,
  Tooltip,
  type TooltipTriggerProps,
} from "@/design-system/overlays/Tooltip"
import { withRefusals } from "./RowMenu"
import { RowMenuItemContent } from "./RowMenuItemContent"

export type RowActionProps = {
  onPress: () => void
  /** The armed half of a confirm, alone on a row. Never at rest. */
  isDanger?: boolean
  /**
   * In a row's menu, an action that is hard to undo (delete, remove, revoke):
   * its item wears the danger ink and is listed after a rule. The menu is
   * closed until somebody opens it, so this is never ink on the row at rest.
   */
  isDestructive?: boolean
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
function InlineRowAction({
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

/** A confirmation a menu item asks for, shown once the menu has closed. */
export interface RowConfirmation {
  heading: string
  confirmLabel: string
  /** A confirm that is hard to undo, on the danger button. */
  isDestructive?: boolean
  onConfirm: () => void
}

/**
 * Set inside a `RowActionRow`, where its actions are menu items rather than
 * buttons. `ConfirmRowAction` asks through it.
 */
export const RowMenuContext = createContext<
  ((confirmation: RowConfirmation) => void) | null
>(null)

type ItemProps = Pick<
  RowActionProps,
  "label" | "children" | "ariaLabel" | "isDisabled"
>

function itemText({ label, children, ariaLabel }: ItemProps): string {
  return (
    label ?? (typeof children === "string" ? children : (ariaLabel ?? "Action"))
  )
}

// Why an item is refused. A standalone glyph says it in its accessible name;
// a menu item says it under its label.
function refusal(props: ItemProps): string | null {
  const { isDisabled, ariaLabel } = props
  return isDisabled && ariaLabel && ariaLabel !== itemText(props)
    ? ariaLabel
    : null
}

/**
 * Each refused item in a lane, as it reads: its label, then why. A disabled
 * item takes no focus, so the trigger's name is where a keyboard reaches the
 * reason.
 */
function refusals(children: ReactNode): string[] {
  return Children.toArray(children).flatMap((part) => {
    if (!isValidElement<ItemProps>(part)) return []
    if (part.type === Fragment) return refusals(part.props.children)
    if (part.type !== RowAction) return []
    const reason = refusal(part.props)
    return reason ? [`${itemText(part.props)} ${reason}`] : []
  })
}

/**
 * A row's action. Inside a `RowActionRow`, which is where a table's actions
 * go, it is an item of that row's three-dot menu, spelled out with its glyph
 * and its label; write the label as the menu should show it, with the ellipsis
 * when the action asks for more before anything happens (`Delete…`). Alone, it
 * is a glyph or a word on the row.
 */
export function RowAction(props: RowActionProps) {
  const menu = use(RowMenuContext)
  if (!menu) return <InlineRowAction {...props} />
  const {
    icon: Icon,
    label,
    children,
    ariaLabel,
    isDisabled,
    isDestructive,
  } = props
  const isRed = !isDisabled && Boolean(isDestructive)
  return (
    <Dropdown.Item
      textValue={itemText(props)}
      // The name a caller gives the action, which says which row it acts on.
      aria-label={!isDisabled && ariaLabel ? ariaLabel : undefined}
      isDisabled={isDisabled}
      onAction={props.onPress}
      className={isRed ? "otari-row-menu-danger" : undefined}
    >
      <RowMenuItemContent
        icon={Icon}
        label={label ?? children}
        hint={refusal(props)}
        isRed={isRed}
      />
    </Dropdown.Item>
  )
}

/**
 * A row's actions as one three-dot menu, the one shape every table here uses
 * (`RowMenu` draws the same trigger from data). A note in the lane that is
 * not an action, a plain element rather than a `RowAction`, stays on the row
 * beside the trigger.
 */
export function RowActionRow({ children }: { children: ReactNode }) {
  const trigger = useRef<HTMLButtonElement>(null)
  const [confirming, setConfirming] = useState<RowConfirmation | null>(null)
  const parts = Children.toArray(children)
  const notes = parts.filter(
    (part) => isValidElement(part) && typeof part.type === "string",
  )
  const actions = parts.filter((part) => !notes.includes(part))
  return (
    <div className="flex items-center justify-end gap-3">
      {notes}
      {actions.length > 0 ? (
        <Dropdown>
          <HeroTooltip.Root delay={TOOLTIP_OPEN_DELAY_MS}>
            <Dropdown.Trigger
              ref={trigger}
              aria-label={withRefusals("Row actions", refusals(actions))}
              className="otari-row-menu-trigger"
            >
              <FiMoreHorizontal aria-hidden className="size-4 shrink-0" />
            </Dropdown.Trigger>
            <HeroTooltip.Content>Actions</HeroTooltip.Content>
          </HeroTooltip.Root>
          <Dropdown.Popover placement="bottom end" className="otari-row-menu">
            <RowMenuContext value={setConfirming}>
              <Dropdown.Menu
                aria-label="Row actions"
                onAction={() => trigger.current?.focus()}
              >
                {actions}
              </Dropdown.Menu>
            </RowMenuContext>
          </Dropdown.Popover>
        </Dropdown>
      ) : null}
      {confirming ? (
        <ConfirmDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setConfirming(null)
          }}
          heading={`${confirming.heading}?`}
          body="This takes effect as soon as you confirm."
          confirmLabel={confirming.confirmLabel}
          confirmVariant={confirming.isDestructive ? "danger" : "primary"}
          isPending={false}
          onConfirm={() => {
            confirming.onConfirm()
            setConfirming(null)
          }}
        />
      ) : null}
    </div>
  )
}
