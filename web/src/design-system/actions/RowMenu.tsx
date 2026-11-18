import { Dropdown, Tooltip } from "@heroui/react"
import { type ReactNode, useRef } from "react"
import type { IconType } from "react-icons"
import { FiMoreHorizontal } from "react-icons/fi"
import { TOOLTIP_OPEN_DELAY_MS } from "../overlays/Tooltip"
import { RowMenuItemContent } from "./RowMenuItemContent"

export interface RowMenuAction {
  id: string
  /** What the action does, in full: "Set a spend limit…", not a glyph. */
  label: string
  /** The consequence or the current state, under the label. */
  hint?: string
  icon: IconType
  isDisabled?: boolean
  /** Hard to undo: revoke, remove, delete. Listed last, in the danger ink. */
  isDestructive?: boolean
  run: () => void
}

/**
 * The trigger's accessible name with each refused action's reason after it:
 * "Actions for Wei Zhang. Remove… This is the last active owner." A disabled
 * menu item takes no focus, so a keyboard never reaches what is written under
 * it; the trigger is where it reads.
 */
export function withRefusals(name: string, refused: readonly string[]): string {
  if (refused.length === 0) return name
  return `${[name, ...refused].map((part) => part.replace(/\.$/, "")).join(". ")}.`
}

/**
 * A table row's actions behind one three-dot control, each spelled out with
 * what it does. Every table's row actions open from this trigger: a lane of
 * `RowAction`s in a `RowActionRow` renders it too, so the two shapes cannot
 * drift apart.
 */
export function RowMenu({
  label,
  actions,
  head,
  foot,
}: {
  /** Names the row for assistive technology: "Actions for Wei Zhang". */
  label: string
  actions: readonly RowMenuAction[]
  /** What the row is, above its actions. */
  head?: ReactNode
  /** A line under the actions that is not one, such as why there are no more. */
  foot?: ReactNode
}) {
  const trigger = useRef<HTMLButtonElement>(null)
  if (actions.length === 0) return null
  const refused = actions
    .filter((action) => action.isDisabled && action.hint)
    .map((action) => `${action.label} ${action.hint}`)
  return (
    <Dropdown>
      <Tooltip.Root delay={TOOLTIP_OPEN_DELAY_MS}>
        <Dropdown.Trigger
          ref={trigger}
          aria-label={withRefusals(label, refused)}
          className="otari-row-menu-trigger"
        >
          <FiMoreHorizontal aria-hidden className="size-4 shrink-0" />
        </Dropdown.Trigger>
        <Tooltip.Content>{label}</Tooltip.Content>
      </Tooltip.Root>
      <Dropdown.Popover placement="bottom end" className="otari-row-menu">
        {head}
        <Dropdown.Menu
          aria-label={label}
          items={actions}
          onAction={(id) => {
            trigger.current?.focus()
            actions.find((action) => action.id === id)?.run()
          }}
        >
          {(action) => (
            <Dropdown.Item
              id={action.id}
              textValue={action.label}
              isDisabled={action.isDisabled}
              className={
                action.isDestructive ? "otari-row-menu-danger" : undefined
              }
            >
              <RowMenuItemContent
                icon={action.icon}
                label={action.label}
                hint={action.hint}
                isRed={Boolean(action.isDestructive) && !action.isDisabled}
              />
            </Dropdown.Item>
          )}
        </Dropdown.Menu>
        {foot}
      </Dropdown.Popover>
    </Dropdown>
  )
}
