import { Dropdown, Label } from "@heroui/react"
import { FiMoreHorizontal } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Toggle } from "@/design-system/forms/Toggle"
import { Chip } from "@/design-system/indicators/Chip"
import { SettingRow } from "@/design-system/layout/SettingRow"

/** Who settles the upstream bill for what a row serves. */
export type AccessPayer = "own-key" | "deployment" | "workspace"

const PAYER_LABEL: Record<AccessPayer, string> = {
  "own-key": "Your key",
  deployment: "Billed by deployment",
  workspace: "Workspace server",
}

/** A less common action on a row, offered from its menu. */
export interface AccessRowAction {
  id: string
  label: string
  onAction: () => void
}

/**
 * One source a workspace could use, in the grammar every Access section shares.
 *
 * Built on `SettingRow`, so its controls sit in the same lane as the tool cards
 * below it: the switch first, where those cards place theirs, the scope control
 * filling the rest, then a menu slot every row reserves whether or not it has
 * actions. Reset is in that menu, and the Customized mark beside the name
 * says it is there. `note` is for what the row cannot do because of something
 * above the workspace, which greys the name rather than hiding the row.
 *
 * One line: who pays and the note sit beside the name rather than under it, so
 * a list of many providers stays short enough to scan.
 */
export function AccessRow({
  name,
  displayName = name,
  isNested = false,
  hasChevronGutter = false,
  payer,
  isEnabled,
  onToggle,
  scopeLabel = "",
  onOpenScope,
  isCustomized,
  onReset,
  note,
  isMuted = false,
  isDisabled = false,
  actions = [],
}: {
  /** The full name, which every control's accessible name carries. */
  name: string
  /** What the row shows, shorter inside a provider's group ("team"). */
  displayName?: string
  /** Indented under the provider row that groups it. */
  isNested?: boolean
  /**
   * Reserves the width of a group row's chevron before the name, so a list
   * holding groups keeps every name on one edge.
   */
  hasChevronGutter?: boolean
  payer: AccessPayer
  isEnabled: boolean
  /** Absent for a source this workspace cannot switch. */
  onToggle?: (next: boolean) => void
  /** What the scope control says, such as "Models: All". Empty for no control. */
  scopeLabel?: string
  onOpenScope?: () => void
  isCustomized: boolean
  onReset?: () => void
  note?: string
  /** Greys the name, for a row something upstream keeps from being used. */
  isMuted?: boolean
  /** Holds every control, while a write is in flight or for a reader. */
  isDisabled?: boolean
  /** The row's less common actions, ahead of Reset in its menu. */
  actions?: readonly AccessRowAction[]
}) {
  const menu =
    isCustomized && onReset
      ? [...actions, { id: "reset", label: "Reset", onAction: onReset }]
      : actions
  return (
    <SettingRow
      isNested={isNested}
      label={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {hasChevronGutter ? (
            <span aria-hidden="true" className="w-4 shrink-0" />
          ) : null}
          <span className={isMuted ? "text-subtle" : undefined}>
            {displayName}
          </span>
          <span className="text-caption text-subtle">
            {note ? `${PAYER_LABEL[payer]} · ${note}` : PAYER_LABEL[payer]}
          </span>
          {isCustomized ? (
            <Chip tone="accent" size="sm">
              Customized
            </Chip>
          ) : null}
        </span>
      }
      control={
        <div className="flex items-center gap-3">
          {onToggle ? (
            <Toggle
              label={`Use ${name} in this workspace`}
              isSelected={isEnabled}
              onChange={onToggle}
              isDisabled={isDisabled}
            />
          ) : null}
          <div className="min-w-0 flex-1">
            {scopeLabel && onOpenScope ? (
              <Button
                size="sm"
                variant="ghost"
                fullWidth
                aria-label={`${scopeLabel} on ${name}`}
                isDisabled={isDisabled}
                onPress={onOpenScope}
              >
                {scopeLabel}
              </Button>
            ) : null}
          </div>
          <div className="flex size-11 shrink-0 items-center justify-center">
            {menu.length > 0 ? (
              <RowMenu name={name} actions={menu} isDisabled={isDisabled} />
            ) : null}
          </div>
        </div>
      }
    />
  )
}

function RowMenu({
  name,
  actions,
  isDisabled,
}: {
  name: string
  actions: readonly AccessRowAction[]
  isDisabled: boolean
}) {
  return (
    <Dropdown>
      <Dropdown.Trigger
        aria-label={`More for ${name}`}
        isDisabled={isDisabled}
        className="inline-flex size-11 items-center justify-center text-muted hover:text-foreground"
      >
        <FiMoreHorizontal aria-hidden className="size-4 shrink-0" />
      </Dropdown.Trigger>
      <Dropdown.Popover placement="bottom end">
        <Dropdown.Menu
          aria-label={`Actions for ${name}`}
          onAction={(id) =>
            actions.find((action) => action.id === id)?.onAction()
          }
        >
          {actions.map((action) => (
            <Dropdown.Item
              key={action.id}
              id={action.id}
              textValue={action.label}
            >
              <Label className="text-body">{action.label}</Label>
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown>
  )
}

/** "Models: All" or "Models: 3 selected", which is what a row's scope control says. */
export function modelsScopeLabel(models: readonly string[] | null): string {
  if (models === null || models.length === 0) return "Models: All"
  return `Models: ${models.length} selected`
}
