import { Dropdown, Label, Tooltip } from "@heroui/react"
import { type RefObject, useRef } from "react"
import { Text } from "react-aria-components"
import {
  FiEdit2,
  FiMoreHorizontal,
  FiPause,
  FiPlay,
  FiRefreshCw,
  FiTrash2,
} from "react-icons/fi"
import type { ApiKey } from "@/client"
import { TOOLTIP_OPEN_DELAY_MS } from "@/design-system/overlays/Tooltip"
import { formatDateTime } from "@/shared/helpers/format"
import { keyFingerprint } from "./secretCaption"

export function KeyActionsMenu({
  apiKey,
  triggerRef,
  onAction,
  owner,
  hasDetails,
  isPending,
  onToggle,
  onEdit,
  onRegenerate,
  onDelete,
}: {
  apiKey: ApiKey
  triggerRef?: RefObject<HTMLButtonElement | null>
  onAction: () => void
  owner?: string
  /**
   * Created, Last used, Expires and Models are lanes on the wide table; below it
   * they leave the row, and the menu is the only place they can still be read
   * whole.
   */
  hasDetails?: boolean
  isPending: boolean
  onToggle: () => void
  onEdit: () => void
  onRegenerate: () => void
  onDelete: () => void
}) {
  const trigger = useRef<HTMLButtonElement>(null)
  const name = apiKey.key_name ?? "(unnamed)"
  const actions = [
    {
      id: "toggle",
      label: apiKey.is_active ? "Disable" : "Enable",
      hint: apiKey.is_active ? "Callers get 401" : "Allow requests",
      icon: apiKey.is_active ? FiPause : FiPlay,
      isDisabled: isPending,
      run: onToggle,
    },
    {
      id: "edit",
      label: "Edit…",
      hint: undefined,
      icon: FiEdit2,
      isDisabled: false,
      run: onEdit,
    },
    {
      id: "regenerate",
      label: "Regenerate…",
      hint: "Replaces the secret",
      icon: FiRefreshCw,
      isDisabled: isPending,
      run: onRegenerate,
    },
    {
      id: "delete",
      label: "Delete…",
      hint: apiKey.is_active ? "Disable it first" : "Cannot be undone",
      icon: FiTrash2,
      isDisabled: apiKey.is_active || isPending,
      run: onDelete,
    },
  ]
  return (
    <Dropdown>
      {/* HeroUI reads `--tooltip-delay` off the root, which this app never
          declares, so an unset delay is react-aria's 1.5s warmup. The tooltip
          is the only visible label this trigger has. */}
      <Tooltip.Root delay={TOOLTIP_OPEN_DELAY_MS}>
        <Dropdown.Trigger
          ref={(element) => {
            trigger.current = element
            if (triggerRef) triggerRef.current = element
          }}
          aria-label={`Actions for ${name}`}
          className="otari-key-actions-trigger"
        >
          <FiMoreHorizontal aria-hidden className="size-4 shrink-0" />
        </Dropdown.Trigger>
        <Tooltip.Content>Actions for {name}</Tooltip.Content>
      </Tooltip.Root>
      <Dropdown.Popover
        placement="bottom end"
        className="otari-key-actions-menu"
      >
        <div className="flex flex-col gap-1 border-b border-border px-3 py-3">
          <p className="break-words text-body">{name}</p>
          {owner ? <p className="break-all text-caption">{owner}</p> : null}
          <p className="break-all text-mono-caption text-muted">
            {keyFingerprint(apiKey) ?? "No key prefix"}
          </p>
          {hasDetails ? (
            <dl className="flex flex-col gap-1 pt-2 text-caption">
              <div>
                <dt className="inline">Created: </dt>
                <dd className="inline">{formatDateTime(apiKey.created_at)}</dd>
              </div>
              <div>
                <dt className="inline">Last used: </dt>
                <dd className="inline">
                  {apiKey.last_used_at
                    ? formatDateTime(apiKey.last_used_at)
                    : "never"}
                </dd>
              </div>
              <div>
                <dt className="inline">Expires: </dt>
                <dd className="inline">
                  {apiKey.expires_at
                    ? formatDateTime(apiKey.expires_at)
                    : "never"}
                </dd>
              </div>
              <div>
                <dt className="inline">Models: </dt>
                <dd className="inline break-words">
                  {apiKey.allowed_models === null
                    ? "All models"
                    : apiKey.allowed_models.join(", ") || "No models"}
                </dd>
              </div>
              {apiKey.exclude_from_budget ? (
                <div>
                  <dt className="inline">Budget: </dt>
                  <dd className="inline">Exempt</dd>
                </div>
              ) : null}
              {apiKey.reject_user_mismatch !== null ? (
                <div>
                  <dt className="inline">User matching: </dt>
                  <dd className="inline">
                    {apiKey.reject_user_mismatch ? "Strict" : "Lenient"}
                  </dd>
                </div>
              ) : null}
            </dl>
          ) : null}
        </div>
        <Dropdown.Menu
          aria-label={`Actions for ${name}`}
          items={actions}
          onAction={(id) => {
            // Dialogs restore focus to the persistent trigger, not the closing item.
            trigger.current?.focus()
            onAction()
            actions.find((action) => action.id === id)?.run()
          }}
        >
          {(action) => (
            <Dropdown.Item
              id={action.id}
              textValue={action.label}
              isDisabled={action.isDisabled}
            >
              <action.icon aria-hidden className="size-3.5 shrink-0" />
              <Label className="text-body">{action.label}</Label>
              {action.hint ? (
                <Text slot="description" className="text-caption">
                  {action.hint}
                </Text>
              ) : null}
            </Dropdown.Item>
          )}
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown>
  )
}
