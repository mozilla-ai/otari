import { useState } from "react"
import { FiChevronDown, FiStar } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { SearchField } from "@/design-system/forms/SearchField"
import { Popover } from "@/design-system/overlays/Popover"

import {
  groupPlaygroundModels,
  type PlaygroundModel,
} from "./helpers/playgroundModels"

/**
 * The model picker: searchable, grouped by provider instance, with pinned
 * models on top.
 *
 * A `Popover` holding a search field and a list of buttons rather than the
 * shared `ComboBoxField`, and this is the one deliberate departure from "reach
 * for the primitive first". The rows here are not options: each carries a second
 * control, the pin, which has to be pressable without selecting the row, and a
 * react-aria collection has no room for a control inside an option (pressing
 * anywhere in one selects it). The alternative was a pin control somewhere else
 * entirely, which is what makes pinning a model something nobody discovers.
 *
 * Search matches the whole key rather than the label, so typing a provider name
 * narrows to that instance and typing a model name finds it across instances.
 */
export function ModelSelect({
  value,
  onChange,
  models,
  pinnedKeys,
  onTogglePin,
  unavailableKeys,
  label,
  className = "",
}: {
  /** The selected `instance:model` key, or "" for none. */
  value: string
  onChange: (key: string) => void
  models: readonly PlaygroundModel[]
  pinnedKeys: readonly string[]
  onTogglePin: (key: string) => void
  /** Keys that cannot be picked here, such as the other panel's model. */
  unavailableKeys?: readonly string[]
  /** The accessible name, which is what tells the two compare pickers apart. */
  label: string
  className?: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState("")

  const unavailable = new Set(unavailableKeys ?? [])
  const pinned = new Set(pinnedKeys)
  const selectedLabel = models.find((model) => model.key === value)?.key ?? ""
  const groups = groupPlaygroundModels({ models, pinnedKeys, search })

  const select = (key: string) => {
    onChange(key)
    setIsOpen(false)
  }

  return (
    <Popover
      isOpen={isOpen}
      onOpenChange={(next) => {
        setIsOpen(next)
        // Cleared on open rather than on close, so the list is whole every time
        // it is opened without the previous query flashing away as it closes.
        if (next) setSearch("")
      }}
      placement="bottom"
      trigger={
        <Button
          aria-label={label}
          className={`justify-between font-normal ${className}`}
        >
          <span className="truncate">{selectedLabel || "Select a model"}</span>
          <FiChevronDown aria-hidden className="size-4 shrink-0 text-muted" />
        </Button>
      }
    >
      <div className="flex w-72 max-w-[calc(100vw-2rem)] flex-col gap-2 p-2">
        <SearchField
          label={`Search ${label}`}
          value={search}
          onChange={setSearch}
          placeholder="Search models"
        />
        <div className="flex max-h-72 flex-col gap-1 overflow-y-auto">
          {groups.length === 0 ? (
            <p className="px-2 py-3 text-center text-sm text-muted">
              No models match.
            </p>
          ) : (
            groups.map((group, index) => (
              <div
                key={group.id}
                className={`flex flex-col ${
                  index > 0 ? "border-border border-t pt-2" : ""
                }`}
              >
                <span className="sticky top-0 z-10 bg-surface px-2 py-1 text-overline">
                  {group.label}
                </span>
                {group.models.map((model) => {
                  const isSelected = model.key === value
                  const isPinned = pinned.has(model.key)
                  return (
                    <div
                      key={model.key}
                      className="group/row flex items-center gap-1 rounded-md pr-1 transition-colors hover:bg-surface-subtle"
                    >
                      <button
                        type="button"
                        disabled={unavailable.has(model.key)}
                        onClick={() => select(model.key)}
                        className="flex min-h-11 min-w-0 flex-1 items-center justify-between gap-2 px-2 text-left text-sm text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        <span className="truncate">{model.label}</span>
                        {isSelected ? (
                          <span className="shrink-0 text-caption text-link">
                            Selected
                          </span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        aria-label={
                          isPinned ? `Unpin ${model.key}` : `Pin ${model.key}`
                        }
                        aria-pressed={isPinned}
                        onClick={() => onTogglePin(model.key)}
                        className={`flex size-11 shrink-0 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface hover:text-foreground ${
                          isPinned
                            ? "text-link"
                            : "md:opacity-0 md:group-hover/row:opacity-100 md:group-focus-within/row:opacity-100"
                        }`}
                      >
                        <FiStar
                          aria-hidden
                          className={`size-4 ${isPinned ? "fill-current" : ""}`}
                        />
                      </button>
                    </div>
                  )
                })}
              </div>
            ))
          )}
        </div>
      </div>
    </Popover>
  )
}
