import { useEffect, useState } from "react"
import { FiArrowDown, FiArrowUp, FiPlus, FiTrash2 } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { FieldAction } from "@/design-system/forms/FieldAction"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { OfferingPicker } from "@/features/models/OfferingPicker"

// When `walk_attempts` (`api/routes/_attempts.py`) moves on to the next model.
const BACKUP_HELP =
  "Tried in order when the main model fails or is unavailable before it starts answering, including a rate limit, a missing or rejected provider key, or a request it refuses as invalid. Not once the answer has started streaming or a tool call is under way, or when the remaining budget cannot cover a pricier backup."

let nextRowKey = 0
const newRowKey = () => {
  nextRowKey += 1
  return nextRowKey
}

/**
 * The backups a policy falls over to, in order: its `on_failure` chain.
 *
 * Rows are keyed by identity rather than position, so a picker's own state
 * (a search in progress, the model it last resolved) moves with its model.
 */
export function BackupList({
  backups,
  onChange,
  isRequired = false,
  problems = [],
  isAtCap,
  maxModels,
}: {
  backups: string[]
  onChange: (next: string[]) => void
  /** At least one backup: the last row cannot be removed. */
  isRequired?: boolean
  /** Per row, why its backup cannot be saved, or "" when it can. */
  problems?: string[]
  /** Whether the policy already dispatches as many models as it may. */
  isAtCap: boolean
  maxModels: number
}) {
  const [keys, setKeys] = useState<number[]>(() => backups.map(newRowKey))
  // Every change goes through here, so this only repairs a list replaced from
  // outside, which then starts over with fresh rows.
  if (keys.length !== backups.length) setKeys(backups.map(newRowKey))
  // The backup just moved, by row key, for the focus effect below.
  const [moved, setMoved] = useState<{
    key: number
    direction: "up" | "down"
  }>()

  const move = (from: number, to: number) => {
    const swap = <T,>(list: T[]) => {
      const next = [...list]
      const [moved] = next.splice(from, 1)
      if (moved !== undefined) next.splice(to, 0, moved)
      return next
    }
    const key = keys[from]
    setKeys(swap)
    onChange(swap(backups))
    if (key !== undefined)
      setMoved({ key, direction: to < from ? "up" : "down" })
  }
  // The moved row's node moves with it and loses focus on the way, so focus
  // follows it, onto its other arrow at either end. Found by row key rather
  // than an `id`: the button keeps the first `id` it was given.
  useEffect(() => {
    if (moved === undefined) return
    const { key, direction } = moved
    const other = direction === "up" ? "down" : "up"
    const target = [direction, other]
      .map((dir) =>
        document.querySelector<HTMLElement>(
          `[data-backup-move="${key}:${dir}"]`,
        ),
      )
      .find(
        (button) =>
          button !== null &&
          !button.matches(":disabled, [aria-disabled='true']"),
      )
    target?.focus()
    setMoved(undefined)
  }, [moved])

  const remove = (index: number) => {
    setKeys((prev) => prev.filter((_, i) => i !== index))
    onChange(backups.filter((_, i) => i !== index))
  }
  const add = () => {
    setKeys((prev) => [...prev, newRowKey()])
    onChange([...backups, ""])
  }

  return (
    <div className="flex flex-col gap-3">
      <ControlField label="If unavailable, use" description={BACKUP_HELP} />
      {backups.length === 0 ? (
        <p className="text-caption">
          No backups yet. If the main model fails, the request fails.
        </p>
      ) : (
        <ol className="flex flex-col gap-3">
          {backups.map((entry, index) => (
            <li key={keys[index] ?? index}>
              <OfferingPicker
                id={`backup-${keys[index] ?? index}`}
                label={`Backup ${index + 1}`}
                providerLabel={`Provider ${index + 1}`}
                value={entry}
                onChange={(selector) =>
                  onChange(
                    backups.map((current, i) =>
                      i === index ? selector : current,
                    ),
                  )
                }
                isRequired
                errorMessage={problems[index] ?? ""}
                actions={
                  <FieldAction>
                    <div className="flex gap-1">
                      <IconButton
                        data-backup-move={`${keys[index]}:up`}
                        variant="ghost"
                        isIconOnly
                        label={`Move backup ${index + 1} up`}
                        isDisabled={index === 0}
                        onPress={() => move(index, index - 1)}
                      >
                        <FiArrowUp aria-hidden />
                      </IconButton>
                      <IconButton
                        data-backup-move={`${keys[index]}:down`}
                        variant="ghost"
                        isIconOnly
                        label={`Move backup ${index + 1} down`}
                        isDisabled={index === backups.length - 1}
                        onPress={() => move(index, index + 1)}
                      >
                        <FiArrowDown aria-hidden />
                      </IconButton>
                      <IconButton
                        variant="ghost"
                        isIconOnly
                        label={`Remove backup ${index + 1}`}
                        isDisabled={isRequired && backups.length === 1}
                        onPress={() => remove(index)}
                      >
                        <FiTrash2 aria-hidden />
                      </IconButton>
                    </div>
                  </FieldAction>
                }
              />
            </li>
          ))}
        </ol>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="ghost" isDisabled={isAtCap} onPress={add}>
          <FiPlus aria-hidden />
          Add a backup
        </Button>
        {isAtCap ? (
          <span className="text-caption">
            A policy dispatches at most {maxModels} models in total.
          </span>
        ) : null}
      </div>
    </div>
  )
}
