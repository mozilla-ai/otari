import { useState } from "react"

import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useOrganizationMemory,
  useSetOrganizationMemory,
} from "@/shared/api/memory"

const LABEL = "Agent memory"

/**
 * The organization's memory switch, above every workspace's own.
 *
 * Only the organization's owners and admins see it, since only they may read
 * or change it. On lets each workspace's managers turn memory on for their
 * workspace; off stops it in all of them at once and deletes nothing, which the
 * confirmation says before it happens.
 */
export function OrganizationMemorySection() {
  const memory = useOrganizationMemory(true)
  const save = useSetOrganizationMemory()
  const [next, setNext] = useState<boolean>()

  const close = () => {
    setNext(undefined)
    save.reset()
  }

  return (
    <SettingsGroup
      title="Agent memory"
      description="A memory each workspace's agents share across models and sessions. Each workspace still turns it on for itself."
    >
      <SettingRow
        label={LABEL}
        help="Off stops memory in every workspace of this organization, whatever their own switches say, and deletes nothing."
        control={
          <Toggle
            isSelected={memory.data?.enabled ?? false}
            onChange={setNext}
            label={LABEL}
            isDisabled={!memory.data || save.isPending}
          />
        }
      />
      <ErrorBanner error={next === undefined ? memory.error : null} />
      <ConfirmDialog
        isOpen={next !== undefined}
        onOpenChange={(open) => {
          if (!open) close()
        }}
        heading={
          next
            ? "Turn on memory for this organization?"
            : "Turn off memory for every workspace?"
        }
        body={
          <p>
            {next
              ? "Each workspace's owners and admins can then turn memory on for their workspace. Nothing runs in a workspace until they do."
              : "Agents stop recalling and proposing in every workspace of this organization at once. What each workspace holds stays, and is still deleted by its retention."}
          </p>
        }
        confirmLabel={next ? "Turn on memory" : "Turn off memory"}
        confirmVariant="primary"
        isPending={save.isPending}
        error={save.error}
        onConfirm={() =>
          save.mutate({ enabled: next ?? false }, { onSuccess: close })
        }
      />
    </SettingsGroup>
  )
}
