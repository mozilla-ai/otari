import { useState } from "react"

import type { UpdateWorkspaceMemoryRequest, WorkspaceMemory } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Field } from "@/design-system/forms/Field"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useEraseWorkspaceMemory,
  useSetWorkspaceMemory,
  useWorkspaceMemory,
} from "@/shared/api/memory"

const MEMORY_LABEL = "Agent memory"
const SHARED_LABEL = "Shared workspace memory"
const EVERY_REQUEST_LABEL = "Offer memory to every request"
const REVIEW_LABEL = "Review shared proposals"
const HIDE_EXTERNAL_LABEL = "Leave outside content out of recall"
const RETENTION_LABEL = "Keep unused entries for (days)"

type Pending = "on" | "erase"

function switches(memory: WorkspaceMemory): UpdateWorkspaceMemoryRequest {
  return {
    enabled: memory.enabled,
    workspace_scope_enabled: memory.workspace_scope_enabled,
    write_policy: memory.write_policy,
    offer_to_every_request: memory.offer_to_every_request,
    hide_external: memory.hide_external,
    retention_days: memory.retention_days,
  }
}

function holdings(memory: WorkspaceMemory): string {
  const active = memory.entries.active ?? 0
  const waiting = memory.entries.quarantined ?? 0
  const parts = [
    `${active} ${active === 1 ? "entry" : "entries"} in recall`,
    `${memory.proposals_waiting} ${memory.proposals_waiting === 1 ? "proposal" : "proposals"} not decided yet`,
  ]
  if (waiting > 0) parts.push(`${waiting} flagged out of recall`)
  return `${parts.join(", ")}.`
}

/**
 * A workspace's agent memory: whether its agents share one, how proposals
 * become memory, and erasing what it holds.
 *
 * Shown as a tab of the workspace's page. Every switch applies as it is
 * flipped, except turning memory on, which says first what is stored and who
 * reads it, and erasing, which cannot be undone. Memory runs only while the
 * organization has it on too, and the page says so rather than hiding the
 * switches, so a workspace can be made ready before the organization turns on.
 */
export function WorkspaceMemorySection({
  workspaceId,
  canManage,
}: {
  workspaceId: string
  canManage: boolean
}) {
  const memory = useWorkspaceMemory(workspaceId)
  const save = useSetWorkspaceMemory()
  const erase = useEraseWorkspaceMemory()
  const [pending, setPending] = useState<Pending>()
  const [retentionDraft, setRetentionDraft] = useState<string>()

  if (memory.isPending && !memory.data) {
    return <PageLoading label="Loading memory…" />
  }
  if (!memory.data) {
    return <ErrorBanner error={memory.error} />
  }
  const current = memory.data
  const locked = !canManage || save.isPending
  const change = (patch: Partial<UpdateWorkspaceMemoryRequest>) =>
    save.mutate({ workspaceId, body: { ...switches(current), ...patch } })

  const retention =
    retentionDraft ??
    (current.retention_days === null ? "" : String(current.retention_days))
  const retentionValue = retention.trim() === "" ? null : Number(retention)
  const retentionInvalid =
    retentionValue !== null &&
    (!Number.isInteger(retentionValue) ||
      retentionValue < 1 ||
      retentionValue > 3650)

  const close = () => {
    setPending(undefined)
    save.reset()
    erase.reset()
  }
  const confirm = async () => {
    try {
      if (pending === "on") {
        await save.mutateAsync({
          workspaceId,
          body: { ...switches(current), enabled: true },
        })
      } else if (pending === "erase") {
        await erase.mutateAsync({ workspaceId })
      }
      close()
    } catch {
      // Shown by the dialog.
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {current.organization_enabled ? null : (
        <InfoBanner>
          Memory is off for this organization, so nothing here runs yet. An
          organization owner or admin turns it on under Organization, General.
        </InfoBanner>
      )}
      {canManage ? null : (
        <InfoBanner>
          Only the organization's owners and admins, and this workspace's, can
          change its memory.
        </InfoBanner>
      )}
      <SettingsGroup
        title="Agent memory"
        description="One memory this workspace's agents share, whatever model they run on. Agents propose what they learn and a consolidator decides what is kept. Everything is stored encrypted, and nothing is read back without this deployment's key."
      >
        <SettingRow
          label={MEMORY_LABEL}
          help="Each member's agents keep and recall entries of their own. No other member, and no admin, recalls them."
          control={
            <Toggle
              isSelected={current.enabled}
              onChange={(next) =>
                next ? setPending("on") : change({ enabled: false })
              }
              label={MEMORY_LABEL}
              isDisabled={locked}
            />
          }
        />
        <SettingRow
          label={SHARED_LABEL}
          help="The memory every member's agents share: facts, decisions and procedures of this workspace."
          isNested
          control={
            <Toggle
              isSelected={current.workspace_scope_enabled}
              onChange={(next) => change({ workspace_scope_enabled: next })}
              label={SHARED_LABEL}
              isDisabled={locked || !current.enabled}
            />
          }
        />
        <SettingRow
          label={REVIEW_LABEL}
          help="Hold every proposal for the shared memory until an admin approves it. Off, the consolidator decides alone, and still holds a conflict for review."
          isNested
          control={
            <Toggle
              isSelected={current.write_policy === "review"}
              onChange={(next) =>
                change({ write_policy: next ? "review" : "auto" })
              }
              label={REVIEW_LABEL}
              isDisabled={locked || !current.enabled}
            />
          }
        />
        <SettingRow
          label={EVERY_REQUEST_LABEL}
          help="Give memory's tools to every request a member sends through Otari, not only to one that asks for them with otari_memory or Otari-Memory: on. A request can still decline with Otari-Memory: off."
          isNested
          control={
            <Toggle
              isSelected={current.offer_to_every_request}
              onChange={(next) => change({ offer_to_every_request: next })}
              label={EVERY_REQUEST_LABEL}
              isDisabled={locked || !current.enabled}
            />
          }
        />
        <SettingRow
          label={HIDE_EXTERNAL_LABEL}
          help="Keep entries resting on a web page, a fetched file or a third-party tool out of search."
          isNested
          control={
            <Toggle
              isSelected={current.hide_external}
              onChange={(next) => change({ hide_external: next })}
              label={HIDE_EXTERNAL_LABEL}
              isDisabled={locked || !current.enabled}
            />
          }
        />
        <div className="flex flex-col gap-2 md:flex-row md:items-end">
          <Field
            label={RETENTION_LABEL}
            value={retention}
            onChange={setRetentionDraft}
            inputMode="numeric"
            placeholder={String(current.deployment_retention_days)}
            description={`An entry nobody uses for this long is deleted. Empty keeps this deployment's ${current.deployment_retention_days} days.`}
            isDisabled={!canManage}
            isInvalid={retentionInvalid}
            errorMessage="Between 1 and 3650 days."
          />
          <Button
            variant="primary"
            className="min-h-11 md:min-h-9 shrink-0"
            isDisabled={
              locked || retentionInvalid || retentionDraft === undefined
            }
            onPress={() =>
              save.mutate(
                {
                  workspaceId,
                  body: {
                    ...switches(current),
                    retention_days: retentionValue,
                  },
                },
                { onSuccess: () => setRetentionDraft(undefined) },
              )
            }
          >
            Save
          </Button>
        </div>
      </SettingsGroup>
      <p className="text-caption">{holdings(current)}</p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          className="min-h-11 md:min-h-9"
          onPress={() => setPending("erase")}
          isDisabled={!canManage || erase.isPending}
        >
          Erase this workspace's memory
        </Button>
      </div>
      <ErrorBanner error={pending === undefined ? save.error : null} />
      <ConfirmDialog
        isOpen={pending !== undefined}
        onOpenChange={(open) => {
          if (!open) close()
        }}
        heading={
          pending === "on"
            ? "Turn on memory for this workspace?"
            : "Erase this workspace's memory?"
        }
        body={
          pending === "on" ? (
            <p>
              Agents of this workspace's members can then keep and recall what
              they learn. Each entry is encrypted, carries who proposed it and
              from which session, and is deleted once nobody has used it for the
              retention period. Proposals with credentials, keys, email
              addresses or card numbers are refused.
            </p>
          ) : (
            <p>
              Deletes every entry, proposal and read record this workspace's
              memory holds, and the keys they were encrypted with, so nothing
              deleted can be read from a backup either. The switches stay as
              they are. This cannot be undone.
            </p>
          )
        }
        confirmLabel={pending === "on" ? "Turn on memory" : "Erase memory"}
        confirmVariant={pending === "erase" ? "danger" : "primary"}
        isPending={save.isPending || erase.isPending}
        error={save.error ?? erase.error}
        onConfirm={() => void confirm()}
      />
    </div>
  )
}
