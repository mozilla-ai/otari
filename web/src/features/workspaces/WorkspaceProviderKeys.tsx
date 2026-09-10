import { Button } from "@heroui/react"
import { useState } from "react"

import type { OrgProviderKey, WorkspaceProviderKeyOverride } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Field } from "@/design-system/forms/Field"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { useOrgProviderKeys } from "@/shared/api/organizations"
import {
  useAddWorkspaceProviderKeyModel,
  useRemoveWorkspaceProviderKeyModel,
  useResetWorkspaceProviderKeyOverride,
  useSetWorkspaceProviderKeyOverride,
  useWorkspaceProviderKeyModels,
  useWorkspaceProviderKeys,
} from "@/shared/api/workspaces"

/**
 * One workspace's departures from the provider keys its organization holds.
 *
 * The other half of `OrganizationProviderKeysPage`: that page owns the
 * credentials themselves, this owns what one workspace does with them. Absence
 * of an override is full inheritance, so every row starts at "Inherited" and a
 * departure is something an admin chose.
 *
 * The three states are the two stored flags plus their absence, because the
 * gateway stores no no-op row (see `WorkspaceProviderKeyOverrideRequest`):
 * pinning is `is_default`, opting out is `disabled`, and inheriting is the row
 * being deleted. One flag is sent at a time and the other is left to
 * auto-resolve, which is what the tri-state request is for.
 *
 * The allow-list is per (workspace, key) and narrows rather than widens: no rows
 * means every model that key serves. It is what
 * `services/tenancy/organization_model_access.py` reads when it builds a
 * session's catalog scope, so emptying a workspace's catalog here also takes
 * `/v1/models` down to nothing for it, and with it the first-request setup guide
 * that asks the same endpoint whether a call could succeed (#980).
 */

/** What a row's picker is showing, which is the pair of stored flags named. */
type Departure = "inherited" | "pinned" | "disabled"

const DEPARTURE_OPTIONS: { value: Departure; label: string }[] = [
  { value: "inherited", label: "Inherited" },
  { value: "pinned", label: "Workspace default" },
  { value: "disabled", label: "Disabled" },
]

function departureOf(row: WorkspaceProviderKeyOverride): Departure {
  if (row.disabled) return "disabled"
  if (row.is_default) return "pinned"
  return "inherited"
}

/**
 * Narrow the picker's string back to the vocabulary above.
 *
 * A guard rather than a cast, following `asMembershipRole`: the options come
 * from `DEPARTURE_OPTIONS`, and a value that is not one of them is a bug worth
 * dropping the write for rather than sending and having refused.
 */
function asDeparture(value: string): Departure | undefined {
  return DEPARTURE_OPTIONS.find((option) => option.value === value)?.value
}

/** How a key is named on screen: its provider and the name the organization gave it. */
function keyLabel(key: OrgProviderKey | undefined, keyId: string): string {
  // A key the organization list does not carry is not a state the gateway
  // produces (both reads cover the same non-archived set), so this is a fallback
  // rather than a case: naming the id keeps the row addressable if it happens.
  if (!key) return keyId
  return `${key.provider} / ${key.name}`
}

function ModelAllowList({
  workspaceId,
  keyId,
  keyName,
  models,
  failed,
}: {
  workspaceId: string
  keyId: string
  keyName: string
  /** Undefined until this key's own read answers; an empty list is a real answer. */
  models: string[] | undefined
  /** Whether that read was refused, which is a wait that never ends otherwise. */
  failed: boolean
}) {
  const add = useAddWorkspaceProviderKeyModel()
  const remove = useRemoveWorkspaceProviderKeyModel()
  const [draft, setDraft] = useState("")
  const pending = add.isPending || remove.isPending
  const trimmed = draft.trim()

  return (
    <div className="flex flex-col gap-2">
      <ErrorBanner error={add.error ?? remove.error} />
      {failed ? (
        // Said on the row as well as in the banner above: without this the row
        // waits forever on a read that already answered, and a caption saying
        // "loading" is a claim that the answer is still coming.
        <span className="text-caption">
          The allowed models for this key could not be read.
        </span>
      ) : models === undefined ? (
        // Not "every model is allowed", which is what an empty list means and
        // what an unanswered read would otherwise be read as: a narrowed
        // workspace would say it was open for as long as the read took.
        <span className="text-caption">Loading allowed models…</span>
      ) : models.length === 0 ? (
        <span className="text-caption">
          Every model this key serves is allowed.
        </span>
      ) : (
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {models.map((model) => (
            <li key={model}>
              <DismissChip
                value={model}
                // Only what the press does: dropping one entry leaves the rest
                // of the allow-list in force, so a label promising every model
                // back is true of the last entry alone.
                dismissLabel={`Stop allowing ${model} on ${keyName}`}
                onDismiss={() => remove.mutate({ workspaceId, keyId, model })}
              />
            </li>
          ))}
        </ul>
      )}
      {/* A text input rather than a picker over the catalog: the only listing a
          tenant may read is `/v1/models`, which is already filtered through
          these very restrictions, so a picker built on it would stop offering
          the models this control exists to add back. */}
      <div className="flex items-end gap-2">
        <Field
          label={`Allow a model on ${keyName}`}
          value={draft}
          onChange={setDraft}
          placeholder="gpt-4o"
        />
        <Button
          size="sm"
          variant="ghost"
          isDisabled={pending || trimmed === ""}
          onPress={() =>
            add.mutate(
              { workspaceId, keyId, model: trimmed },
              { onSuccess: () => setDraft("") },
            )
          }
        >
          Allow
        </Button>
      </div>
    </div>
  )
}

export function WorkspaceProviderKeys({
  workspaceId,
}: {
  workspaceId: string
}) {
  const orgKeys = useOrgProviderKeys()
  const overrides = useWorkspaceProviderKeys(workspaceId)
  const rows = overrides.data ?? []
  const models = useWorkspaceProviderKeyModels(
    workspaceId,
    rows.map((row) => row.org_provider_key_id),
  )
  const setOverride = useSetWorkspaceProviderKeyOverride()
  const resetOverride = useResetWorkspaceProviderKeyOverride()

  const byId = new Map((orgKeys.data ?? []).map((key) => [key.id, key]))
  const pending = setOverride.isPending || resetOverride.isPending

  const choose = (keyId: string, next: Departure) => {
    if (next === "inherited") {
      resetOverride.mutate({ workspaceId, keyId })
      return
    }
    // One flag, never both: the gateway un-pins a key being disabled and
    // re-enables one being pinned, and sending both true is the one combination
    // it refuses.
    setOverride.mutate({
      workspaceId,
      keyId,
      body: next === "pinned" ? { is_default: true } : { disabled: true },
    })
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <span className="text-body">Provider keys</span>
        <span className="max-w-md text-caption">
          This workspace inherits every provider key the organization holds. Pin
          one as its default, opt out of one, or narrow a key to named models.
          Narrowing hides every other model of that provider from this
          workspace, including from its model list.
        </span>
      </div>
      <ErrorBanner
        error={
          orgKeys.error ??
          overrides.error ??
          models.error ??
          setOverride.error ??
          resetOverride.error
        }
      />
      {overrides.isPending && overrides.data === undefined ? (
        <span className="text-caption">Loading…</span>
      ) : rows.length === 0 ? (
        <span className="text-caption">
          This organization holds no provider keys, so there is nothing for this
          workspace to depart from.
        </span>
      ) : (
        <ul className="flex flex-col gap-4">
          {rows.map((row) => {
            const keyId = row.org_provider_key_id
            const name = keyLabel(byId.get(keyId), keyId)
            const departure = departureOf(row)
            return (
              <li key={keyId} className="flex flex-col gap-2">
                <div className="flex flex-wrap items-end gap-2">
                  <span className="text-mono-caption text-foreground">
                    {name}
                  </span>
                  <FilterSelect
                    ariaLabel={`This workspace's use of ${name}`}
                    value={departure}
                    onChange={(next) => {
                      const chosen = asDeparture(next)
                      if (chosen) choose(keyId, chosen)
                    }}
                    options={DEPARTURE_OPTIONS}
                    disabled={pending}
                  />
                  {/* Which key the provider actually resolves to, which the
                      flags alone do not say: an unpinned key is still the one
                      serving this workspace when it is the organization's
                      default, or the only one that provider has. */}
                  {row.is_effective_default ? (
                    <span className="text-caption">In use</span>
                  ) : null}
                </div>
                {departure === "disabled" ? (
                  // Not a control: the gateway refuses an allow-list write on a
                  // disabled key, and deletes the rows it already had when the
                  // key is disabled.
                  <span className="text-caption">
                    No model of this key is available to this workspace.
                  </span>
                ) : (
                  <ModelAllowList
                    workspaceId={workspaceId}
                    keyId={keyId}
                    keyName={name}
                    models={models.data.get(keyId)?.models}
                    failed={models.data.get(keyId)?.failed ?? false}
                  />
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
