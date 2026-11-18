import { useState } from "react"

import type {
  OrgProviderKey,
  WorkspaceHostedProvider,
  WorkspaceProviderKeyOverride,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { DisclosureRow } from "@/design-system/navigation/DisclosureRow"
import { Segmented } from "@/design-system/navigation/Segmented"
import {
  dataLocationLabel,
  refusesProvider,
} from "@/features/hostedProviders/dataLocations"
import {
  AccessRow,
  modelsScopeLabel,
} from "@/features/workspaces/access/AccessRow"
import {
  AllowedDataLocationsDialog,
  describeAllowedDataLocations,
} from "@/features/workspaces/access/AllowedDataLocationsDialog"
import { HostedModelsDialog } from "@/features/workspaces/access/HostedModelsDialog"
import { KeyModelsDialog } from "@/features/workspaces/access/KeyModelsDialog"
import {
  keyLabel,
  keyModelProblem,
  keyPrefix,
} from "@/features/workspaces/access/keyModels"
import {
  ACCESS_FILTERS,
  type AccessEntry,
  type AccessFilter,
  filterGroups,
  groupByProvider,
  SEARCH_THRESHOLD,
  summarizeGroups,
} from "@/features/workspaces/access/providerGroups"
import { useOrgProviderKeys } from "@/shared/api/organizations"
import {
  useRemoveWorkspaceProviderKeyModel,
  useResetWorkspaceHostedProvider,
  useResetWorkspaceProviderKeyOverride,
  useSetWorkspaceHostedProvider,
  useSetWorkspaceProviderKeyOverride,
  useSetWorkspaceProviderPolicy,
  useWorkspaceHostedProviders,
  useWorkspaceProviderKeys,
  useWorkspaceProviderPolicy,
} from "@/shared/api/workspaces"

/** One source in the list, with the row it renders from. */
type Entry = AccessEntry &
  (
    | { kind: "key"; row: WorkspaceProviderKeyOverride }
    | { kind: "hosted"; row: WorkspaceHostedProvider }
  )

/** What a provider's folded row says about the sources inside it. */
function groupSummary(entries: readonly Entry[]): string {
  const keys = entries.filter((entry) => entry.kind === "key").length
  const customized = entries.filter((entry) => entry.isCustomized).length
  const off = entries.filter((entry) => entry.isOff).length
  return [
    keys > 0 ? `${keys} ${keys === 1 ? "key" : "keys"}` : "",
    entries.some((entry) => entry.kind === "hosted") ? "hosted" : "",
    customized > 0 ? `${customized} customized` : "",
    off > 0 ? `${off} off` : "",
  ]
    .filter(Boolean)
    .join(" · ")
}

/** Which row's scope is open, if any: a key by its id, or a hosted provider by its name. */
// Tagged with the workspace it was opened in, so a dialog never outlives a
// switch to another workspace, and a save still landing for the old one cannot
// close a dialog opened in the new one.
type OpenScope =
  | { kind: "key"; id: string; workspaceId: string }
  | { kind: "hosted"; provider: string; workspaceId: string }
  | undefined

/**
 * The providers and models a workspace may call: its organization's own keys,
 * then the hosted providers the deployment serves the organization.
 *
 * Every row starts inherited, and a departure is something an admin chose. For
 * a key, the switch is the stored `disabled` flag and the scope is the
 * workspace's model allow-list on it; which of a provider's keys is always used
 * is about the credential rather than about access, so it sits in the row's
 * menu and only where the provider has more than one key. For a hosted
 * provider, the switch and the model list are one stored rule.
 *
 * Every change here is saved as it is made, apart from the hosted model list,
 * which is built over several presses and waits for its drawer's Save.
 */
export function InferenceAccessSection({
  workspaceId,
  canManage,
  canManageOrganization,
}: {
  workspaceId: string
  canManage: boolean
  canManageOrganization: boolean
}) {
  // The organization's key list is how a key is named, and it answers only an
  // organization owner or admin, so nobody else asks: a workspace admin sees
  // the rows by fallback name rather than a refusal.
  const orgKeys = useOrgProviderKeys(canManageOrganization)
  const overrides = useWorkspaceProviderKeys(workspaceId)
  const hosted = useWorkspaceHostedProviders(workspaceId)
  const setOverride = useSetWorkspaceProviderKeyOverride()
  const resetOverride = useResetWorkspaceProviderKeyOverride()
  const removeModel = useRemoveWorkspaceProviderKeyModel()
  const setHosted = useSetWorkspaceHostedProvider()
  const resetHosted = useResetWorkspaceHostedProvider()
  const policy = useWorkspaceProviderPolicy(workspaceId)
  const setPolicy = useSetWorkspaceProviderPolicy()
  const requiresZeroRetention =
    policy.data?.require_zero_data_retention ?? false
  // Its own mutation, so a failed retention write is not shown in the
  // locations dialog, nor a failed locations save beside the switch.
  const setLocations = useSetWorkspaceProviderPolicy()
  const isPolicyWriting = setPolicy.isPending || setLocations.isPending
  const allowedLocations = policy.data?.allowed_data_locations ?? null
  const [locationsOpening, setLocationsOpening] = useState(0)
  // The workspace the locations dialog was opened in, so it never reopens
  // over another workspace after a switch.
  const [locationsOpenFor, setLocationsOpenFor] = useState("")
  const isEditingLocations = locationsOpenFor === workspaceId
  const [open, setOpen] = useState<OpenScope>()
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set())
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<AccessFilter>("all")

  const keys = (orgKeys.data ?? []).filter((key) => key.archived_at === null)
  const byId = new Map(keys.map((key) => [key.id, key]))
  const keysPerProvider = keys.reduce(
    (counts, key) =>
      counts.set(keyPrefix(key), (counts.get(keyPrefix(key)) ?? 0) + 1),
    new Map<string, number>(),
  )
  const keyRows = overrides.data ?? []
  const hostedRows = hosted.data ?? []
  // What a typed model name's prefix is checked against.
  const knownProviders: ReadonlySet<string> = new Set([
    ...keys.flatMap((key) => [key.provider, keyPrefix(key)]),
    ...hostedRows.map((row) => row.provider),
  ])
  const isPending =
    setOverride.isPending ||
    resetOverride.isPending ||
    removeModel.isPending ||
    setHosted.isPending ||
    resetHosted.isPending
  const isDisabled = !canManage || isPending

  // A key's departure is the override row and its allow-list, which are stored
  // apart, so going back to inheritance clears both.
  const resetKey = async (row: WorkspaceProviderKeyOverride) => {
    const keyId = row.org_provider_key_id
    if (row.is_default || row.disabled) {
      await resetOverride.mutateAsync({ workspaceId, keyId })
    }
    for (const model of row.allowed_models) {
      await removeModel.mutateAsync({ workspaceId, keyId, model })
    }
  }

  // No dialog without the policy, in another workspace than it was opened in,
  // or on a source the policy now blocks: each would save where the page says
  // nothing may be changed.
  const isOpenHere = open?.workspaceId === workspaceId && policy.data
  const isUsable = (row: { zero_data_retention: boolean }) =>
    !requiresZeroRetention || row.zero_data_retention
  const openKey =
    open?.kind === "key" && isOpenHere
      ? keyRows.find(
          (row) => row.org_provider_key_id === open.id && isUsable(row),
        )
      : undefined
  const openKeyRecord = openKey
    ? byId.get(openKey.org_provider_key_id)
    : undefined
  // A hosted provider none of whose locations the workspace allows is refused
  // as one not declared zero data retention is.
  const isLocationRefused = (row: WorkspaceHostedProvider) =>
    refusesProvider(allowedLocations, row.data_locations ?? [])
  const openHosted =
    open?.kind === "hosted" && isOpenHere
      ? hostedRows.find(
          (row) =>
            row.provider === open.provider &&
            isUsable(row) &&
            !isLocationRefused(row),
        )
      : undefined

  // Held for the policy too: until it answers, no row knows it is blocked, and
  // a key's model dialog would offer what the workspace refuses.
  const isLoading =
    (overrides.isPending && !overrides.data) ||
    (hosted.isPending && !hosted.data) ||
    (policy.isPending && !policy.data)

  // One row per provider: a provider with several sources (two keys, or a key
  // and a hosted provider) folds them under a row that opens in place.
  const entries: Entry[] = [
    ...keyRows.map((row): Entry => {
      const orgKey = byId.get(row.org_provider_key_id)
      return {
        kind: "key",
        row,
        id: row.org_provider_key_id,
        // A key whose name this caller cannot read stands alone.
        provider: orgKey ? keyPrefix(orgKey) : row.org_provider_key_id,
        name: keyLabel(orgKey, row.org_provider_key_id),
        isCustomized: keyIsCustomized(row),
        isOff:
          row.disabled || (requiresZeroRetention && !row.zero_data_retention),
      }
    }),
    ...hostedRows.map(
      (row): Entry => ({
        kind: "hosted",
        row,
        id: `hosted:${row.provider}`,
        provider: row.provider,
        name: row.provider,
        isCustomized: row.customized,
        isOff:
          !row.enabled ||
          (requiresZeroRetention && !row.zero_data_retention) ||
          isLocationRefused(row),
      }),
    ),
  ]
  const groups = groupByProvider(entries)
  const isSearchable = groups.length > SEARCH_THRESHOLD
  const isNarrowed = isSearchable && (query.trim() !== "" || filter !== "all")
  const shown = isNarrowed ? filterGroups(groups, query, filter) : groups
  // Where any group folds rows, every name keeps the room its chevron takes.
  const hasGroups = shown.some((group) => group.entries.length > 1)

  const renderEntry = (entry: Entry, isNested: boolean) =>
    entry.kind === "key" ? (
      <KeyRow
        key={entry.id}
        row={entry.row}
        orgKey={byId.get(entry.row.org_provider_key_id)}
        hasSiblings={(keysPerProvider.get(entry.provider) ?? 0) > 1}
        isNested={isNested}
        hasChevronGutter={hasGroups}
        requiresZeroRetention={requiresZeroRetention}
        isDisabled={isDisabled}
        onToggle={(next) =>
          setOverride.mutate({
            workspaceId,
            keyId: entry.id,
            body: { disabled: !next },
          })
        }
        onPin={(next) =>
          setOverride.mutate({
            workspaceId,
            keyId: entry.id,
            body: { is_default: next },
          })
        }
        onOpenScope={() => setOpen({ kind: "key", id: entry.id, workspaceId })}
        // A refused write is shown by its mutation's own error, in the banner.
        onReset={() => resetKey(entry.row).catch(() => undefined)}
      />
    ) : (
      <HostedRow
        key={entry.id}
        row={entry.row}
        allowedLocations={allowedLocations}
        isNested={isNested}
        hasChevronGutter={hasGroups}
        requiresZeroRetention={requiresZeroRetention}
        isDisabled={isDisabled}
        onToggle={(next) =>
          setHosted.mutate({
            workspaceId,
            provider: entry.row.provider,
            body: { enabled: next, models: entry.row.models },
          })
        }
        onOpenScope={() =>
          setOpen({
            kind: "hosted",
            provider: entry.row.provider,
            workspaceId,
          })
        }
        onReset={() =>
          resetHosted.mutate({ workspaceId, provider: entry.row.provider })
        }
      />
    )

  return (
    <div className="flex flex-col gap-3">
      <ErrorBanner
        error={
          orgKeys.error ??
          overrides.error ??
          hosted.error ??
          setOverride.error ??
          resetOverride.error ??
          removeModel.error ??
          setHosted.error ??
          resetHosted.error ??
          policy.error ??
          setPolicy.error
        }
      />
      {/* Above the list it governs: one switch that holds every row below to
          the organization's or the deployment's declaration. */}
      <SettingsGroup isBounded>
        <SettingRow
          label="Require zero data retention"
          help="Only providers declared to keep no prompt or completion data serve this workspace. Providers configured in config.yml and endpoints this workspace registered cannot be declared, so they are blocked too."
          control={
            <Toggle
              label="Require zero data retention"
              isSelected={requiresZeroRetention}
              isDisabled={!canManage || !policy.data || isPolicyWriting}
              onChange={(next) =>
                setPolicy.mutate({
                  workspaceId,
                  requireZeroDataRetention: next,
                })
              }
            />
          }
        />
        <SettingRow
          label="Allowed data locations"
          help={
            policy.data
              ? `${describeAllowedDataLocations(allowedLocations)}. Hosted providers serve this workspace only where they declare one of these locations.`
              : policy.isPending
                ? "Loading…"
                : "Unknown until this workspace's provider policy can be read."
          }
          control={
            <Button
              size="sm"
              className="min-h-11 md:min-h-9"
              aria-label="Change allowed data locations"
              isDisabled={!canManage || !policy.data || isPolicyWriting}
              onPress={() => {
                setLocations.reset()
                setLocationsOpening((n) => n + 1)
                setLocationsOpenFor(workspaceId)
              }}
            >
              Change
            </Button>
          }
        />
      </SettingsGroup>
      {policy.data ? (
        <AllowedDataLocationsDialog
          key={`${workspaceId}:${locationsOpening}`}
          allowed={allowedLocations}
          providers={hostedRows}
          providersState={
            hosted.isError
              ? "error"
              : hosted.data && !hosted.isFetching
                ? "ready"
                : "loading"
          }
          isOpen={isEditingLocations}
          onClose={() => setLocationsOpenFor("")}
          isPending={setLocations.isPending}
          error={setLocations.error}
          onSave={(allowed) =>
            setLocations.mutate(
              { workspaceId, allowedDataLocations: allowed },
              { onSuccess: () => setLocationsOpenFor("") },
            )
          }
        />
      ) : null}
      {isLoading ? (
        <span className="text-caption">Loading…</span>
      ) : !policy.data ? (
        // A failed read is not "not required": no row can say whether it is
        // blocked, so none is offered until the policy can be read.
        <span className="text-caption">
          The providers are held until this workspace's data retention policy
          can be read.
        </span>
      ) : keyRows.length === 0 && hostedRows.length === 0 ? (
        <span className="text-caption">
          This organization holds no provider keys and the deployment hosts no
          providers for it, so there is nothing for this workspace to use yet.
        </span>
      ) : (
        <>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <span className="text-caption text-subtle">
              {summarizeGroups(groups)}
            </span>
            {isSearchable ? (
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <SearchField
                  label="Search providers"
                  placeholder="Search providers"
                  value={query}
                  onChange={setQuery}
                  className="sm:w-56"
                />
                <Segmented
                  label="Show"
                  options={ACCESS_FILTERS}
                  value={filter}
                  onChange={(next) =>
                    setFilter(
                      ACCESS_FILTERS.find((option) => option.value === next)
                        ?.value ?? "all",
                    )
                  }
                />
              </div>
            ) : null}
          </div>
          {shown.length === 0 ? (
            <span className="text-caption">No provider matches.</span>
          ) : (
            <SettingsGroup isBounded>
              {shown.map((group) =>
                group.entries.length === 1 ? (
                  renderEntry(group.entries[0], false)
                ) : (
                  <DisclosureRow
                    key={group.provider}
                    label={group.provider}
                    chevronPlacement="start"
                    trailing={
                      <span className="text-caption text-subtle">
                        {groupSummary(group.entries)}
                      </span>
                    }
                    // A search or a filter opens every group it leaves, so
                    // what matched is on screen rather than behind a chevron.
                    isOpen={isNarrowed || expanded.has(group.provider)}
                    onToggle={() =>
                      setExpanded((current) => {
                        const next = new Set(current)
                        if (!next.delete(group.provider))
                          next.add(group.provider)
                        return next
                      })
                    }
                  >
                    {/* Tinted, with a rule down its edge, so where the group
                        ends is as plain as where it begins. */}
                    <div className="flex flex-col divide-y divide-border-subtle border-l-2 border-border-strong bg-surface-subtle">
                      {group.entries.map((entry) => renderEntry(entry, true))}
                    </div>
                  </DisclosureRow>
                ),
              )}
            </SettingsGroup>
          )}
        </>
      )}

      {/* Keyed on the source alone: a dialog mounts only while open and seeds
          from what is stored when it does, and a key's save refetches its list
          between writes, which must not remount the dialog mid-save. */}
      {openKey ? (
        <KeyModelsDialog
          key={`${workspaceId}:${openKey.org_provider_key_id}`}
          workspaceId={workspaceId}
          keyId={openKey.org_provider_key_id}
          keyName={keyLabel(openKeyRecord, openKey.org_provider_key_id)}
          provider={openKeyRecord ? keyPrefix(openKeyRecord) : ""}
          allowed={openKey.allowed_models}
          validateEntry={(name) =>
            keyModelProblem(
              name,
              openKeyRecord ? keyPrefix(openKeyRecord) : undefined,
              knownProviders,
            )
          }
          onClose={() => setOpen(closeIn(workspaceId))}
        />
      ) : null}
      {openHosted ? (
        <HostedModelsDialog
          key={`${workspaceId}:${openHosted.provider}`}
          workspaceId={workspaceId}
          row={openHosted}
          onClose={() => setOpen(closeIn(workspaceId))}
        />
      ) : null}
    </div>
  )
}

/** Close the open dialog, unless it now belongs to another workspace. */
function closeIn(workspaceId: string) {
  return (open: OpenScope): OpenScope =>
    open?.workspaceId === workspaceId ? undefined : open
}

/**
 * What a row says about zero data retention: that it is declared, or, where the
 * workspace requires it, that the row is blocked for want of it. Nothing where
 * neither applies, which is the common case.
 */
function retentionNote(isDeclared: boolean, isBlocked: boolean): string {
  if (isBlocked) return "Blocked, not zero data retention"
  return isDeclared ? "Zero data retention" : ""
}

/** Whether a key's row departs from what the workspace inherits. */
function keyIsCustomized(row: WorkspaceProviderKeyOverride): boolean {
  return row.is_default || row.disabled || row.allowed_models.length > 0
}

function KeyRow({
  row,
  orgKey,
  hasSiblings,
  isNested,
  hasChevronGutter,
  requiresZeroRetention,
  isDisabled,
  onToggle,
  onPin,
  onOpenScope,
  onReset,
}: {
  row: WorkspaceProviderKeyOverride
  orgKey: OrgProviderKey | undefined
  hasSiblings: boolean
  isNested: boolean
  hasChevronGutter: boolean
  requiresZeroRetention: boolean
  isDisabled: boolean
  onToggle: (next: boolean) => void
  onPin: (next: boolean) => void
  onOpenScope: () => void
  onReset: () => void
}) {
  const name = keyLabel(orgKey, row.org_provider_key_id)
  const isCustomized = keyIsCustomized(row)
  // What the row cannot say with its switch: a credential the deployment cannot
  // read still resolves, and of several keys only one serves.
  const state = !row.usable
    ? "Unreadable credential"
    : row.is_default
      ? "Always used for this provider"
      : hasSiblings && row.is_effective_default && !row.disabled
        ? "In use"
        : ""
  const isBlocked = requiresZeroRetention && !row.zero_data_retention
  const note = [state, retentionNote(row.zero_data_retention, isBlocked)]
    .filter(Boolean)
    .join(" · ")
  return (
    <AccessRow
      name={name}
      // Under its provider's row, the key's own name is enough.
      displayName={isNested && orgKey ? orgKey.name : name}
      isNested={isNested}
      hasChevronGutter={hasChevronGutter}
      payer="own-key"
      isEnabled={!row.disabled}
      onToggle={onToggle}
      // A disabled key holds no allow-list: the gateway refuses the write and
      // deleted the rows it had.
      scopeLabel={row.disabled ? "" : modelsScopeLabel(row.allowed_models)}
      onOpenScope={onOpenScope}
      isCustomized={isCustomized}
      onReset={onReset}
      note={note}
      isMuted={row.disabled || isBlocked}
      // Held while blocked: nothing set here can serve the workspace until
      // the requirement is lifted or the source is declared.
      isDisabled={isDisabled || isBlocked}
      // Which credential pays rather than what the workspace may call, so it is
      // kept off the row's face, and offered only where there is a choice.
      actions={
        hasSiblings && !row.disabled
          ? [
              {
                id: "pin",
                label: row.is_default
                  ? "Stop always using this key"
                  : "Always use this key",
                onAction: () => onPin(!row.is_default),
              },
            ]
          : []
      }
    />
  )
}

/** Why the workspace's location list refuses a provider, or empty where it does not. */
function locationRefusalNote(
  allowed: readonly string[] | null,
  locations: readonly (string | null)[],
): string {
  if (!refusesProvider(allowed, locations)) return ""
  const declared = locations.filter((location) => location !== null)
  return declared.length === 0
    ? "Refused: location not declared"
    : `Refused: data kept in ${declared.map(dataLocationLabel).join(", ")}`
}

function HostedRow({
  row,
  isNested,
  hasChevronGutter,
  allowedLocations,
  requiresZeroRetention,
  isDisabled,
  onToggle,
  onOpenScope,
  onReset,
}: {
  row: WorkspaceHostedProvider
  isNested: boolean
  hasChevronGutter: boolean
  /** The data locations the workspace accepts; null for any. */
  allowedLocations: readonly string[] | null
  requiresZeroRetention: boolean
  isDisabled: boolean
  onToggle: (next: boolean) => void
  onOpenScope: () => void
  onReset: () => void
}) {
  const locationNote = locationRefusalNote(
    allowedLocations,
    row.data_locations ?? [],
  )
  const isBlocked =
    (requiresZeroRetention && !row.zero_data_retention) || locationNote !== ""
  return (
    <AccessRow
      name={row.provider}
      displayName={isNested ? "Hosted" : row.provider}
      isNested={isNested}
      hasChevronGutter={hasChevronGutter}
      payer="deployment"
      isEnabled={row.enabled}
      onToggle={onToggle}
      scopeLabel={row.enabled ? modelsScopeLabel(row.models) : ""}
      onOpenScope={onOpenScope}
      isCustomized={row.customized}
      onReset={onReset}
      // The organization's own key wins over the deployment's wherever both
      // could serve, so this row decides nothing until that key goes.
      note={[
        row.served_by_own_key ? "Not used, your key serves this provider" : "",
        retentionNote(
          row.zero_data_retention,
          requiresZeroRetention && !row.zero_data_retention,
        ),
        locationNote,
      ]
        .filter(Boolean)
        .join(" · ")}
      isMuted={row.served_by_own_key || !row.enabled || isBlocked}
      isDisabled={isDisabled || isBlocked}
    />
  )
}
