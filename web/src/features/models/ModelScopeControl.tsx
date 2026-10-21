import { type ReactNode, useMemo, useState } from "react"
import { ControlField } from "@/design-system/forms/FieldMessages"
import {
  MultiSelect,
  type MultiSelectOption,
} from "@/design-system/forms/MultiSelect"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { providerInstanceOf, providerModelId } from "@/features/models/modelKey"
import { canManage } from "@/features/organization/roles"
import { useModels } from "@/shared/api/models"
import {
  useDeploymentOperator,
  useOrganizationContext,
  useOrgProviderKeys,
} from "@/shared/api/organizations"
import { useDeploymentProviderNames } from "@/shared/api/providers"
import { useAliases } from "@/shared/api/routing"
import { providerDisplayName } from "@/shared/helpers/providers"

// The per-key model access-list is a tri-state:
//   undefined -> "any"   (unrestricted, the default; `null` on the wire)
//   []        -> "block" (deny all)
//   list      -> "only"  (restrict to these entries)
// A bare multi-select cannot tell "any" from "block" (both look empty), so the
// mode is an explicit 3-way choice. "Only selected" with no entries is an
// INCOMPLETE form (Save disabled), never a silent deny-all.
type Mode = "any" | "only" | "block"

function modeOf(allowedModels: string[] | undefined): Mode {
  if (!allowedModels) return "any"
  return allowedModels.length === 0 ? "block" : "only"
}

// The control is reused for two layers of the same allow-list grammar: a user's
// default and a key's (narrower) override. The wording differs between them, so
// the heading, help text, and the "any" mode label are parameterized. "any" means
// `null` on the wire, converted where it arrives: unrestricted for a user, but
// "inherit the owner's default" for a key (a key with no list of its own falls
// back to its user).
export function ModelScopeControl({
  initial,
  onChange,
  title = "Model access",
  description,
  anyLabel = "Any model",
}: {
  initial: string[] | undefined
  onChange: (value: string[] | undefined, valid: boolean) => void
  title?: ReactNode
  description?: ReactNode
  anyLabel?: string
}) {
  const models = useModels()
  const { isOperator, isSettled } = useDeploymentOperator()
  // Operator-only: an organization admin is refused it.
  const providers = useDeploymentProviderNames(isSettled && isOperator)
  // Owner/admin-only, like the operator read above.
  const orgKeys = useOrgProviderKeys(canManage(useOrganizationContext().data))
  const aliases = useAliases()
  const [mode, setMode] = useState<Mode>(modeOf(initial))
  const [entries, setEntries] = useState<string[]>(initial ?? [])

  // The pickable catalog: a wildcard per provider instance, every model, and
  // each alias resolved to its target. It comes from GET /models, which is
  // scoped to the caller's organization (its own providers and the deployment's
  // managed models), not from the operator-only discovery or provider-config
  // reads, which an organization admin cannot see. Deduped by value, so an alias
  // whose target is already listed does not appear twice. Every stored entry is
  // a canonical selector the backend will accept, so names that resolve through
  // no instance are left out. An operator also gets a wildcard for every
  // configured provider, which covers one whose models GET /models cannot list
  // (discovery failed or unsupported, nothing priced). An organization owner or
  // admin likewise gets one for each of its own provider keys: a key nobody has
  // refreshed offers no models, so GET /models lists nothing for it.
  // Each option is named for a reader (the model, or "All OpenAI models") and
  // keeps the stored selector findable in its hint, since that is what a key's
  // allow-list and the gateway's refusals spell.
  const catalog = useMemo<MultiSelectOption[]>(() => {
    const ids = (models.data?.data ?? [])
      .map((model) => model.id)
      .filter((id) => providerInstanceOf(id) !== undefined)
      .sort((a, b) => a.localeCompare(b))
    const instances = [
      ...new Set([
        ...providers,
        ...(orgKeys.data ?? [])
          .filter((key) => !key.archived_at)
          .map((key) => key.provider),
        ...ids.map((id) => providerInstanceOf(id) as string),
      ]),
    ].sort((a, b) => a.localeCompare(b))
    const providerOf = (id: string) =>
      providerDisplayName(providerInstanceOf(id) ?? "")
    const candidates: MultiSelectOption[] = [
      ...instances.map((instance) => ({
        id: `${instance}:*`,
        label: `All ${providerDisplayName(instance)} models`,
        hint: `${instance}:* · includes models added later`,
      })),
      ...ids.map((id) => ({
        id,
        label: providerModelId(id),
        hint: providerOf(id),
      })),
      ...(aliases.data ?? []).map((alias) => ({
        id: alias.target,
        label: alias.name,
        hint: `Alias for ${providerModelId(alias.target)} · ${providerOf(alias.target)}`,
      })),
    ]
    // First label wins, so a model keeps its own name over the alias that
    // resolves to it.
    const seen = new Set<string>()
    return candidates.filter((option) => {
      if (!option.id || seen.has(option.id)) return false
      seen.add(option.id)
      return true
    })
  }, [models.data, providers, orgKeys.data, aliases.data])

  const emit = (nextMode: Mode, nextEntries: string[]) => {
    if (nextMode === "any") onChange(undefined, true)
    else if (nextMode === "block") onChange([], true)
    else onChange(nextEntries, nextEntries.length > 0)
  }

  const chooseMode = (next: Mode) => {
    setMode(next)
    emit(next, entries)
  }

  const pick = (next: string[]) => {
    setEntries(next)
    emit("only", next)
  }

  // The shared tab, which is what every segmented choice in the product uses. A
  // raised chip (`bg-surface` plus an elevation that is `none` now shadows are
  // zeroed) leaves the selected segment reading 1.1:1 against its own track.
  const modeButton = (value: Mode, label: string) => (
    <Tab
      key={value}
      isActive={mode === value}
      onPress={() => chooseMode(value)}
    >
      {label}
    </Tab>
  )

  // A failed read and an empty one both leave little or nothing to pick; only
  // the second means there is nothing to configure. A failed read is said even
  // when aliases or provider wildcards still fill the list, which is then
  // incomplete.
  const catalogHint = models.isError
    ? "Could not load the model catalog. Reload to try again."
    : !models.isPending && catalog.length === 0
      ? "No providers or models discovered yet. Add a provider first."
      : null

  return (
    <div className="flex flex-col gap-3">
      <ControlField
        label={title}
        description={
          description ??
          "Which models this key may list and call. The master key is never restricted, so blocking a key cannot lock you out of the dashboard."
        }
      />
      {/* No track: a tab row is bare, and the selected segment's own fill is
          what marks it. A track plus a fill was two ways of saying one thing. */}
      <TabRow>
        {modeButton("any", anyLabel)}
        {modeButton("only", "Only selected")}
        {modeButton("block", "Block all")}
      </TabRow>

      {mode === "block" ? (
        <div className="flex items-center gap-2 border-y border-border py-2 text-xs text-muted">
          Blocked from <strong>every</strong> model until you change this
          access.
        </div>
      ) : null}

      {mode === "only" ? (
        <MultiSelect
          label="Allowed models"
          options={catalog}
          value={entries}
          onChange={pick}
          description={
            catalogHint ??
            (entries.length === 0
              ? "Pick at least one model, or choose “Block all”."
              : undefined)
          }
          searchPlaceholder="Search providers, models, aliases…"
          emptyMessage={
            models.isPending ? "Loading the model catalog…" : catalogHint
          }
        />
      ) : null}
    </div>
  )
}

// A compact label describing a key's access, for the table row. Deliberately not
// a count: an entry like `openai:*` is one entry but many models, so a number
// would mislead. The exact entries are surfaced on hover / in the edit form.
export function accessLabel(allowedModels: string[] | undefined): {
  text: string
  tone: "muted" | "normal" | "danger"
} {
  // Absent is "not restricted", which an empty list cannot say: `[]` is a real
  // answer meaning no model at all. The wire spells absent `null`, so callers
  // convert it where it arrives.
  if (!allowedModels) return { text: "All models", tone: "muted" }
  if (allowedModels.length === 0) return { text: "No models", tone: "danger" }
  return { text: "Selected models", tone: "normal" }
}
