import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react"
import { type ReactNode, useMemo, useState } from "react"
import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { providerInstanceOf } from "@/features/models/modelKey"
import { useModels } from "@/shared/api/models"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useDeploymentProviderNames } from "@/shared/api/providers"
import { useAliases } from "@/shared/api/routing"

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

interface CatalogOption {
  // The canonical value stored on the allow-list (instance:model or instance:*).
  id: string
  // What the operator sees in the dropdown.
  label: string
}

const MAX_VISIBLE = 50

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
  const aliases = useAliases()
  const [mode, setMode] = useState<Mode>(modeOf(initial))
  const [entries, setEntries] = useState<string[]>(initial ?? [])
  const [query, setQuery] = useState("")

  // The pickable catalog: a wildcard per provider instance, every model, and
  // each alias resolved to its target. It comes from GET /models, which is
  // scoped to the caller's organization (its own providers and the deployment's
  // managed models), not from the operator-only discovery or provider-config
  // reads, which an organization admin cannot see. Deduped by value, so an alias
  // whose target is already listed does not appear twice. Every stored entry is
  // a canonical selector the backend will accept, so names that resolve through
  // no instance are left out. An operator also gets a wildcard for every
  // configured provider, which covers one whose models GET /models cannot list
  // (discovery failed or unsupported, nothing priced).
  const catalog = useMemo<CatalogOption[]>(() => {
    const ids = (models.data?.data ?? [])
      .map((model) => model.id)
      .filter((id) => providerInstanceOf(id) !== undefined)
      .sort((a, b) => a.localeCompare(b))
    const instances = [
      ...new Set([
        ...providers,
        ...ids.map((id) => providerInstanceOf(id) as string),
      ]),
    ].sort((a, b) => a.localeCompare(b))
    const candidates: CatalogOption[] = [
      ...instances.map((instance) => ({
        id: `${instance}:*`,
        label: `${instance}:*  ·  all ${instance} models`,
      })),
      ...ids.map((id) => ({ id, label: id })),
      ...(aliases.data ?? []).map((alias) => ({
        id: alias.target,
        label: `${alias.name}  ·  alias`,
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
  }, [models.data, providers, aliases.data])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return catalog
      .filter((option) => !entries.includes(option.id))
      .filter(
        (option) =>
          !q ||
          option.id.toLowerCase().includes(q) ||
          option.label.toLowerCase().includes(q),
      )
      .slice(0, MAX_VISIBLE)
  }, [catalog, entries, query])

  const emit = (nextMode: Mode, nextEntries: string[]) => {
    if (nextMode === "any") onChange(undefined, true)
    else if (nextMode === "block") onChange([], true)
    else onChange(nextEntries, nextEntries.length > 0)
  }

  const chooseMode = (next: Mode) => {
    setMode(next)
    emit(next, entries)
  }

  const addEntry = (id: string) => {
    const next = entries.includes(id) ? entries : [...entries, id]
    setEntries(next)
    setQuery("")
    emit("only", next)
  }

  const removeEntry = (id: string) => {
    const next = entries.filter((entry) => entry !== id)
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
      ? "No providers or models discovered yet. Configure a provider first, then scope this key."
      : null
  const showPicker = catalog.length > 0 || models.isPending

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
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-1.5">
            {entries.length === 0 ? (
              <span className="text-caption">
                Pick at least one model below, or choose “Block all”.
              </span>
            ) : (
              entries.map((entry) => (
                <DismissChip
                  key={entry}
                  value={entry}
                  onDismiss={() => removeEntry(entry)}
                  dismissLabel={`Remove ${entry}`}
                />
              ))
            )}
          </div>
          {catalogHint ? (
            <span className="text-caption">{catalogHint}</span>
          ) : null}
          {showPicker ? (
            <ComboBox.Root
              allowsEmptyCollection
              menuTrigger="input"
              inputValue={query}
              onInputChange={setQuery}
              selectedKey={null}
              onSelectionChange={(key) => {
                if (key != null) addEntry(String(key))
              }}
              // Cap the width so the field and its dropdown trigger stay within
              // easy reach instead of stretching across a wide form.
              className="flex max-w-md flex-col gap-1"
            >
              <ComboBox.InputGroup>
                <Input
                  aria-label="Add a model"
                  placeholder="Search providers, models, aliases…"
                  autoComplete="off"
                />
                <ComboBox.Trigger />
              </ComboBox.InputGroup>
              <ComboBox.Popover>
                <ListBox
                  items={visible}
                  className="max-h-72 overflow-auto"
                  renderEmptyState={() => (
                    <ComboBoxEmpty
                      isSourceEmpty={catalog.every((option) =>
                        entries.includes(option.id),
                      )}
                      emptyMessage={
                        catalog.length === 0
                          ? "Loading the model catalog…"
                          : "Everything available is already on the list."
                      }
                      noMatchesMessage="Nothing matches what you typed."
                    />
                  )}
                >
                  {(option: CatalogOption) => (
                    <ListBoxItem id={option.id} textValue={option.label}>
                      {option.label}
                    </ListBoxItem>
                  )}
                </ListBox>
              </ComboBox.Popover>
            </ComboBox.Root>
          ) : null}
        </div>
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
