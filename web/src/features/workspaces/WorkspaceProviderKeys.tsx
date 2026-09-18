import { useState } from "react"

import type { OrgProviderKey, WorkspaceProviderKeyOverride } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { useModels } from "@/shared/api/models"
import { useOrgProviderKeys } from "@/shared/api/organizations"
import {
  useAddWorkspaceProviderKeyModel,
  useRemoveWorkspaceProviderKeyModel,
  useResetWorkspaceProviderKeyOverride,
  useSetWorkspaceProviderKeyOverride,
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
 * The allow-list is per (workspace, key) and narrows rather than widens: no
 * entries means every model that key serves. It is what
 * `services/tenancy/organization_model_access.py` reads when it builds a
 * session's catalog scope, so emptying a workspace's catalog here also takes
 * `/models` down to nothing for it, and with it the first-request setup guide
 * that asks the same endpoint whether a call could succeed (#980).
 */

/** What a row's picker is showing, which is the pair of stored flags named. */
type Departure = "inherited" | "pinned" | "disabled"

// What the gateway does with this key, rather than what the row is called.
// "Workspace default" and then "Pinned as default" both failed the same way:
// the setting is about which *credential* serves a request, and a reader asked
// what it meant for a model (#2106). A sentence cannot be read that way.
const DEPARTURE_OPTIONS: { value: Departure; label: string }[] = [
  { value: "inherited", label: "Follow the organization" },
  { value: "pinned", label: "Always use this key" },
  { value: "disabled", label: "Never use this key" },
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
  // produces (both reads cover the same non-archived set), so this is a
  // fallback rather than a case. It says what the row is before it says which
  // one: a bare UUID named nothing a person could act on, where "Unknown key"
  // plus a fragment is both readable and enough to find the row again (#2106).
  if (!key) return `Unknown key ${keyId.slice(0, 8)}`
  return `${key.provider} / ${key.name}`
}

/** At most this many suggestions in the popover, so a broad catalog stays scannable. */
const MODEL_SUGGESTION_LIMIT = 50

/**
 * Why the suggestion list is empty, which the list itself cannot say.
 *
 * "No catalog entry for this provider" is a real answer and a refused catalog
 * read is not, and both leave the same empty popover behind.
 */
type CatalogState = "pending" | "failed" | "ready"

function ModelAllowList({
  workspaceId,
  keyId,
  keyName,
  provider,
  models,
  suggestions,
  knownProviders,
  catalogState,
}: {
  workspaceId: string
  keyId: string
  keyName: string
  /** The key's own provider, or undefined for a key the organization list does not carry. */
  provider: string | undefined
  /** This key's allow-list. Empty is a real answer: every model the key serves. */
  models: string[]
  /** Every model the catalog lists for this provider, already narrowed to it. */
  suggestions: string[]
  /** Every provider this deployment names, which is what a typed prefix is matched against. */
  knownProviders: ReadonlySet<string>
  catalogState: CatalogState
}) {
  const add = useAddWorkspaceProviderKeyModel()
  const remove = useRemoveWorkspaceProviderKeyModel()
  const [draft, setDraft] = useState("")
  const pending = add.isPending || remove.isPending
  const trimmed = draft.trim()

  // The catalog spells a model `provider:model` and the allow-list stores the
  // bare name, so the id somebody copies off the models page is the one entry
  // that looks right and narrows the workspace to a model that does not exist.
  // Named rather than silently stripped: the two spellings mean different
  // things elsewhere, and a control that quietly rewrote one would teach that
  // they are interchangeable.
  //
  // Matched against the providers this deployment actually names rather than
  // against any text before a colon, because a colon is not always a prefix:
  // `llama3:8b` is a whole model name on Ollama, and a blanket rule would
  // refuse it. Only the colon form is checked at all, since a slash is part of
  // a real name on a provider that routes to others (`openrouter/auto`).
  const separator = trimmed.indexOf(":")
  const typedPrefix = separator === -1 ? undefined : trimmed.slice(0, separator)
  const isPrefixed =
    typedPrefix !== undefined && knownProviders.has(typedPrefix)
  const isDuplicate = models.includes(trimmed)
  const invalidReason = isPrefixed
    ? typedPrefix === provider
      ? `Name the model without its "${typedPrefix}:" prefix.`
      : `"${typedPrefix}:" names another provider. Name a model this key serves.`
    : isDuplicate
      ? "This model is already allowed on this key."
      : undefined

  // What the list could ever offer, which is what tells "this provider has no
  // catalog entry" from "nothing matches what you typed".
  const available = suggestions.filter((model) => !models.includes(model))
  // Four ways to have nothing left to suggest, and only the last is a fact
  // about the provider rather than about this workspace or the read.
  const emptyMessage =
    catalogState === "pending"
      ? "Reading the catalog…"
      : catalogState === "failed"
        ? "The model catalog could not be read. Type the model id as the provider spells it."
        : suggestions.length > 0
          ? "Every model the catalog lists for this provider is already allowed."
          : "No catalog entry for this provider. Type the model id as the provider spells it."
  const options: ComboBoxOption[] = available
    .filter((model) => model.toLowerCase().includes(trimmed.toLowerCase()))
    .slice(0, MODEL_SUGGESTION_LIMIT)
    .map((model) => ({ value: model, label: model }))

  return (
    <div className="flex flex-col gap-2">
      <ErrorBanner error={add.error ?? remove.error} />
      {models.length === 0 ? (
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
      {/* Suggestions rather than a whitelist, and free text is what commits:
          `/models` is filtered through these very restrictions for a plain
          workspace manager, so a closed picker would stop offering exactly the
          models this control exists to add back. An organization owner or admin
          reads it unfiltered (`resolve_session_catalog_scope` answers them from
          the organization's keys), which is who this form usually belongs to. */}
      <div className="flex flex-wrap items-end gap-2">
        <ComboBoxField
          // Short and visible, with the key only in the accessible name: at one
          // key the repetition is invisible and at five it is the loudest thing
          // in the form (#2106). Said twice rather than continued, because the
          // name is computed by concatenating the label's text with no
          // separator, which runs "Allow a model" into what follows it.
          label={
            <>
              <span aria-hidden="true">Allow a model</span>
              <span className="sr-only">Allow a model on {keyName}</span>
            </>
          }
          value={draft}
          onChange={setDraft}
          options={options}
          allowsCustomValue
          menuTrigger="input"
          // An example from this key's own provider: a hardcoded `gpt-4o` on an
          // Anthropic key is an instruction to type the wrong thing.
          placeholder={suggestions[0] ?? "model name"}
          isInvalid={invalidReason !== undefined}
          errorMessage={invalidReason}
          reserveMessage={false}
          isSourceEmpty={available.length === 0}
          emptyMessage={emptyMessage}
          noMatchesMessage="No catalog entry matches. Type the model id to allow it anyway."
        />
        <Button
          size="sm"
          variant="ghost"
          // Named per key, as the picker above it is: the form holds one of
          // these per key, and "Allow" alone names all of them the same.
          aria-label={`Allow a model on ${keyName}`}
          isDisabled={pending || trimmed === "" || invalidReason !== undefined}
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
  const catalog = useModels()
  const setOverride = useSetWorkspaceProviderKeyOverride()
  const resetOverride = useResetWorkspaceProviderKeyOverride()

  const catalogState: CatalogState = catalog.isPending
    ? "pending"
    : catalog.isError
      ? "failed"
      : "ready"

  const byId = new Map((orgKeys.data ?? []).map((key) => [key.id, key]))
  const pending = setOverride.isPending || resetOverride.isPending

  // `provider:model`, which is how the catalog names an entry and how a model
  // restriction does not. Grouped once rather than per key, since an
  // organization's keys share few providers.
  const catalogByProvider = (catalog.data?.data ?? []).reduce(
    (groups, model) => {
      const separator = model.id.indexOf(":")
      if (separator === -1) return groups
      const provider = model.id.slice(0, separator)
      const name = model.id.slice(separator + 1)
      const entries = groups.get(provider)
      if (entries) entries.push(name)
      else groups.set(provider, [name])
      return groups
    },
    new Map<string, string[]>(),
  )

  // Every provider this deployment names, from the catalog and from the
  // organization's own keys, which is what tells a pasted `openai:` prefix from
  // a model whose real name carries a colon.
  const knownProviders = new Set([
    ...catalogByProvider.keys(),
    ...(orgKeys.data ?? []).map((key) => key.provider),
  ])

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
        {/* `text-title` is the role for a group of fields, which is what this
            is inside the edit dialog. On `text-body` it was the same 14/400 as
            the prose under it, so nothing said a new section had started. */}
        <span className="text-title">Provider keys</span>
        <span className="max-w-md text-caption">
          This workspace inherits every provider key the organization holds.
          Where a provider has more than one, say which of them this workspace
          uses; that choice is about the credential a request is sent with, not
          about models. Narrowing a key to named models is the separate control
          below it, and it hides every other model of that provider from this
          workspace, including from its model list.
        </span>
      </div>
      {/* The rest of this form waits for Save, and this section does not. An
          admin who sets a key to Disabled and then presses Cancel otherwise has
          every reason to expect it undone (#2106). */}
      <InfoBanner>
        Each change in this section is saved as you make it, so Cancel does not
        undo one.
      </InfoBanner>
      <ErrorBanner
        error={
          orgKeys.error ??
          overrides.error ??
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
        // Ruled rather than boxed, which is how this design system separates a
        // repeated block: at one key the rule says nothing and at five it is
        // what tells one key's controls from the next one's.
        <ul className="flex flex-col divide-y divide-border-subtle">
          {rows.map((row) => {
            const keyId = row.org_provider_key_id
            const key = byId.get(keyId)
            const name = keyLabel(key, keyId)
            const departure = departureOf(row)
            return (
              <li key={keyId} className="flex flex-col gap-2 py-3 first:pt-0">
                {/* The name is 12px mono and a field is 36px, so the two never
                    sat on one line without one dwarfing the other. Stacked, the
                    name pairs with the marker beside it and the control reads
                    as the answer to it. */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-mono-caption text-foreground">
                    {name}
                  </span>
                  {/* Which key the provider actually resolves to, which the
                      flags alone do not say: an unpinned key is still the one
                      serving this workspace when it is the organization's
                      default, or the only one that provider has. */}
                  {row.is_effective_default ? (
                    <span className="text-caption">In use</span>
                  ) : null}
                  {/* "In use" is about resolution and says nothing about whether
                      the credential can be read. An unreadable key still
                      resolves, so without this the row reads as the one serving
                      the workspace while every request through it fails. */}
                  {row.usable ? null : (
                    <span className="text-caption text-danger">
                      Unreadable credential
                    </span>
                  )}
                </div>
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
                    provider={key?.provider}
                    models={row.allowed_models}
                    suggestions={
                      key ? (catalogByProvider.get(key.provider) ?? []) : []
                    }
                    knownProviders={knownProviders}
                    catalogState={catalogState}
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
