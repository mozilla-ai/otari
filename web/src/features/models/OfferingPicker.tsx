import { type ReactNode, useRef, useState } from "react"

import type { CatalogModelSummary } from "@/client"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { Select } from "@/design-system/forms/Select"
import {
  findOfferingOwner,
  offeringChoices,
  preferredSelector,
} from "@/features/models/offerings"
import { useSelectorOwner } from "@/features/models/useSelectorOwner"
import { useCatalogSearch } from "@/shared/api/models"
import { providerDisplayName } from "@/shared/helpers/providers"
import { useDebounced } from "@/shared/hooks/useDebounced"

// Past this the popover is a wall nobody scrolls; typing narrows faster.
const MAX_VISIBLE = 50
// How many providers a row names before counting the rest.
const NAMED_PROVIDERS = 3

function providersHint(model: CatalogModelSummary): string {
  const names = model.providers.map(providerDisplayName)
  const shown = names.slice(0, NAMED_PROVIDERS).join(", ")
  const rest = names.length - NAMED_PROVIDERS
  return rest > 0 ? `${shown} +${rest} more` : shown
}

function modelOption(model: CatalogModelSummary): ComboBoxOption {
  return { value: model.id, label: model.name, hint: providersHint(model) }
}

/**
 * A model from the catalog, then one of its providers, saved as the
 * `provider:model` selector of that offering.
 *
 * Rows are models rather than selectors, so one model served by several
 * providers is one row with the providers named under it, and the provider is
 * chosen beside it. Grouping in the list itself is not available: React Aria's
 * sectioned collections break picking under React 19.2's development build
 * (see `ModelComboBox`).
 *
 * Free text still stands, because the catalog is a shortcut rather than a
 * whitelist: a typed selector is saved exactly as typed.
 *
 * `actions` joins the control row, and the row's message renders under the
 * whole row (`web/design/forms.md`, "Control rows").
 */
export function OfferingPicker({
  id,
  label,
  providerLabel = "Provider",
  value,
  onChange,
  description,
  errorMessage = "",
  isRequired,
  autoFocus,
  actions,
}: {
  /** Unique on the page; names the message line under the row. */
  id: string
  label: string
  providerLabel?: string
  /** A selector, or text being typed. */
  value: string
  onChange: (selector: string) => void
  description?: ReactNode
  /** Why the value cannot be saved, shown under the field; "" when it can. */
  errorMessage?: string
  isRequired?: boolean
  autoFocus?: boolean
  actions?: ReactNode
}) {
  const [query, setQuery] = useState("")
  // The field reports a keystroke and a pick through the same `onChange`, a
  // keystroke right after its query. The query it last reported tells them
  // apart within that one event.
  const lastQuery = useRef("")
  const [picked, setPicked] = useState<CatalogModelSummary>()
  const isTyping = query !== ""

  const term = useDebounced(query.trim())
  const list = useCatalogSearch(term, MAX_VISIBLE + 1)
  const listed = list.data?.models ?? []
  const known = findOfferingOwner([picked, ...listed], value)
  const looked = useSelectorOwner(value, known === undefined && !isTyping)
  const owner = known ?? looked

  const visible = listed.slice(0, MAX_VISIBLE)
  const models =
    owner && !visible.some((model) => model.id === owner.id)
      ? [owner, ...visible]
      : visible
  const choices = owner ? offeringChoices(owner) : []

  const choose = (next: string) => {
    if (next === lastQuery.current) {
      onChange(next)
      return
    }
    const model = models.find((entry) => entry.id === next)
    if (model === undefined) {
      onChange(next)
      return
    }
    setPicked(model)
    // Picking the model already chosen keeps its provider.
    onChange(
      model.selectors.includes(value.trim())
        ? value.trim()
        : preferredSelector(model),
    )
  }

  const total = Math.max(listed.length, list.data?.count ?? 0)
  const hint = !isTyping
    ? undefined
    : list.isLoading
      ? "Loading the model catalog…"
      : list.isError
        ? "Could not read the model list. Type the selector directly."
        : total > MAX_VISIBLE
          ? `Showing ${MAX_VISIBLE} of ${total} models. Keep typing to narrow them.`
          : undefined
  const message = hint ?? description
  const messageId = `${id}-message`

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex flex-wrap items-end gap-3">
        <ComboBoxField
          className="min-w-48 flex-1"
          label={label}
          // While typing, the text is the value; otherwise a resolved selector
          // shows as its model.
          value={isTyping || owner === undefined ? value : owner.id}
          onChange={choose}
          onQueryChange={(next) => {
            lastQuery.current = next
            setQuery(next)
          }}
          options={models.map(modelOption)}
          placeholder="Search models"
          autoFocus={autoFocus}
          isRequired={isRequired}
          isInvalid={errorMessage !== ""}
          errorMessage={errorMessage}
          allowsCustomValue
          // Not "focus": an open list aria-hides the rest of the dialog.
          menuTrigger="input"
          // A chosen model shows as its name, so typing replaces it.
          shouldSelectOnFocus
          describedBy={message ? messageId : undefined}
          isSourceEmpty={!term && listed.length === 0}
          emptyMessage={
            list.isLoading
              ? "Looking for models…"
              : list.isError
                ? "The model list could not be read. Type the selector; it is sent exactly as typed either way."
                : "No models to offer yet. Models this deployment serves appear here once a provider is configured; until then, type the selector."
          }
          noMatchesMessage="No model matches. Type a provider:model selector to use it anyway."
        />
        {choices.length > 0 ? (
          <Select
            className="w-full sm:w-44"
            label={providerLabel}
            value={value.trim()}
            options={choices}
            onChange={(selector) => {
              lastQuery.current = ""
              setQuery("")
              onChange(selector)
            }}
          />
        ) : null}
        {actions}
      </div>
      {message ? (
        <FieldMessages shouldReserve={false}>
          <span id={messageId} className="text-muted">
            {message}
          </span>
        </FieldMessages>
      ) : null}
    </div>
  )
}
