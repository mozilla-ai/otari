import type { ReactNode } from "react"

import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import { providerInstanceOf } from "@/features/models/modelKey"
import { useModels } from "@/shared/api/models"

// The same ceiling `ModelComboBox` uses, and for the same reason: past this the
// popover is a wall of text nobody scrolls. A deployment configures far fewer
// provider instances than a provider serves models, so it is a guard rather
// than a routine cap.
const MAX_VISIBLE = 50

/**
 * Picker over the provider instances the caller can actually route to.
 *
 * The sibling of `ProviderComboBox`, and not the same question: that one offers
 * the provider *types* any-llm knows, for a form about to create a credential,
 * while this one offers the instance names that already exist, for a form that
 * refers to one. A ceiling narrowed to `openai-eu` binds nothing unless an
 * instance is actually called that.
 *
 * Read off the model catalog rather than `useProviders()`, which is the list
 * this is really about: the whole `/v1/providers` router is
 * `require_deployment_operator`, and the one form that asks this question is
 * shown only to an organization admin who does not operate the deployment
 * (`BudgetsPage` routes an operator to the deployment page instead), so that
 * read would be refused every time it was made. `GET /v1/models` answers any
 * signed-in caller and is already narrowed to what that caller could route to,
 * and the prefix of a catalog id is the resolved instance a budget scope is
 * matched on (`BudgetScopeRequest.provider_instance`), so it is both readable
 * and the right set.
 *
 * Free text stands, as everywhere else here. `ScopedBudget.provider_key_id`
 * carries an instance name rather than a foreign key precisely because an
 * instance may be configured in `config.yml` and have no row to point at, so a
 * name this list has not seen is a legitimate answer rather than a typo by
 * definition.
 */
export function ProviderInstanceComboBox({
  label,
  value,
  onChange,
  description,
  placeholder,
  isRequired,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  description?: ReactNode
  placeholder?: string
  isRequired?: boolean
}) {
  const catalog = useModels()

  const query = value.trim().toLowerCase()
  const instances = new Set(
    (catalog.data?.data ?? []).flatMap((model) => {
      const instance = providerInstanceOf(model.id)
      return instance ? [instance] : []
    }),
  )
  // Sorted on its own rather than inheriting the catalog's order, which is by
  // model key: that clusters instances but leaves one trailing far down the
  // list whenever a model of another instance sorts between two of its own.
  const all: ComboBoxOption[] = [...instances]
    .sort((a, b) => a.localeCompare(b))
    .map((instance) => ({ value: instance, label: instance }))
  const hits = query
    ? all.filter((option) => option.value.toLowerCase().includes(query))
    : all
  const visible = hits.slice(0, MAX_VISIBLE)
  const isSourceEmpty = all.length === 0

  const hint = ((): ReactNode => {
    if (catalog.isLoading) return "Loading the model catalog…"
    // A failed read leaves the same empty list as a deployment with no provider,
    // and the two want opposite things said.
    if (catalog.isError) {
      return "Could not read the model list. Type the instance name directly."
    }
    if (hits.length > visible.length) {
      return `Showing ${visible.length} of ${hits.length} matches. Keep typing to narrow them.`
    }
    return description
  })()

  return (
    <ComboBoxField
      label={label}
      value={value}
      onChange={onChange}
      options={visible}
      description={hint}
      placeholder={placeholder}
      isRequired={isRequired}
      // A list of what this deployment happens to serve, never a whitelist.
      allowsCustomValue
      isSourceEmpty={isSourceEmpty}
      emptyMessage={
        catalog.isLoading
          ? "Looking for configured providers…"
          : catalog.isError
            ? "The model list could not be read, so there is nothing to offer. Type the instance name as the gateway spells it."
            : "No provider serves a model here yet. The instances you can route to appear here; until then, type the name."
      }
      noMatchesMessage="No provider matches. Type an instance name to use it anyway."
    />
  )
}
